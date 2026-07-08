<!--
SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
SPDX-License-Identifier: MIT
-->

# Node-skill package co-install (omnimarket) — OMN-13829

## Symptom

Node-backed skill CLIs run from the omnibase_infra venv fail:

```console
$ onex run-node node_pr_lifecycle_orchestrator
Unknown node 'node_pr_lifecycle_orchestrator'. Known nodes: node_aislop_sweep, ...
```

## Root cause (verified)

The omnibase_infra venv held a **manually `uv pip install`-ed omnimarket `0.1.0`
snapshot** (git `main` @ `5f88d0d`, 2026-04-12 — see the package's
`direct_url.json`). omnimarket is **not** declared in `pyproject.toml` / `uv.lock`,
so `uv sync` never installs or refreshes it; the April snapshot simply went stale.
It predates `node_pr_lifecycle_orchestrator` and the current node set (25 nodes in
the snapshot vs 345+ today), so the `onex.nodes` entry-point lookup returns
`Unknown node`.

## Why omnimarket is co-installed, not a `pyproject` dependency (the canonical design)

This is **not** a limitation to fix upstream — it is the repo layering boundary.
Repo layering is `compat -> core -> spi -> infra`, and **omnimarket sits ABOVE
infra** (omnimarket depends on `omnibase-infra >=0.38.3,<0.39.0`). Declaring
omnimarket as an omnibase_infra dependency would **invert the layer graph** and
publish a dependency cycle in the omnibase-infra wheel. The direction is fixed by
the architecture: **omnibase_infra must not depend on omnimarket.**

Instead, the `onex` CLI shipped in omnibase_infra composes market nodes at
**runtime** via co-installed `onex.nodes` entry-points. omnimarket is a *provider*
discovered at runtime, never a build/lock dependency. `scripts/install-node-skill-package.sh`
is the **canonical co-install** step that places that provider into an operator's
venv. It installs `--no-deps` because the infra venv already supplies every
lower-layer omni dependency; `--no-deps` layers the provider on top without
re-resolving (or downgrading) the layer beneath it.

This matches the repo's existing runtime contract — the skipped test in
`tests/unit/runtime/test_event_bus_subscriber_container_resolution.py` asserts
*"omnimarket is no longer an omnibase_infra runtime dependency"*: the runtime
composes market; it does not depend on it.

> For completeness, a `pyproject` dependency is also mechanically impossible today
> (published PyPI wheels ≤0.4.6 pin `omnibase-core<0.45.0` / `spi<0.22.0`; a `uv
> lock` against the dev rev hits the circular back-reference). But even if those
> pins were clean, the co-install would remain the correct mechanism because of
> the layering boundary above. OMN-13836 already cleaned omnimarket's `[tool.uv]`
> overrides (`omnibase-core>=0.46.1`), so the pinned rev below carries clean
> metadata.

## The co-install (operator-run — mutates the venv)

`scripts/install-node-skill-package.sh` encapsulates the vetted install. It is
gated behind `--execute`; without it, it prints the plan only. It never runs on
import or in CI — nothing invokes it automatically.

```bash
# from the omnibase_infra repo root, with the target venv active
scripts/install-node-skill-package.sh                    # dry-run plan
scripts/install-node-skill-package.sh --execute          # apply to $VIRTUAL_ENV
scripts/install-node-skill-package.sh --execute /path/to/venv/bin/python
# bump the rev without editing the script:
OMNIMARKET_REF=<newer-sha> scripts/install-node-skill-package.sh --execute
```

The ref is **resolved dynamically** (OMN-14060) from omnimarket's live `dev` HEAD
via `git ls-remote`, falling back to the local canonical clone at
`$OMNI_HOME/omnimarket` when offline. A hand-edited SHA literal here goes stale
the moment `omnimarket@dev` advances past it — that staleness was the OMN-13829
recurrence mechanism (see "Recurrence and the drift guard" below). Set
`OMNIMARKET_REF=<sha>` to pin an exact rev for reproducibility or offline use;
that override always wins outright.

## Before / after (proven in a clean throwaway venv, canonical `.venv` untouched)

```console
# BEFORE — fresh infra venv (uv pip install -e .), no omnimarket
$ onex skill merge_sweep --inventory-only
Unknown node 'node_pr_lifecycle_orchestrator'. ...

# AFTER — scripts/install-node-skill-package.sh --execute <throwaway-venv-python>
== step 3: verify node resolution ==
OK: N onex.nodes entry points; required nodes resolved:
['node_aislop_sweep', 'node_pr_lifecycle_orchestrator', 'node_session_orchestrator']

$ onex skill merge_sweep --inventory-only   # resolves (no 'Unknown node')
$ onex skill session --help                 # resolves (no 'Unknown node')
```

## Recurrence and the drift guard (OMN-14060)

The co-install above fixed the venv once (OMN-13829, 2026-07-02) and then
silently drifted back to a stale, non-git install within days (OMN-14060,
2026-07-06): something re-installed `omnimarket` from PyPI instead of the
canonical git-source co-install. Two compounding facts make this recur unless
guarded against:

1. **PyPI is not a fallback path.** omnimarket's last published PyPI release
   predates fixes that already merged to `dev` by weeks, and the newest
   published release is outright uninstallable (it pins a sibling package
   version that was never published — OMN-14064). There is no PyPI version of
   omnimarket that is ever "correct" here; only the git-source co-install is.
2. **Nothing re-asserted the venv's install state.** The install script is
   operator-run once; nothing re-checked that the venv still matched the
   git-pinned install afterward.

Fact (1) is a separate release-pipeline lane (OMN-14064). This runbook covers
the venv-side guard for fact (2), split detect/repair per CLAUDE.md's
"enforcement, not detection" rule:

- **Pre-flight (hot path, every `onex skill` dispatch):**
  `src/omnibase_infra/cli/omnimarket_drift_guard.py` — cheap and LOCAL ONLY
  (compares the current interpreter's installed omnimarket commit against the
  already-checked-out `$OMNI_HOME/omnimarket` clone's HEAD; no network). Fails
  OPEN (silently, no block) whenever either side can't be determined locally
  (fresh machine, CI, no canonical clone) so it never blocks environments where
  the `$OMNI_HOME` convention doesn't apply. On a real mismatch it raises a
  `click.ClickException` naming both commits and pointing at the repair
  command below — it never re-installs anything itself.
- **Repair (session/cron tick, or run by hand):**
  `scripts/check-omnimarket-venv-drift.sh [--repair] [PYTHON]` — refreshes the
  canonical clone from `origin/dev` (network), compares against the target
  venv, and with `--repair` re-runs `install-node-skill-package.sh` pinned to
  the fresh SHA. Exit 0 = no drift (or repaired); exit 1 = drift found and not
  repaired.

```bash
# detect only (safe, no mutation)
scripts/check-omnimarket-venv-drift.sh /path/to/venv/bin/python

# detect + repair
scripts/check-omnimarket-venv-drift.sh --repair /path/to/venv/bin/python
```

Wire the repair invocation to a session/cron tick so drift self-heals instead
of waiting on the next operator to hit the pre-flight error.
