<!--
SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
SPDX-License-Identifier: MIT
-->

# Node-skill package install (omnimarket) — OMN-13829

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

## Why not a `pyproject` dependency + `uv lock`?

Both the declared-dependency route and a plain version bump are **impossible**,
verified with `uv lock` / `uv pip install`:

| Attempt | Result |
|---|---|
| `omnimarket>=0.4.6` from PyPI | Published wheels pin `omnibase-core<0.45.0` / `omnibase-spi<0.22.0`; conflict with infra's `core>=0.46.1` / `spi>=0.23.0`. |
| Declare omnimarket + `uv lock` (git dev rev) | omnimarket depends back on `omnibase-infra` (circular), and its `[tool.uv] override-dependencies` git-pin `omnibase-core`/`omnibase-infra` to foreign revs (uv applies them transitively) → `conflicting URLs`. |
| Force it via workspace source | Produces an inconsistent lock (no `omnimarket` package block, dropped transitive deps). |
| `uv pip install` (deps mode) | omnimarket's required `omninode-memory==0.15.0` hard-pins `omnibase-infra==0.30.1` / `spi 0.20.x` → unsatisfiable. |

The compatible pins (`omnibase-core>=0.45.0,<0.47.0`, `omnibase-spi>=0.23.0,<0.24.0`)
plus the current nodes exist only on **omnimarket@dev**
(`363e4319aa7288bdf0f2858af7993bf8aa91fca0`). It installs cleanly only with
`--no-deps` (the infra venv already supplies the omni-internal deps; only
`omnibase-compat==0.5.5` + `omninode-memory==0.15.0` are added, also `--no-deps`).

> Durable fix (upstream, separate ticket): omnimarket and its `omninode-memory`
> dependency must drop the stale/self-referential internal pins and publish a
> wheel resolvable against current omnibase_infra. Then replace this script with
> a plain `uv add omnimarket>=X,<Y`.

## The fix (operator-run — mutates the venv)

`scripts/install-node-skill-package.sh` encapsulates the vetted install. It is
gated behind `--execute`; without it, it prints the plan only.

```bash
# from the omnibase_infra repo root, with the target venv active
scripts/install-node-skill-package.sh                    # dry-run plan
scripts/install-node-skill-package.sh --execute          # apply to $VIRTUAL_ENV
scripts/install-node-skill-package.sh --execute /path/to/venv/bin/python
# bump the rev without editing the script:
OMNIMARKET_REF=<newer-sha> scripts/install-node-skill-package.sh --execute
```

## Before / after (proven in a clean throwaway venv, canonical `.venv` untouched)

```console
# BEFORE — fresh infra venv (uv sync), no omnimarket
$ onex run-node node_pr_lifecycle_orchestrator
Unknown node 'node_pr_lifecycle_orchestrator'. ...

# AFTER — scripts/install-node-skill-package.sh --execute
OK: 459 onex.nodes entry points; required nodes resolved:
['node_aislop_sweep', 'node_pr_lifecycle_orchestrator']
```
