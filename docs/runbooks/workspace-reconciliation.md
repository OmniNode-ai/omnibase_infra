# Workspace reconciliation, and the invocation-time floor

**Tickets:** OMN-17305 (epic) · OMN-17307 (movement proof) · OMN-17309 (floor) · OMN-17311 (schedulers)
**Builds on:** OMN-17190 (Mac venv reconciler + `onex` wrapper) · OMN-17291 (`.201` deploy-source clone reconciler)

---

## The rule

> A reconcile step is judged by **reading the surface back**, never by the exit
> status of the command that was supposed to move it.

Four incidents, one shape — a surface that did not move while everything that
could have noticed reported success:

| Incident | Surface | Why nothing noticed |
| --- | --- | --- |
| OMN-17291 | `.201` `omnibase_core` clone | `core.bare=true` with a working tree: `git fetch` exited 0 forever, `git checkout` exited 128 forever |
| OMN-17291 | dev-lane image | built from that clone; `DEPLOY_REF` unset printed a warning into a 4000-line log |
| OMN-17190 | this Mac's CLI venv | `uv run` resolved a different `onex` off `PATH`; the self-heal was never the code that ran |
| OMN-16932 | a delegation probe | ran against a build nobody chose and **produced a receipt** |

Drift is expected and cheap to fix. The defect is that a repair and a no-op were
the same observation.

---

## The pieces, and who owns what

```
       scheduler (per host)
              │
              ▼
   scripts/reconcile-host.sh          ← owns ORDERING and PROOF, no repair logic
       │            │
       │            └─→ scripts/reconcile-workspace-venvs.sh          (OMN-17190)
       └──────────────→ scripts/runtime_build/reconcile_deploy_clones.sh (OMN-17291)
              │
              ▼
   scripts/reconcile_verify_movement.py   ← the verdict table; stdlib-only
              │
              ▼
   ${OMNI_HOME}/.onex-workspace-floor.json  ← stamped ONLY on an all-proven run
              │
              ▼
   scripts/onex   ← refuses evidence-producing commands below the floor
```

`reconcile-host.sh` runs identically on every machine. There is no per-host
variant, and adding one would defeat the epic.

### Verdicts

| verdict | condition | outcome |
| --- | --- | --- |
| `MOVED` | `after == target`, `after != before` | ok |
| `ALREADY_AT_TARGET` | `before == after == target` | ok |
| `DID_NOT_MOVE` | `after != target` | FAIL + Slack alert, no floor stamped |
| `INDETERMINATE` | `after` or `target` unreadable | FAIL + Slack alert, no floor stamped |
| `UNCOVERED` | a delegate is absent | FAIL + Slack alert — a surface nobody reconciles is not a skip |

`INDETERMINATE` fails closed: "could not determine" is never "fine".

**Targets are established by `reconcile-host.sh` itself** — it fetches before
verdicting rather than trusting the delegate to have fetched, or reading the
delegate's own receipt. A verifier that takes its target from the thing under
verification is not a verifier.

---

## The floor

`${OMNI_HOME}/.onex-workspace-floor.json` records the minimum installed state
that has been **proven** on this host: per-distribution version floors plus
omnimarket's target commit. It is written only on a run where every surface
verdicted ok, so a failed reconcile leaves the last proven floor in place rather
than replacing it with an aspiration.

`scripts/onex` checks it before exec:

| condition | evidence-producing subcommand | ordinary subcommand |
| --- | --- | --- |
| at or above floor | run | run |
| below floor | reconcile once, then **REFUSE** (exit 3) | run, after ONE stderr warning |
| floor absent / unparseable | reconcile once, then **REFUSE** (exit 3) | run, after ONE stderr warning |

*Evidence-producing* = `delegate skill node run-node run gate occ compliance
validate doctor health db ledger`, or any invocation carrying `--output`,
`--receipt`, `--report`, `--emit-receipt`, `--evidence`. The list is a data table
at the top of `scripts/onex` with a test over it.

A venv **ahead** of the floor is silent — dev-tip dogfooding means the installed
version legitimately leads the last stamp. An omnimarket **commit that merely
differs** is not "ahead" (commits have no ordering), so it reads as unproven.

The check costs one `awk`: versions come from `*.dist-info` directory names, the
commit from one `direct_url.json` read with the `read` builtin. No Python starts
and no socket opens — which is also why it still works when the venv's own
interpreter is broken, the case where it matters most.

There is **no bypass variable**, and a test asserts none is added.

---

## Schedulers — read the host, do not assume

| host | mechanism | why |
| --- | --- | --- |
| this Mac | omniclaude plugin `PostToolUse` tick, throttled to 10 min | ships with the plugin, no per-host install, fires while a session is doing work |
| `.201` | `/etc/cron.d/omninode-workspace-reconcile` at **:19 hourly** | the same governed path that already runs the system Slack reporter |

`launchd` is deliberately not used on the Mac: it reconciles on a wall clock,
including mid-dispatch through the venv it is rewriting, and needs a per-host
install a plugin update does not carry.

`:19` is chosen to clear the other two root jobs — `*/15` (system Slack report)
and `:37` (maintenance sync) — so three jobs never contend on the same clones or
the same Slack rate limit. `tests/scripts/test_workspace_reconcile_cron_omn17311.py`
asserts the separation.

### Attaching a new machine (mini, air, …)

There is no porting step:

1. Ensure `OMNI_HOME` points at that host's canonical registry root.
2. Pick the scheduler that actually fires on that host.
3. Point it at `${OMNI_HOME}/omnibase_infra/scripts/reconcile-host.sh`.

Do **not** write a per-host reconciler. If a host needs different behaviour, that
is a bug in `reconcile-host.sh` or a missing flag on it.

---

## Operating it

```bash
# Verdict only — mutates nothing, runs no delegate, stamps no floor.
bash "$OMNI_HOME/omnibase_infra/scripts/reconcile-host.sh" --check --omni-home "$OMNI_HOME"

# Reconcile and prove.
bash "$OMNI_HOME/omnibase_infra/scripts/reconcile-host.sh" --omni-home "$OMNI_HOME"

# What was last proven on this host.
cat "$OMNI_HOME/.onex-workspace-floor.json"

# What the last run decided, per surface.
cat "$OMNI_HOME/.onex-workspace-reconcile.json"
```

Exit codes: `0` every surface proven · `2` a surface could not be proven ·
`3` indeterminate configuration (no `OMNI_HOME`, no `git`, no `python3`).

`OMNI_HOME` has **no default**. A guessed root would reconcile some other
checkout and report success for a workspace nobody is running.

### When a run fails

The stderr report names the surface, the verdict, and the detail. Common causes:

- **`clone:<repo>: UNHEALTHY … core.bare=true`** — `git -C <clone> config core.bare false`, then re-run.
- **`clone:<repo>: DID_NOT_MOVE`** — the clone is dirty or diverged. The delegate refuses to touch a dirty canonical clone by design; commit or stash the work, in a worktree.
- **`venv:<dist>: DID_NOT_MOVE`** — the lock moved and the sync did not land it. Run the venv delegate by hand and read its error.
- **`venv:omnimarket: INDETERMINATE`** — omnimarket is installed from PyPI, so there is no `direct_url.json` commit to compare. Re-run the provider co-install.
- **`*-surface: UNCOVERED`** — a delegate is missing from this checkout. Pull the clone.

A `.201` maintenance window can bounce containers and runners mid-run; a fetch
that fails for that reason is a retry, not a diagnosis.

---

## The gate

`scripts/check_reconciler_movement_proof.py` runs as a pre-commit hook and as the
`reconciler-movement-proof` CI job. Two parts, of deliberately different strength:

1. **Structural** — `reconcile-host.sh` must invoke the verifier, and
   `verdict()` must take exactly `(before, after, target)`. Adding an
   exit-status parameter is the one change that would quietly re-open the defect
   class, so the signature is pinned by the gate as well as by tests. Neither
   half can be satisfied by editing a comment.
2. **Declaration ratchet** — any `scripts/reconcile*.sh` must invoke the verifier
   or carry `# movement-proof: <how>` / `# movement-proof-delegated-to: <path>`.
   Discovery is a filesystem glob, not a manifest, so a new reconciler cannot
   arrive unnoticed. This half cannot tell a real readback from a marker someone
   typed to quiet the gate; its value is that it turns an invisible omission into
   a reviewable line in the diff — the same value, and the same limit, as this
   repo's existing `# raw-prod-bypass-ok:` and `# canonical-inference-ok:`
   ratchets.

---

## Known limits

- **Between ticks, host state can drift.** That is what the invocation-time floor
  covers, and it is why the floor refuses rather than warns for evidence.
- **The declaration ratchet is a declaration.** See above.
- **`reconcile-host.sh` runs from the clone it reconciles.** A stale clone runs a
  stale reconciler for exactly one tick.
- **Script-level, not yet node-level.** The successor named on OMN-17190 is a
  `NodeCompute` drift-detect handler behind a `NodeEffect` reconcile publisher,
  emitting its receipt to the bus. `verdict()` is already shaped as that pure
  function and the receipt is already shaped as that event, so the port is a lift
  rather than a rewrite.
