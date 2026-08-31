# Deploy-source clone reconcile (`.201`)

**Ticket:** OMN-17291 · **Scripts:** `scripts/runtime_build/reconcile_deploy_clones.sh`,
`scripts/runtime_build/deploy_source_ref.py`

## What this covers

`/data/omninode/omni_home` on `.201` holds the canonical clones every lane image
is built from. OMN-17190 reconciles the clones on the operator's Mac; nothing
reconciled these. This runbook covers the reconciler that closes that gap.

## The failure this exists to prevent

On 2026-08-31 the dev-lane advance at `05:38:22Z` baked an `omnimarket` **11
commits behind** `origin/dev`. Readback from the running container:

```
docker exec omninode-runtime cat /app/.venv/.../omnimarket-0.4.11.dist-info/direct_url.json
  -> vcs_info.commit_id = 05e3882f9e2a...   # the .201 clone's HEAD
origin/dev                                 = 2f123b4c01ea...   # 11 ahead
```

Two independent routes produced it, neither of which failed:

1. **`omnibase_core` was structurally unpullable.** Its clone carried
   `core.bare=true` while still having a full working tree on disk. In that
   shape `git fetch` exits **0** (and advances `origin/dev`) while every
   `git checkout` / `git status` exits **128**. Any loop that reads the fetch's
   exit status reports success forever while HEAD never moves. `git config
   core.bare false` repaired it in one command.
2. **`stage_workspace.sh` built the ambient tree when `DEPLOY_REF` was unset.**
   It printed `WARNING: DEPLOY_REF unset` and continued. A warning in a
   4000-line deploy log is not a gate.

## Running the reconciler

```bash
ssh jonah@192.168.86.201
OMNI_HOME=/data/omninode/omni_home \
  bash /data/omninode/omni_home/omnibase_infra/scripts/runtime_build/reconcile_deploy_clones.sh
```

Environment:

| Var | Default | Meaning |
|-----|---------|---------|
| `OMNI_HOME` | *(required)* | root holding the deploy-source clones |
| `RECONCILE_BRANCH` | `dev` | tracked branch to reconcile onto |
| `RECONCILE_RECEIPT` | `${OMNI_HOME}/.deploy-clone-reconcile.json` | receipt path |

The repo set comes from `sibling_clone_manifest.sh` — the same single source of
truth `ensure_runner_clones.sh` and the sibling-pin preflight read.

## What it guarantees

* **A clean `fetch` is never accepted as sync.** After the fast-forward it
  re-reads `HEAD` and compares it to the fetched tip; a mismatch exits `5` and
  names the repo. This is the post-condition the 2026-08-31 defect had none of.
* **`core.bare=true` on a clone with a working tree is a named failure** (exit
  `3`), with the one-command repair printed — never git's generic
  `fatal: this operation must be run in a work tree` three calls later.
* **Detached clones are recovered onto the branch.** Four of the five `.201`
  clones were detached; a reconciler that only handles the on-branch case leaves
  them stale.
* **It refuses rather than destroys.** No `reset --hard`, no `clean -ffdx`.
  A tracked modification blocks the reconcile and is reported by path.
* **A receipt is emitted every run** (success and failure), naming each repo and
  its `before_sha -> after_sha`.

### The one tolerated kind of dirt

The build writes its own outputs into its own build context, so the
`omnibase_infra` clone is dirty by construction. Measured on `.201`
2026-08-31:

```
 M workspace/sibling-pin-comparison.json      # check_sibling_lock_pins.py output
 M workspace/sibling-vcs-provenance.json      # stage_workspace.sh output
?? workspace/deploy-source-refs.json          # deploy_source_ref.py output
?? origin/                                    # stray directory
```

Untracked paths anywhere, and anything under `workspace/`, are reported and
tolerated. Untracked files are left untouched by the checkout; a **tracked**
build output under `workspace/` is reset to the branch's content, which is
correct — the next build regenerates it, and nothing an operator authored lives
there. A tracked modification **outside** `workspace/` still blocks. The
exclusion set is `BUILD_SCRATCH_PREFIXES` in `deploy_source_ref.py`.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | every present manifest clone is at `origin/<branch>` |
| `3` | a clone refused: dirty (tracked), diverged, or bare-with-working-tree |
| `5` | a fetch succeeded but HEAD did not land on the fetched tip |
| `64` | precondition failure (`OMNI_HOME` unset/missing, `git`/`python3` absent, no clone found) |

## Related change: unpinned builds are refused

`stage_workspace.sh` now **exits 5** when `DEPLOY_REF` is unset. The
ambient-tree build is still reachable, but only via
`ALLOW_UNPINNED_DEPLOY_SOURCE=1` — loud, named, and never the default. Pinned
callers (`cut-lab-ref.sh`, `refresh_dev_lane.sh`, `refresh_stability_lane.sh`)
already export `DEPLOY_REF` and are unaffected.

## Still open

Two ticket ACs are **not** closed by this change and remain tracked on
OMN-17291:

* **AC4** — release-mode builds derive `OMNIMARKET_REF` / `OMNIBASE_COMPAT_REF`
  from the clone's HEAD (`read_repo_ref_or_main` in `deploy-runtime.sh`), which
  `docker/Dockerfile.runtime:403` installs via `git+https://...@${OMNIMARKET_REF}`.
  Those build-args are still not asserted against `origin/<branch>`.
* **AC5** — no post-deploy readback compares the deployed `direct_url` SHA per
  sibling against that sibling's `origin/dev` tip.
