# Release-Train Tag Trigger for Dev + Stability Lab Lanes (OMN-14889)

Runtime updates go through release trains; deploying to a dev/stability lab
lane is "push a git tag" (operator directive, 2026-07-20). This runbook
covers the tag-driven mechanism this ticket builds. For the underlying
per-lane refresh mechanics (health-gate, rollback, receipt shape), see
`docs/runbooks/stability-lane-refresh.md` (stability) — the dev-lane analog
is documented inline in `scripts/runtime_build/refresh_dev_lane.sh`.

## Two-train architecture (context)

- **Train 1 (lab lanes: dev, stability)** — git-ref deploys, no PyPI
  involved. `lab/<lane>/<utc>-<shortsha>` tags. This runbook.
- **Train 2 (prod)** — PyPI-backed. A `v*` release tag drives `release.yml`
  → PyPI publish. Promoting a published digest to the `.201` prod lane is a
  SEPARATE, grant-gated step through `node_redeploy_orchestrator`'s
  prod-promotion gate (CLAUDE.md Rules 2a/12) — nothing in this runbook
  affects, satisfies, or bypasses that gate. `lab/*` and `v*` are disjoint
  tag namespaces; this workflow only reacts to `lab/dev/**` and
  `lab/stability/**`.

## The mechanism

1. **Cut + push a tag** — `scripts/runtime_build/cut_release_train_tag.sh
   --lane {dev|stability} --ref <ref> --execute` (or via
   `gh workflow run release-train-lab.yml -f lane=stability -f ref=origin/dev
   -f execute=true`). Cuts `lab/<lane>/<utc>-<shortsha>` locally on all 5
   sibling clones (`omnibase_infra`, `omnibase_core`, `omnibase_compat`,
   `onex_change_control`, `omnimarket` — reusing `cut-lab-ref.sh`'s
   `compute_lab_tag_name`/`cut_lab_tags` logic verbatim) but pushes to
   GitHub on the **anchor repo only** (`omnibase_infra`) — the other 4
   repos' tags stay local-only; `git checkout <tag>` against a local tag
   works fine during the deploy step's own checkout, and pushing to 4 more
   GitHub repos would add write surface for no behavior difference.

2. **Tag push fires the `deploy` job** in
   `.github/workflows/release-train-lab.yml`
   (`on: push: tags: ['lab/dev/**', 'lab/stability/**']`). The job parses
   the lane from `github.ref_name` and calls:
   - `refresh_stability_lane.sh --ref <tag> --execute` for a `lab/stability/*`
     tag — REUSED VERBATIM (OMN-14873, zero changes).
   - `refresh_dev_lane.sh --ref <tag> --triggering-tag <tag> --execute` for
     a `lab/dev/*` tag — new (OMN-14889), cold-aware (see script header).

3. **Receipt** — each script writes its own durable JSON receipt
   (`~/.omnibase/state/{stability_lane_refresh,dev_lane_refresh}/history/*.json`
   + `latest.json`). The workflow's summary step reads `latest.json` and
   augments a COPY with `triggering_tag`/`lane` for the `$GITHUB_STEP_SUMMARY`
   view (it does not mutate the receipt file on disk — `refresh_stability_lane.sh`
   is reused verbatim and does not itself know about the triggering tag;
   `refresh_dev_lane.sh` already records `triggering_tag` natively via its
   `--triggering-tag` flag).

## Execution surface: the dedicated `omnibase-deploy` runner

These scripts need real host-path git access to the `$OMNI_HOME` sibling
clones (`git fetch`/`reset --hard`/tag the ambient clones directly, then run
`deploy-runtime.sh`'s `BUILD_SOURCE=workspace` path against them). The
shared 48–64 `omnibase-ci` fleet (`docker/docker-compose.runners.yml`)
deliberately does NOT have this access (docker.sock only, no `OMNI_HOME`
mount) — mounting `OMNI_HOME` into that whole fleet would give every CI job
host git-mutation access (rejected, blast radius). `refresh_stability_lane.sh`'s
own docstring also explicitly disallows an SSH-hop wrapper.

The decided answer (OMN-14889 Fork 1): **one new dedicated runner**,
`omninode-deploy-runner` (defined in `docker-compose.runners.yml`), labeled
`omnibase-deploy`, in its own runner group, with `OMNI_HOME` bind-mounted at
the identical host path (required for docker-outside-of-docker relative-path
resolution against the host daemon — see the compose file's comment block
for why a container-local alias path would silently break).

### Provisioning status: DONE — the runner is registered and online

**Verified live 2026-07-21:**

```bash
gh api repos/OmniNode-ai/omnibase_infra/actions/runners \
  --jq '.runners[] | select(.name=="omninode-deploy-runner")'
# -> {"name":"omninode-deploy-runner","status":"online","busy":false,
#     "labels":["self-hosted","Linux","X64","omnibase-deploy"]}
```

The runner is registered at the **repository** level, not the org level — the
historical recipe below is org-scoped and is kept only for reference /
re-provisioning. Verify against the `repos/...` endpoint above, not
`orgs/OmniNode-ai/actions/runners`.

> **This mechanism is LIVE, not dormant.** Pushing a `lab/dev/**` or
> `lab/stability/**` tag DOES dispatch the `deploy` job onto this runner and
> DOES refresh the target lane in execute mode. That is intended behavior —
> dev and stability lanes are pre-authorized — but a lab tag push is a real
> deploy, not a rehearsal or a no-op. Note the trigger only requires
> `release-train-lab.yml` to exist at the **tagged commit**, so a tag cut
> from a feature branch fires it exactly as a tag on `dev` would.

### Fixed blocker: host-side git access (OMN-14900 private OMNI_HOME)

Five runs on 2026-07-21T03:59–04:19Z all failed in the **`Refresh stability
lane`** step, in three distinct forms as intermediate fixes were attempted:

| Runs | Error | Exit |
|---|---|---|
| 03:59 | `fatal: detected dubious ownership in repository at /data/omninode/omni_home/omnibase_infra` | 128 |
| 04:01, 04:08 | `error: cannot open .git/FETCH_HEAD: Permission denied` | 255 |
| 04:14, 04:19 | `fatal: 'dev' is already checked out at /data/omninode/runtime-sync-worktrees/OMN-12618/...` | 128 |

All five died **before any container action** — zero `docker compose`,
`up -d`, or `--force-recreate` lines appear in any of the five job logs, so no
lane was mutated. Root cause: the runner container bind-mounted the **shared**
host clones, owned by a different uid and contended by host worktrees — a
write surface into clones the runner does not own. The interim live relief
(exec'd `git config --global`, a hand-edited compose adding `group_add 1000` +
`GIT_CONFIG_*` env) was **uncommitted container state** that any
`--force-recreate` silently drops.

The committed fix (OMN-14900) has three legs:

1. **Private OMNI_HOME** — `docker/docker-compose.runners.yml` gives the
   deploy runner its own runner-uid-owned clone tree at
   `DEPLOY_RUNNER_OMNI_HOME` (identical host:container bind path, required
   for docker-outside-of-docker path resolution) and **removes the shared
   `${OMNI_HOME}` mount entirely** — shared-clone writes become structurally
   impossible, and host worktree contention (`'dev' is already checked out`)
   can no longer occur because no host process holds branches in the private
   clones.
2. **Scoped `git -c safe.directory=<clone>`** on every git invocation in
   `scripts/runtime_build/` (refresh scripts, tag cut, stage_workspace
   probes, `deploy_source_ref.py`, `check_sibling_lock_pins.py`) — committed
   defense-in-depth that needs no container-global state.
3. **Automatic provisioning** — `scripts/runtime_build/ensure_runner_clones.sh`
   clones any missing repo (all 5 are public; anonymous https) and asserts
   euid operability at the top of every entry script.

Tag pushes changed with this: the cut-tag job now creates the GitHub tag ref
via `gh api repos/OmniNode-ai/omnibase_infra/git/refs` using the workflow's
own token (`permissions: contents: write` on the job) — the private clones
have no push credentials, so `git push` from a clone is not a supported tag
source. Externally-cut tags (operator workstation) still work for the anchor
repo, but note the 4 sibling **private** clones can only resolve a lab tag
that the runner's own cut-tag job cut locally; a deploy from an
externally-cut tag will fail loudly at RT-1 sibling checkout rather than
silently building mixed refs.

**The compose-side fix only takes effect at a container RECREATE from the
committed file — an operator-gated action** (the live container also carries
the OMN-13915 zombie). Until that recreate, the live container still runs
its pre-fix hand-edited config.

Seven `lab/stability/*` tags exist on origin; only five produced runs — the
two earliest (`43935c84…`, `e5404e36…`) predate the workflow file at their
tagged commits, so nothing fired for them.

### Recreate / re-provision procedure (OMN-14900)

```bash
# On the runner host (omninode-pc / .201):

# 1. Choose + export the PRIVATE clone-tree path (REQUIRED: the compose file
#    fail-fast interpolates DEPLOY_RUNNER_OMNI_HOME; any `docker compose`
#    against docker-compose.runners.yml needs it in the compose environment —
#    export it or add it to the .env next to the compose file).
export DEPLOY_RUNNER_OMNI_HOME=/data/omninode/runner_omni_home

# 2. (Only if the omninode-deploy-runner-creds volume was lost) mint a
#    REPOSITORY-scoped registration token (valid 1h) — registration is
#    repo-scoped, NOT org-scoped; RUNNER_GROUP is intentionally empty:
export DEPLOY_RUNNER_TOKEN="$(gh api -X POST \
  repos/OmniNode-ai/omnibase_infra/actions/runners/registration-token --jq '.token')"

# 3. Recreate the runner from the COMMITTED compose file:
cd /data/omninode/omni_home/omnibase_infra/docker   # shared infra clone: compose file source only
docker compose -f docker-compose.runners.yml up -d omninode-deploy-runner

# The compose entrypoint wrapper (root phase) mkdir/chowns
# ${DEPLOY_RUNNER_OMNI_HOME} for the runner uid; the 5 private clones are
# then provisioned automatically by ensure_runner_clones.sh on the first
# release-train job (or pre-seed them manually as the runner uid).

# 4. Verify registration (REPOSITORY endpoint, not org):
gh api repos/OmniNode-ai/omnibase_infra/actions/runners \
  --jq '.runners[] | select(.name=="omninode-deploy-runner")'
```

Both the `deploy` job and the `cut-tag` job run on this label (the
tag-cutting host needs the same `OMNI_HOME` access). As of 2026-07-21 that
runner is online, so both jobs pick up: the GitHub Actions trigger path
OMN-14889 originally flagged as unexercised has now fired end-to-end through
to the refresh step, where it fails on the host-side git access issue
documented above. The underlying deploy scripts were separately proven
directly on `.201` (see the ticket's canary evidence).

## Manual canary (exercises the deploy step without going through a tag)

The release-train scripts can be exercised directly on `.201`, which is
useful for isolating a refresh-script problem from the runner/trigger path.
This is exactly what the `deploy` job runs — same script, same flags:

```bash
ssh omni-201-ts
export OMNI_HOME=/data/omninode/omni_home
cd "${OMNI_HOME}/omnibase_infra"

# Cut + push the tag (from a workstation with OMNI_HOME sibling clones, or
# from .201 itself if its clones are current):
scripts/runtime_build/cut_release_train_tag.sh --lane stability --ref origin/dev --execute

# Exercise the deploy step directly against the pushed tag:
scripts/runtime_build/refresh_stability_lane.sh --ref lab/stability/<ts>-<sha> --execute
```

## Deliberately broken health-gate (rollback proof)

`--min-contracts 999999` on either refresh script forces the manifest-floor
check to fail regardless of the actual lane state, exercising the
rollback-and-reverify path without needing a genuinely bad deploy. The
receipt records `FAILED_ROLLED_BACK` (rollback restored health) or `FAILED`
(rollback also unhealthy — STOP AND REPORT, never masked as success).

## Prod is out of scope

No file in this ticket's diff creates, wires, or implies an autonomous
tag→prod path. `omninode-deploy-runner` has no prod-lane access beyond what
any host process already has via `OMNI_HOME`; the prod-promotion gate is
enforced at the `node_redeploy_orchestrator` layer, which this workflow
never calls.

This remains true now that the runner is online. The workflow triggers only
on `lab/dev/**` and `lab/stability/**`, a namespace disjoint from the `v*`
tags that drive `release.yml` → PyPI publish (Train 2). Promoting any digest
to the `.201` prod lane still requires a fresh, CODEOWNERS-approved
`ModelProdPromotionGrant` through `node_redeploy_orchestrator`'s gate
(CLAUDE.md Rules 2a/12) — nothing here can satisfy or bypass it.
