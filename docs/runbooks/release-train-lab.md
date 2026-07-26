# Release-Train Tag Trigger for Dev + Stability Lab Lanes (OMN-14889)

Runtime updates go through release trains; deploying to a dev/stability lab
lane is "push a git tag" (operator directive, 2026-07-20). This runbook
covers the tag-driven mechanism this ticket builds. For the underlying
per-lane refresh mechanics (health-gate, rollback, receipt shape), see
`docs/runbooks/stability-lane-refresh.md` (stability) — the dev-lane analog
is documented inline in `scripts/runtime_build/refresh_dev_lane.sh`.

**2026-07-26 update (OMN-15151):** the mechanism below is no longer
theoretical — it went end-to-end SUCCESS on
[run 30180376657](https://github.com/OmniNode-ai/omnibase_infra/actions/runs/30180376657)
after a six-iteration hardening chain under parent
OMN-14900. This
update adds the operator-facing procedure (§ "Operational procedure"),
the preflight catalog (§ "What the preflights catch"), the T0/T1 readback +
rollback discipline (§ "Readback and rollback discipline"), dev-lane gotchas,
and a hard prod pointer. Nothing below supersedes the architecture sections
that already existed (Two-train architecture / mechanism / runner
provisioning) — those stand unchanged.

## Operational procedure: cut a stability tagged deploy

This is the exact sequence proven live 2026-07-26 (run 30180376657, tag
`lab/stability/20260725T235956Z-87ec5b3165ce`, `overall: PASS`). Run from an
operator workstation with `gh` authenticated against `OmniNode-ai`.

**Step 1 — confirm dev tip and cut the tag.**

```bash
gh api repos/OmniNode-ai/omnibase_infra/commits/dev --jq '.sha'
# -> 87ec5b3165ce631ef4692edc6b00b3874a4d5446   (the commit you are about to tag)

gh workflow run release-train-lab.yml \
  --repo OmniNode-ai/omnibase_infra \
  -f lane=stability -f ref=origin/dev -f execute=true
```

> **WARNING — tag content, never a parked tag.** The tag's content is
> whatever commit `--ref` resolves to **at cut time**, not a name you can
> reuse. `cut_release_train_tag.sh` computes `lab/<lane>/<utc>-<shortsha>` —
> the timestamp+shortsha in the tag name is generated fresh every cut, so
> there is no "reuse tag X" path; re-running the same command against a
> moved `dev` produces a **new** tag object pointing at the new tip. Never
> hand-construct or reuse a `lab/stability/*` tag string — always let the
> cut step mint it, and always confirm the `dev` SHA immediately before
> cutting (a stale local clone silently tags an old tree — see
> `reference_stale_canonical_clone_phantom_behavior`).

**Step 2 — watch the run.**

```bash
gh run list --repo OmniNode-ai/omnibase_infra --workflow release-train-lab.yml --limit 1
gh run watch <run-id> --repo OmniNode-ai/omnibase_infra --exit-status
```

Expected job sequence (both jobs on the `omnibase-deploy` runner):

```
✓ Cut + push release-train tag   (~15s: checkout, cut+push tag)
✓ Deploy triggering tag to its lane
    ✓ Determine lane from tag
    ✓ Refresh stability lane      (~9 min: preflights, build, restart, health-gate)
    - Refresh dev lane            (skipped — lane=stability)
    ✓ Publish receipt to job summary
    ✓ Fail the job if the refresh did not succeed
```

Terminal proof line in the job log (`Refresh stability lane` step):

```
[refresh-stability-lane] === SUCCESS: health-gate PASS, ancestry OK ===
[refresh-stability-lane] === Receipt written: /home/runner/.omnibase/state/stability_lane_refresh/history/<ts>-<sha>.json ===
[refresh-stability-lane] result: SUCCESS
```

If the run instead fails inside `Refresh stability lane`, do not re-run
blind — read the failing preflight name from the log first and match it
against the table below; most failures as of 2026-07-26 are one of these
six known classes, not a new defect.

## What the preflights catch

Six preflight/environment classes were closed across the OMN-14900 chain
before run 30180376657 went green. Each is now a **hard gate inside
`deploy-runtime.sh` / `refresh_stability_lane.sh`** — a regression in any of
them fails the deploy loudly at the named step, before any container
mutation, rather than producing a silently-wrong build.

| # | Preflight | What it catches | Failure signature (pre-fix) | Fix PR |
|---|---|---|---|---|
| 1 | Sibling clone manifest | A sibling repo (e.g. `omnibase_spi`) missing from the runner's private clone tree — `ensure_runner_clones.sh` only knew about the 5 `TAG_REPOS`, not every transitive dependency | `ModuleNotFoundError` / sibling-lock-pin preflight abort citing an unresolvable path | `#2450` (OMN-15137) |
| 2 | Interpreter resolution (lock-pin check) | `stage_workspace.sh` invoking `check_sibling_lock_pins.py` with an interpreter lacking `pydantic` | `ModuleNotFoundError: No module named 'pydantic'` mid-preflight | `#2446` (OMN-15131) |
| 2b | Interpreter resolution (contracts path) | `deploy-runtime.sh` step 3b resolving `omnibase_core` runtime contracts from the wrong (installed vs. workspace) path | `[deploy] ERROR: Could not locate omnibase_core runtime contracts` — abort before any build | `#2444` (OMN-15122) |
| 3 | `buildx` plugin | Runner image missing the Docker BuildKit `buildx` plugin needed for `--mount` cache builds | Build step fails with an unrecognized `docker buildx` invocation / cache-mount syntax error | `#2452` (OMN-15141) |
| 4 | `rsync` | `deploy-runtime.sh`'s "Syncing runtime build context..." step (`rsync -a --delete scripts/runtime_build/ -> deployed`) needs `rsync` in the runner image | `rsync: command not found` | `#2434` (OMN-15103) |
| 5 | Root-owned workspace debris | A prior run leaving root-owned `.venv`/build artifacts in the runner's Actions workspace, which the `runner-job-started.sh` cleanup hook cannot remove as uid 1001 — traced to a `docker exec ... -u root` step inside an earlier build leaving files it didn't clean up | `[runner-job-started] Resetting workspace` hangs/fails; `Set up runner` step never completes | `#2448` (OMN-15134) |

> **WARNING — these are the *known* classes, not an exhaustive list.** A
> seventh gap (`ensure_runner_clones.sh` never provisioning
> `omnibase_spi` transitively for `omnimarket`'s own dependency) was found
> and fixed as part of #2450/OMN-15137 in the same chain — if a *new*
> preflight failure appears that doesn't match this table, file a ticket
> under the OMN-14900 epic rather than hand-patching the runner
> out-of-band (the OMN-14900 root-cause section above explains why hand
> patches to the container are **not durable evidence** — they vanish on
> the next `--force-recreate`).

## Readback and rollback discipline

Every stability refresh (whether run via the tag workflow above or the
manual canary) captures **T0** (pre-state) before mutating anything and
**T1** (post-state) after the health-gate, and only reports `SUCCESS` if T1
proves forward progress. This is `refresh_stability_lane.sh` +
`verify_stability_refresh.py` (see `docs/runbooks/stability-lane-refresh.md`
for the full mechanism) — this section is the reading guide for the run
30180376657 evidence specifically.

**T0 (captured at the top of the `Refresh stability lane` step):**

```
[refresh-stability-lane] omninode-runtime: pre_image_id=sha256:f06b0bf7d216...
[refresh-stability-lane] runtime-effects:  pre_image_id=sha256:40ea65b5ceaf...
[refresh-stability-lane] runtime-worker:   pre_image_id=sha256:bac8f35eb93f...
[refresh-stability-lane] projection-api:   pre_image_id=sha256:22a2ce853570...
[refresh-stability-lane]   omnibase_infra prior HEAD: 49c886a56697
```

Preflight rollback anchors are tagged from these same pre-images
immediately after capture (`<project>-<service>:latest ->
<project>-<service>:preflight-<UTC>`) — this is the retag the OMN-14796
gap was about; confirm the four `tagged ... -> ...preflight-<UTC>` lines
appear in the log before trusting any rollback path for this run.

**T1 (the health-gate block, written after build+restart):**

```json
{
  "manifest_count": 292, "manifest_floor": 288, "manifest_ok": true,
  "health_ok": true, "health_detail": "status=healthy",
  "cluster_healthy": true,
  "consumer_groups_stable": true,
  "revision_readback_ok": true,
  "overall": "PASS"
}
```

Read these five fields in this order — each one is a necessary, not
sufficient, condition for trusting the deploy:

1. **`http://192.168.86.201:18085/health`** and **`:18086`** (effects port)
   both healthy — the log line `[deploy] Health check passed.` confirms
   18085; 18086 is asserted inside the same health-gate pass (not shown
   separately in this run's log — verify with a direct probe if in doubt:
   `curl -sf http://192.168.86.201:18086/health`).
2. **Contract count** (`manifest_count`) — 292 ≥ floor 288. Never trust a
   PASS if this number is at or below the floor from the *previous* known
   run; a flat or falling contract count on a merge that should have added
   nodes is a silent regression the floor alone won't catch.
3. **`discovery_errors` baseline** — not surfaced as a top-level health-gate
   field in this receipt shape; read it from the manifest endpoint directly
   post-deploy: `curl -s http://192.168.86.201:18085/v1/introspection/manifest | jq '.discovery_errors // empty'`.
   Treat any non-empty result as a regression even if `overall: PASS`.
4. **Consumer groups** — every declared group in
   `consumer_groups_stability.yaml` reads `Stable` or an expected `Empty`
   (no active traffic on that lane's synthetic path yet); this run shows 2
   `Stable` + 4 `Empty`, all expected for the stability lane's current
   traffic profile.
5. **`vcs_ref` / revision ancestry** — `revision_readback_ok: true` plus the
   per-service `revision_match: true` entries confirm every one of the 4
   core containers' `org.opencontainers.image.revision` label equals the
   tagged commit (`87ec5b3165ce`), and the pre-refresh `git merge-base
   --is-ancestor <prior> <new>` assertion (logged as `RT-1: manifest
   assertion passed`) proves the deploy moved the lane **forward**, not
   sideways or backward.

**Rollback discipline.** If `overall` had been `FAIL`, the script rolls
back all 4 core services to their `preflight-<UTC>` tag and re-runs the
health-gate with `--no-require-digest-change` (the rolled-back image is
deliberately old). The receipt records exactly one of:

- `SUCCESS` — this run's outcome.
- `FAILED_ROLLED_BACK` — refresh failed, rollback restored a healthy lane.
  **This is the discipline to insist on in any report**: `FAILED_ROLLED_BACK`
  with a T1 readback that equals T0 (same image IDs, same revision label) is
  the proof the rollback actually worked, not just that the script claimed
  it did. Diff T1's `post_image_id`/`revision_label` fields against T0's
  `pre_image_id`/prior HEAD by hand if the receipt doesn't make the equality
  obvious.
- `FAILED` — rollback ALSO unhealthy. STOP AND REPORT; this is a human
  escalation per `feedback_auto_file_tickets_on_breakage`, never a
  retry-until-green loop.

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

2. **The `deploy` job fires on one of two paths** in
   `.github/workflows/release-train-lab.yml` (OMN-14957): **chained in the
   same run** after an execute-mode `cut-tag` (`needs: cut-tag`, keyed off
   the job's `tag` output) — this is the only path a runner-cut tag can
   take, because refs created with the workflow's own `GITHUB_TOKEN` never
   deliver `push` events (GitHub's documented anti-recursion suppression;
   proven live by run 29977781670, whose ref creation fired nothing) — or
   via `on: push: tags: ['lab/dev/**', 'lab/stability/**']` for tags pushed
   with non-`GITHUB_TOKEN` credentials (operator workstation). The job
   parses the lane from the triggering tag and calls:
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

# 1b. Export the host path of the operator env file (REQUIRED, OMN-14958:
#     fail-fast interpolated the same way). It is bind-mounted READ-ONLY at
#     /run/omnibase-operator.env inside the runner, and the runtime_build
#     scripts + deploy-runtime.sh read it via OMNIBASE_OPERATOR_ENV_FILE —
#     without it the deploy job dies before any build (run 29977968728).
export DEPLOY_RUNNER_OPERATOR_ENV_FILE="${HOME}/.omnibase/.env"

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

## Dev-lane refresh: known gotchas

The dev lane uses `refresh_dev_lane.sh` (same tag-trigger mechanism, `lab/dev/*`
namespace) instead of `refresh_stability_lane.sh`. It is **cold-aware** (see
the script's own header) because the dev lane is ephemeral — it is routinely
GC/idle-reclaimed to zero containers between uses, unlike the always-warm
stability lane. Two gotchas are load-bearing enough to call out here rather
than leaving them buried in script comments:

> **WARNING — workspace builds ship stale `configs/*.yaml`, even when the
> revision label is correct.** On `.201` `BUILD_SOURCE=workspace` builds, a
> config-only change (e.g. a routing YAML) can silently NOT propagate into
> the installed package tree even though `.py` code and the image's
> `org.opencontainers.image.revision` label both show the new SHA — verified
> live 2026-07-14 (OMN-14626/14625 readback): the staged source was
> byte-identical to the branch worktree, but the *installed*
> `routing_tiers.yaml` still had the old content. **Do not trust the
> revision label alone for a config-only change.** `md5sum` the installed
> config inside the container against the branch source before declaring a
> config-only deploy proven:
> ```bash
> docker exec <container> md5sum /app/.venv/lib/python3.12/site-packages/omnimarket/configs/<file>.yaml
> md5sum omnimarket/src/omnimarket/configs/<file>.yaml   # compare against this
> ```
> A content-parity gate exists to catch this class in CI (OMN-14631), but its
> first implementation had a false-fail bug (OMN-14635) — treat CI green on
> this gate as necessary, not sufficient, until you've confirmed it's been
> re-verified live since. Full history: memory
> `reference_workspace_build_ships_stale_config_yaml`.

> **WARNING — the dev-lane health-gate can report `FAILED` on a fully
> healthy deploy (OMN-14968, open as of 2026-07-26).** `refresh_dev_lane.sh`'s
> health-gate expects a `runtime-worker` container that has **no container
> in any state** on the dev lane, before or after the deploy — a
> pre-existing lane-composition gap (the dev lane's compose overlay never
> provisions `runtime-worker`), not a regression from your deploy. If a dev
> refresh reports `FAILED`/exit 2 but `omninode-runtime` / `runtime-effects`
> / `projection-api` are all healthy at the correct revision and
> `:8085/health` is healthy, treat that as a **false FAILED** from this known
> gap, not a real rollback trigger — verify the 3 real services directly
> before escalating.

**Cold vs. warm bring-up.** If the dev lane's core containers are entirely
absent (fully torn down, not just stale), `refresh_dev_lane.sh` alone is not
enough — you need the full cold bring-up procedure, which is **documented
separately and not duplicated here**: see
`docs/runbooks/cold-lane-full-bringup.md` for the `--profile runtime`
requirement, the `BUILD_SOURCE=workspace` forcing, and the deps + migration
one-shot sequencing a cold lane needs that a warm refresh does not.

## Deliberately broken health-gate (rollback proof)

`--min-contracts 999999` on either refresh script forces the manifest-floor
check to fail regardless of the actual lane state, exercising the
rollback-and-reverify path without needing a genuinely bad deploy. The
receipt records `FAILED_ROLLED_BACK` (rollback restored health) or `FAILED`
(rollback also unhealthy — STOP AND REPORT, never masked as success).

## Prod: pointer only — this runbook does not cover prod promotion

**Nothing in this document authorizes, arms, or performs a prod mutation.**
This is a hard pointer-only section, not a procedure to follow here.

- No file in this ticket's diff creates, wires, or implies an autonomous
  tag→prod path. `omninode-deploy-runner` has no prod-lane access beyond
  what any host process already has via `OMNI_HOME`; the prod-promotion gate
  is enforced at the `node_redeploy_orchestrator` layer, which this workflow
  never calls.
- This remains true now that the runner is online. The workflow triggers
  only on `lab/dev/**` and `lab/stability/**`, a namespace disjoint from the
  `v*` tags that drive `release.yml` → PyPI publish (Train 2).
- Promoting any digest to the `.201` prod lane requires a **fresh**,
  `@main`-resolved, digest-scoped `ModelProdPromotionGrant` — see CLAUDE.md
  Rules 2a and 12 for the full gate (health-conditional waiver, standing/
  overnight grants excluded, raw-bypass CI gate). A standing or overnight
  autonomy grant, or a prior stability-lane SUCCESS like run 30180376657
  above, **does not** open prod on its own — it is only ever the
  *prerequisite artifact* a fresh grant PR cites.
- **The correct pattern for referencing a proven stability artifact in a
  grant is prep-only, never self-arming.** See `onex_change_control#4892`
  — a real grant-prep PR that cites this exact run
  (30180376657 / tag `lab/stability/20260725T235956Z-87ec5b3165ce` / digest
  readback cross-checked against live `docker inspect` on `.201`) and is
  explicit that it is "background-lane prep, not an active grant... becomes
  effective only when a human operator explicitly approves and lands it on
  `main`" with no auto-merge armed. Model any future grant PR on that
  shape, not on a self-merging or auto-arming one.
- **Raw docker mutation against the prod lane is forbidden, full stop** —
  no `docker tag`/`docker commit` retag, no direct `docker compose -p
  omnibase-infra-prod ... up --force-recreate`/`up -d`. The
  `no-raw-prod-bypass` CI gate
  (`.github/workflows/no-raw-prod-bypass.yml` →
  `tests/test_no_raw_prod_bypass_policy.py`) rejects any committed recipe of
  that shape unless the line carries a `# raw-prod-bypass-ok: <reason>`
  annotation reserved for forensic/illustrative quotes — never for an actual
  un-gated promotion. All real prod mutation routes through
  `redeploy-start → prod-promotion gate → deploy-agent`.

## Verification checklist

You are done with a stability tagged-deploy when **all** of the following
read back true — not when the workflow run shows green alone:

- [ ] `gh run view <run-id> --repo OmniNode-ai/omnibase_infra --json conclusion --jq .conclusion` == `success`
- [ ] The `Refresh stability lane` step log contains
      `=== SUCCESS: health-gate PASS, ancestry OK ===`
- [ ] The health-gate JSON block shows `"overall": "PASS"`, `"manifest_ok": true`,
      `"consumer_groups_stable": true`, `"revision_readback_ok": true`
- [ ] `manifest_count` is ≥ the floor (288) **and** ≥ the previous known-good
      run's count (a flat/falling count is a silent regression even at PASS)
- [ ] `curl -sf http://192.168.86.201:18085/health` and
      `curl -sf http://192.168.86.201:18086/health` both return healthy
- [ ] `curl -s http://192.168.86.201:18085/v1/introspection/manifest | jq '.discovery_errors // empty'`
      is empty
- [ ] Every service's `revision_label` in the health-gate JSON equals the
      cut commit SHA (`git rev-parse <ref>` at cut time)
- [ ] The receipt file exists and its `result` field is exactly `SUCCESS`
      (not `FAILED_ROLLED_BACK`, not `FAILED`):
      `ssh omni-201-ts 'cat ~/.omnibase/state/stability_lane_refresh/latest.json' | jq .result`

## Rollback

If any checklist item above fails post-hoc (e.g. you discover a regression
after the workflow already reported SUCCESS), the automatic rollback inside
`refresh_stability_lane.sh` has already run and reported its own terminal
`FAILED_ROLLED_BACK`/`FAILED` state in the same run — re-read that run's log
first; do not assume you need to hand-roll a recovery.

If the automated path is unavailable (script itself broken, or you need to
roll back a SUCCESS that later proved bad), use the manual preflight-tag
restore documented in `docs/runbooks/stability-lane-refresh.md` ("Manual
rollback (if the script itself is unavailable)") — it retags the 4 core
services back to their `preflight-<UTC>` anchor and force-recreates them.
Locate the correct anchor timestamp from the T0 log block (`=== Tag
preflight rollback anchor (<UTC>) ===`) or from
`docker images --format '{{.Repository}}:{{.Tag}}' | grep 'omnibase-infra-stability-test-.*:preflight-'`.
After a manual rollback, re-run the health-gate readback checklist above
against the rolled-back state before declaring the lane recovered.
