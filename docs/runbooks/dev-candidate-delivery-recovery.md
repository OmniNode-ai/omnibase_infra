# Dev-candidate delivery recovery (OMN-16906)

**Scope:** the `omnibase_infra` dev merge → onex-dev delivery chain
(`deliver-dev-candidate-to-staging.yml`, OMN-15796). Dev/staging lane only.
Nothing here touches prod, stability-test, or judge.

## What normally happens

A merge to `omnibase_infra` `dev` that touches `src/**`, `docker/**`,
`pyproject.toml`, `uv.lock`, or one of the three delivery workflow files fires
`deliver-dev-candidate-to-staging.yml`, which:

1. builds the runtime stability-candidate from *this* commit
   (`build-workspace-candidate-runtime.yml`),
2. builds the migration bundle from the *same* commit
   (`build-and-push-migrate-image.yml`), and
3. `repository_dispatch`es `omnibase-infra-dev-candidate` to `omninode_infra`
   with both digest-pinned refs, which `deploy-onex-staging.yml` pins into its
   ephemeral checkout and rolls.

Both builds must succeed before the dispatch fires. A half-built bundle — a
runtime rolled forward onto a schema that was never migrated — is never
announced.

## How you find out it did not happen

`dev-candidate-delivery-liveness.yml` runs every 30 minutes and fails with one
of these codes. Read the code first; it says which recovery below applies.

| code | meaning | go to |
| --- | --- | --- |
| `STARTUP_FAILURE` | the newest delivery run never compiled — no job, no log | §1 (workflow graph does not compile) |
| `NOT_FIRED` | a trigger-matching dev commit produced no run at all | §2 (manual delivery of a merged commit) |
| `NOT_DELIVERED` | the run finished and did not succeed | §2 (manual delivery of a merged commit) |
| `DELIVERY_STALLED` | the run has been in flight past the 3h allowance | §2 (manual delivery of a merged commit) |
| `NO_RUNS` | the guard could not read run history | the guard failed closed; check the trigger still exists before assuming an outage |

There is no passive signal. `startup_failure` creates no job, so nothing else in
either repo goes red — the staging deploy downstream keeps reporting green
because from its side nothing is wrong: it faithfully re-applies the digests it
was given. On 2026-08-27/28 that combination hid a two-day outage until a
migration fence that had already merged failed to apply on onex-dev.

## 1. `STARTUP_FAILURE` — the workflow graph does not compile

`startup_failure` is a *pre-run* validation failure. There is no job log to
read, and `gh run view` says only "This run likely failed because of a workflow
file issue". `actionlint` does **not** model these checks and will report clean.

**Bisect it; do not guess.** The workflow file exists on the default branch
(`dev`), so `workflow_dispatch --ref <branch>` runs *your branch's* version of
it. Stub the expensive jobs so a valid graph costs nothing:

```bash
# On a scratch branch, add to each `uses:` job in the delivery workflow:
#   if: false  # BISECT-HARNESS
# then push and dispatch. Verdict lands in ~10 seconds.
gh workflow run deliver-dev-candidate-to-staging.yml \
  --repo OmniNode-ai/omnibase_infra --ref <scratch-branch>
sleep 25
gh run list --workflow deliver-dev-candidate-to-staging.yml \
  --branch <scratch-branch> --limit 3 \
  --json databaseId,status,conclusion,createdAt
```

`conclusion: startup_failure` = graph still broken. `conclusion: skipped` = the
graph compiled and only the `if: false` stubs kept it from building. Change one
variable per dispatch.

**Known cause, 2026-08-27 (fixed, and ratcheted).** A called workflow's
`workflow_call` input declared `type: boolean` with `default: "false"` — a YAML
*string*. GitHub type-checks a callee's input defaults while compiling the
**caller's** graph, so every caller run died at startup while the callee's own
dispatch runs stayed green, which is what made it invisible. Control run
`33224162317` (`startup_failure`) vs. the same file with that one scalar
unquoted, run `33224392268` (compiled).

`scripts/ci/check_workflow_input_default_types.py` now fails pre-commit and CI
on that shape repo-wide, including under `workflow_dispatch` where GitHub
tolerates it — a latent bad default there is one `workflow_call` refactor away
from being fatal, which is exactly how this one was introduced.

## 2. `NOT_FIRED` or `NOT_DELIVERED` — manual delivery of a merged commit

This is the path used on 2026-08-28 to recover the OMN-16493 migration fence
(dev commit `7090f386f`, PR #2974) after the delivery workflow failed to fire.
It is also the **only** way to deliver a migration-only change while the runtime
repin is held, because the committed path is blocked by
`test_runtime_and_migration_images_share_one_source_rev` — that test enforces
SINGLE_SOURCE_REV_BUNDLE, so you cannot commit a migrate-image repin whose
source revision differs from the pinned runtime's.

Prefer re-running the real workflow when it is healthy; the manual path below
bypasses the bundle pairing the delivery workflow exists to guarantee, so only
reach for it when the runtime half is deliberately held.

### 2a. Try the sanctioned path first

```bash
gh workflow run deliver-dev-candidate-to-staging.yml \
  --repo OmniNode-ai/omnibase_infra --ref dev
```

If that completes green, stop — it dispatched the bundle itself.

### 2b. Migration-only recovery (runtime repin held)

**Step 1 — build the migration bundle from the merged dev commit.**

```bash
SHA=$(git rev-parse origin/dev)   # or the exact merged commit you must deliver
gh workflow run build-and-push-migrate-image.yml \
  --repo OmniNode-ai/omnibase_infra --ref dev -f git-ref="$SHA"
gh run list --workflow build-and-push-migrate-image.yml --limit 1 \
  --json databaseId,status,conclusion
```

Wait for `conclusion: success`, then read the digest-pinned ref out of the run's
`resolve-digest` step summary. It must contain `@sha256:` — a mutable tag is not
a deployable identity and the receiving workflow rejects it.

**Step 2 — send the `repository_dispatch` by hand.**

`runtime_image_ref` must be the digest currently pinned in `omninode_infra`'s
k8s manifests (you are *not* rolling the runtime forward here); `migrate_image_ref`
is the ref from step 1. Read the live runtime pin rather than retyping it.

```bash
jq -n \
  --arg runtime "<runtime-image-ref-currently-pinned>" \
  --arg migrate "<migrate-image-ref-from-step-1>" \
  --arg repo   "OmniNode-ai/omnibase_infra" \
  --arg sha    "$SHA" \
  '{
     event_type: "omnibase-infra-dev-candidate",
     client_payload: {
       runtime_image_ref: $runtime,
       migrate_image_ref: $migrate,
       source_repo: $repo,
       source_sha: $sha
     }
   }' \
  | gh api --method POST repos/OmniNode-ai/omninode_infra/dispatches --input -
```

**Step 3 — confirm the deploy actually carried it.**

```bash
gh run list --workflow deploy-onex-staging.yml \
  --repo OmniNode-ai/omninode_infra --limit 3 \
  --json databaseId,status,conclusion,createdAt
```

Green is necessary but not sufficient: a staging deploy reports green when it
re-applies *unchanged* pins. Confirm the migration jobs actually applied
something (`applied=0` across every job is the signature of the stale-image
failure this whole chain exists to prevent), and that the deploy's stamped
`source_sha` is the commit you meant to deliver.

**Step 4 — record it.** A hand-sent dispatch leaves no delivery-workflow run, so
the liveness guard will keep reporting `NOT_FIRED` for that commit until a later
trigger-matching merge delivers normally. That is correct — the commit was
delivered out of band. Note the manual dispatch on the ticket so the next reader
of the guard's output knows why.

## Why the runtime repin may be held

`test_runtime_and_migration_images_share_one_source_rev` refuses a committed pin
pair whose two images come from different revisions. When a migration must ship
ahead of a runtime rollout (OMN-16493), the committed path is therefore closed
by design and §2b is the sanctioned escape. Do not weaken that test to make the
committed path work — the pairing it enforces is the reason a runtime never
rolls forward onto an unmigrated schema.

## References

- `.github/workflows/deliver-dev-candidate-to-staging.yml` — the delivery chain
- `.github/workflows/dev-candidate-delivery-liveness.yml` — the guard that pages
- `scripts/ci/check_dev_candidate_delivery_liveness.py` — the verdict logic
- `scripts/ci/check_workflow_input_default_types.py` — the startup_failure ratchet
- `tests/ci/test_dev_candidate_delivery.py` — SINGLE_SOURCE_REV_BUNDLE invariants
- OMN-15796 (delivery chain), OMN-16493 (held runtime repin), OMN-16906 (this outage)
