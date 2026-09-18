# Scoped effects image replacement

`deploy-runtime.sh --effects-plan FILE` is a separate, fail-closed deployment
path for the existing `runtime-effects` service in the dev Compose project
`omnibase-infra`. It is not a full deployment with optional phases disabled.
Operator authorization and coordination with the current runtime owner remain
required. These tools do not grant deployment authority.

## Boundary

The only running service this path recreates is `runtime-effects`. It does not
run full-lane source staging, builds, migration services, broker preparation,
core recreation, global deployment-registry updates, or retention pruning.
The existing service's normal startup still runs: this can perform database
fingerprinting, idempotent schema initialization, and broker subscription or
topic initialization. This is not a guarantee of zero database or broker I/O.

Admission requires unchanged dependency contents, shared startup files,
activation requirements, image runtime configuration, and effective Compose
configuration. It rejects incompatible candidates rather than introducing new
skip flags. The declared-topic scan is conservative, not an exhaustive dynamic
topic universe or a producer/reader compatibility census. A wire-model change
still needs its own reader and retained-record disposition before execution.

## Immutable inputs

Use the full 40-character commits for `omnibase_core`, `omnibase_compat`, and
`omnimarket`, plus a separate full infrastructure commit. Never substitute
`dev`, a branch, or a shared tag. Prepare owned, disposable clean source clones;
the builder does not fetch, check out, reset, or clean them. In particular, do
not point source preparation at canonical development clones.

The existing staging wrapper also accepts three independent selectors:

```bash
OMNI_HOME="$DEPLOY_SOURCE_CLONES" bash scripts/runtime_build/stage_workspace.sh \
  --repo-ref "omnibase_core=${CORE_SHA}" \
  --repo-ref "omnibase_compat=${COMPAT_SHA}" \
  --repo-ref "omnimarket=${MARKET_SHA}"
```

All three selectors must be supplied together. They cannot be combined with
the global deployment-ref selector or hotpatch/unpinned modes. Resolution of
every selected commit precedes any checkout; dirty or non-owned targets refuse.
The existing single-global-ref workflow remains available separately.

## Build a local candidate

`ModelEffectsCandidatePlan` in `build_effects_candidate.py` owns the JSON schema:

| Field | Required value |
| --- | --- |
| `schema_version` | `"1"` |
| `ticket_id`, `reason` | Actual rollout ticket and concrete justification |
| `base_image_id` | Full `sha256:` image ID of the current effects image |
| `source_root` | Absolute parent of the three clean sibling clones |
| `source_pins` | Object containing the three names and their full commits |
| `infra_source_root`, `infra_source_sha` | Clean infrastructure clone and full commit matching the base's installed infrastructure |

Before the helper executes, provision the frozen development environment from
the infrastructure repository. This is the only permitted dependency-resolution
step; it must occur before, rather than during, the offline candidate build.
`hatchling` is deliberately declared in the `dev` dependency group because
`--no-build-isolation` requires the backend in that environment.

```bash
uv sync --frozen --group dev
uv run --frozen --group dev python -c 'import hatchling; print(hatchling.__version__)'
```

Then run the helper from that already provisioned environment:

```bash
uv run --frozen python scripts/runtime_build/build_effects_candidate.py \
  --plan "$CANDIDATE_PLAN" --output "$NEW_CANDIDATE_DIRECTORY"
```

This previews source validation and local base-image identity; it does not
build, write an output directory, or prove image compatibility. Add `--execute`
to build the artifact. The output directory must not exist. An offline
`hatchling` backend must already be available; the helper itself performs no
dependency bootstrap or network access.

The builder derives a uniquely tagged candidate from the exact local base,
replaces only Market using an offline wheel, and verifies installed content.
Core, compatibility, infrastructure, other dependencies, and shared startup
requirements must remain byte-equivalent. Build provenance honestly records
`scoped-derived`, the base image, and the selected sources; it is not a clean-main
release. The builder does not deploy, publish, or overwrite shared image tags.
Keep the candidate receipt, exact image ID, and retained base artifact together.

## Preview and execute

`ModelEffectsDeployPlan` in `scoped_effects_plan.py` owns the deployment schema:

| Field | Required value |
| --- | --- |
| `schema_version` | `"1"` |
| `ticket_id`, `reason` | Actual rollout ticket and justification |
| `compose_project`, `service` | `"omnibase-infra"`, `"runtime-effects"` |
| `expected_container_id` | Full current effects container ID |
| `expected_image_id`, `candidate_image_id` | Distinct, full immutable image IDs |
| `compose_files` | Ordered absolute file chain from the active container's Compose labels |
| `compose_working_dir` | Absolute active Compose project directory |
| `expected_compose_sha256` | `compose_chain_sha256()` of that exact ordered chain |
| `source_pins`, `infra_source_sha` | Verified candidate source commits |
| `source_clones_root`, `hotpatch_ledger` | Absolute local clone parent and actual hotpatch ledger |
| `health_timeout_seconds` | Optional bounded readiness allowance, default/max 1800 |

The file-chain digest includes each absolute path and its content SHA-256,
encoded as canonical JSON. Use the supplied `compose_chain_sha256()` function;
do not hash concatenated YAML or expose resolved environment values. Capture
inputs immediately before admission; stale identities and source drift refuse.

```bash
bash scripts/deploy-runtime.sh --effects-plan "$EFFECTS_PLAN"
bash scripts/deploy-runtime.sh --effects-plan "$EFFECTS_PLAN" --execute
```

Preview is read-only and reports compatibility as unverified. Execution obtains
the shared deployment lock, probes the baseline/candidate in read-only,
network-disabled containers, checks actual installed live content, validates the
hotpatch ledger against candidate commits, and verifies existing declared topics.
It then recreates only effects with `--no-deps --no-build --pull never`.

Do not combine this mode with ordinary build/ref/service override variables,
`--restart`, `--cold`, `--force`, `--prod`, or `--profile`. It rejects those
combinations rather than silently changing their meaning. The Compose CLI must
match the version that created the active service so its configuration hash can
be checked consistently.

## Evidence and recovery

Execution stores a private plan, intent, immutable resolved Compose snapshot,
image-only overrides, and final receipt below
`~/.omnibase/infra/scoped-effects/<ticket>/<run>/`. Resolved configuration can
contain credentials: keep this directory private and never commit, paste, or
attach its contents wholesale. Forward replacement and rollback use the same
admitted snapshot, not subsequently edited environment or Compose source files.

Success requires the exact probed target to remain healthy on the candidate
image, with non-target identity and liveness unchanged. A failure attempts a
bounded restoration of the retained prior image only when ownership and runtime
state remain provable. The receipt distinguishes verified restoration from
failed or unproved recovery. A timed-out command's process group must be fenced
before recovery; an unfenced command forbids automatic rollback.

Both full and scoped deployment paths now refuse every existing deployment
lock, including one with a missing or apparently dead PID. An operator must
inspect ownership and interrupted-run evidence before recovering a stale lock.
Do not delete a lock to make a competing deployment proceed.

Local fixture tests prove tooling behavior, not a live rollout. Publish actual
runtime readback separately in the ticket's source-bound OCC evidence. The
scoped receipt neither rewrites the global deployment registry nor substitutes
for central OCC receipt-gate evidence.
