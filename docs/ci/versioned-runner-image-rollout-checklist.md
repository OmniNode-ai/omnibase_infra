# Versioned Runner Image Rollout Checklist (OMN-12568)

This is the per-repo migration checklist for moving a repository onto the
OMN-12567 **versioned runner image contract** — the runner image whose "version"
is a *binding* (base image digest + dependency manifest + Python version + uv
version + shared-env version + image version folded into one reproducible
`identity_digest`), not a human label.

OMN-12567 builds and binds the image; **OMN-12568 is the acceptance**: prove the
contract survives a *fresh runner recreation* on one repo, then roll the rest of
the fleet repo-by-repo using this checklist. Do **not** migrate all repos at
once — each repo's dependency graph is validated independently.

## Source of truth

| Artifact | Path |
|----------|------|
| Bound identity lock | `omnibase_infra:docker/runners/runner-image.lock.json` |
| Identity generator / verifier | `omnibase_infra:scripts/ci/runner_image_identity.py` (OMN-12567) |
| Image build script | `omnibase_infra:scripts/ci/build_runner_image.sh` (OMN-12567) |
| Per-job identity emitter | `omnibase_infra:.github/actions/emit-runner-identity/action.yml` (OMN-12567) |
| **Runner validation script** | `omnibase_infra:scripts/ci/validate_runner_image.py` (OMN-12568) |
| **Runner validation wrapper** | `omnibase_infra:scripts/ci/validate_runner_image.sh` (OMN-12568) |
| Canary contract | `omnibase_infra:docs/ci/versioned-ci-env-canary.md` (OMN-12564) |

## Pre-rollout gate (do this once, on `omnibase_infra`)

The canary repo is `omnibase_infra`. Before any other repo migrates, the image
contract must be proven on a **freshly recreated** runner.

- [ ] Build and publish the versioned image (gated live runtime step — the user
      runs this on the runner host):
      `scripts/ci/build_runner_image.sh --tag <registry>/omninode-runner:v<N>`.
- [ ] **Recreate one runner from scratch** on the new image (gated live runtime
      step). Warm-runner state does not count — recreation is the contract under
      test.
- [ ] Run the validation script on the freshly recreated runner and confirm
      `RESULT: GREEN`:
      `scripts/ci/validate_runner_image.sh --json`.
- [ ] Confirm at least one real CI job ran green on the recreated runner and the
      job's startup evidence shows the bound `runner_image_identity` (from the
      `emit-runner-identity` action) matching the committed lock.

Only after this gate is green does the fleet rollout below begin.

## Per-repo migration steps

Run these for **one repo at a time**. A repo is "migrated" only when a CI job has
run green on a freshly recreated runner using the versioned image.

- [ ] **Confirm dependency-graph compatibility.** The shared prebuilt env is keyed
      on `pyproject.toml` + `uv.lock` + Python/uv versions. If the repo's graph is
      incompatible with the infra canary env, it gets its own env root
      (`/home/runner/.cache/omni/ci-envs/<repo>/<digest>`); do not force-share an
      env across incompatible graphs.
- [ ] **Wire the identity emission.** Ensure every Python-prep job uses the
      `setup-python-uv` action (which emits the bound identity), or add the
      reusable `emit-runner-identity` action to non-Python jobs. Image-drift
      debugging requires the identity in startup evidence.
- [ ] **Enumerate mutating jobs and opt them out** (see the opt-out table below).
      Any job that runs `uv sync` / `uv pip install` / editable sibling installs
      after setup must set `shared-env-enabled: "false"` so it gets an isolated
      writable env and never writes into the shared prebuilt env.
- [ ] **Vendor or reference the validation script.** Either call
      `omnibase_infra:scripts/ci/validate_runner_image.{py,sh}` from a shared
      action, or copy it into the repo's `scripts/ci/`. The lock file the script
      reads is the baked `/etc/omni/runner-image.lock.json` on the image.
- [ ] **Recreate a runner and validate** (gated live runtime step). Run
      `validate_runner_image.sh` on the recreated runner; require `RESULT: GREEN`.
- [ ] **Run the repo's full CI on the recreated runner** and confirm green,
      including the Receipt-Gate job, before marking the repo migrated.
- [ ] **Record the migration** in the rollout log with: repo, image version `v<N>`,
      recorded `identity_digest`, and the green CI run URL.

## Mutating-job opt-outs (carried from OMN-12567)

These jobs intentionally mutate the Python environment and **must** opt out of
the shared prebuilt env. Adding a new mutating job means adding its opt-out and a
row here in the same PR.

| Job | Why it mutates | Opt-out |
|-----|----------------|---------|
| `compliance` | runs sweeps that `uv sync` extra groups | `shared-env-enabled: "false"` |
| `schema-handshake` | editable sibling repo installs | `shared-env-enabled: "false"` |
| `kafka-boundary-compat` | editable sibling repo installs | `shared-env-enabled: "false"` |
| `contract-compliance` | pinned `onex_change_control` git+https install | isolated `setup-uv` (no shared env) |

## Explicit opt-outs (repos / jobs that should NOT migrate)

Migration is not universal. Record an explicit opt-out (with reason) rather than
silently skipping:

- [ ] **Public fork PRs** — never use the host-local shared env or the baked
      image env; they keep isolated `setup-python-uv` (trust boundary). This is a
      permanent opt-out, not a migration gap.
- [ ] **Repos on GitHub-hosted runners only** — the versioned image targets the
      *self-hosted trusted pool*. A repo whose CI runs exclusively on
      `ubuntu-latest` hosted runners does not consume the image; mark it opt-out.
- [ ] **Repos with no Python CI** (e.g. pure docs / PHP / static-site repos) —
      no prebuilt env to bake; opt out and note "no Python jobs".
- [ ] **Jobs that require a writable global env** (anything doing
      `uv pip install` into the system env) — opt out at the job level via
      `shared-env-enabled: "false"`, do not migrate the whole repo around them.

Each opt-out must be recorded with: repo/job, reason, and the date reviewed —
so an opt-out is an auditable decision, not an unexplained gap.

## Validation script contract

`scripts/ci/validate_runner_image.py` asserts three things on the runner and
exits non-zero on any required failure:

1. **`image_identity`** — the lock file (`/etc/omni/runner-image.lock.json`,
   falling back to the repo-local lock) is present and internally consistent; the
   baked `OMNI_RUNNER_IMAGE_IDENTITY` matches the recorded `identity_digest`
   (no drift, not `unbound`). When `runner_image_identity.py` is present it
   recomputes the full binding and requires the recorded digest to match.
2. **`zero_uv_sync`** — the prebuilt shared CI env is baked under
   `OMNI_CI_ENV_ROOT` with a `manifest.json` ready marker, and `UV_NO_SYNC=1`
   holds, so the happy path resolves zero `uv sync`.
3. **`receipt_gate_readiness`** — `gh`, `git`, `jq`, `python3`, and `uv` are on
   the runner, so a queued PR's Receipt-Gate verification will not fail on a
   missing binary.

It is a **read-only** assertion — it never recreates, restarts, or registers a
runner. Recreation is a separate, operator-run live runtime step.

## Acceptance (OMN-12568)

- One repo (`omnibase_infra`) runs green on the versioned image **after fresh
  runner recreation** — proven by `validate_runner_image.sh` returning
  `RESULT: GREEN` on the recreated runner plus a green CI run.
- This rollout checklist exists with explicit opt-outs noted (this document).
