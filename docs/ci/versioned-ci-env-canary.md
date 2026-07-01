# Versioned CI Environment Canary

`omnibase_infra` serves as the canary for host-local, immutable CI
dependency environments on the trusted self-hosted runner pool.

## Problem

The prior CI shape made every Python-heavy job build its own dependency
environment. Most jobs called `setup-python-uv` with `cache-enabled: "false"`,
which expanded to `uv sync --no-cache`. Under queue pressure this repeatedly
resolved and downloaded the same locked dependencies from the same runner host.
That is unlike live infra, where a runtime uses a prepared environment.

## Canary Contract

Trusted CI jobs now opt into a shared dependency environment:

```text
/home/runner/.cache/omni/ci-envs/omnibase_infra/<digest>/.venv
```

The digest includes:

- `pyproject.toml`
- `uv.lock`
- Python version
- uv version
- shared environment install arguments
- the setup action and CI environment scripts
- runner platform

The environment is built at its final digest path under `flock`, marked ready
by writing `manifest.json` only after install succeeds, and made read-only.
Each checkout gets a local `.venv` symlink to the published env and a
runner-temp `uv` wrapper that turns ordinary `uv run <tool> ...` calls into
direct execution from `.venv/bin`. Commands with extra uv-run options still
delegate to the real uv binary. Jobs set `UV_PROJECT_ENVIRONMENT=<checkout>/.venv`,
`UV_NO_SYNC=1`, and `PYTHONPATH=<runner-temp metadata>:<checkout>/src`. The
runner-temp metadata is generated from the checkout `pyproject.toml` so
entry-point discovery sees the current PR without installing it into the shared
env. The dependency environment is shared, but the authoritative source under
test remains the current checkout.

Public fork PRs do not use the host-local environment and keep isolated setup.

## Operating Rules

- Do not run `uv sync` into an existing shared env from ordinary CI jobs.
- Jobs that intentionally run `uv sync` or `uv pip install` after setup must set
  `shared-env-enabled: "false"` and use an isolated writable environment.
- Do not install the checked-out project into the shared env; use
  `--no-install-project` so PR source cannot be stale.
- Generate checkout metadata outside the repo so entry-point discovery sees the
  PR's `pyproject.toml` without polluting clean-root checks.
- Build the shared env with bounded retries because a single transient git or
  wheel fetch failure would otherwise poison every job waiting on the digest.
- Do not enable `setup-uv` cache save for direct jobs that intentionally run
  `uv sync --no-cache`; the post-job cache upload can outlive the actual check
  and cancel merge-group CI while adding no useful reuse.
- Do not replace a real workspace `.venv`; the canary only manages the checkout
  `.venv` when it is absent or already a symlink to a shared env.
- Keep the `uv` wrapper narrow: only plain `uv run <tool> ...` is redirected to
  `.venv/bin`; other uv operations continue to use the real uv binary. Keep the
  wrapper under runner temp space so clean-root checks do not see generated
  checkout files.
- Include dependency groups in the canary env (`--all-groups`) so CI tools such
  as `mypy`, `ruff`, and `pytest` are present without per-job resolution.
- Bump the environment by changing lockfile/tool inputs, not by mutating an
  existing digest path.
- Keep old digest directories until active PRs using them have drained.
- If a shared env is suspected corrupt, remove only that digest directory after
  disabling affected runs or after they complete.
- `/home/runner/.cache/omni/ci-envs` is the canary root because the runner user
  can create it without sudo. A future ops hardening step can move the root to
  `/opt/omni/ci-envs` after that path is provisioned with the runner user as
  owner.

## Follow-Up Shape

After the infra canary proves stable, replicate the same pattern per repo:

```text
/home/runner/.cache/omni/ci-envs/<repo>/<digest>/.venv
```

Repos with incompatible dependency graphs get separate env roots. Shared base
layers are fine; shared mutable dependency environments are not.

## Versioned Runner Image Contract

The canary above proved the host-local prebuilt env. This work graduated it from
an infra canary to a versioned **runner image contract**: the repo-specific
prebuilt `.venv` is baked into the runner image, and the image "version" is a
**binding, not a human label**.

### Bound identity, not a label

`docker/runners/runner-image.lock.json` records the bound identity. The
`identity_digest` folds, into one reproducible value:

- the pinned base image digest,
- the dependency manifest (`pyproject.toml` + `uv.lock`),
- the Python version,
- the uv version,
- the runner / gh / kubectl versions,
- the shared CI env (canary) digest from `ci_env_digest.py`, and
- the integer `image_version`.

So "runner image v14" is a reproducible artifact: change any component and the
digest changes. `scripts/ci/runner_image_identity.py` is the generator and the
assertion:

```bash
# Recompute and rewrite the lock after bumping any input (e.g. uv.lock).
PYTHONPATH=scripts/ci python3 scripts/ci/runner_image_identity.py --mode generate
# Fail fast if the committed lock is stale (wired into release + build).
PYTHONPATH=scripts/ci python3 scripts/ci/runner_image_identity.py --mode verify
# Print the machine-readable startup-evidence line.
PYTHONPATH=scripts/ci python3 scripts/ci/runner_image_identity.py --mode emit
```

`scripts/ci/build_runner_image.sh` verifies the lock, stamps the bound identity
into the image via build args (`OMNI_RUNNER_IMAGE_IDENTITY`,
`OMNI_RUNNER_IMAGE_VERSION`), bakes the prebuilt env, and tags the image.

### Startup evidence on every job

Every CI job that prepares Python via `setup-python-uv` emits the bound identity
into its startup evidence (`$GITHUB_STEP_SUMMARY` and
`OMNI_RUNNER_IMAGE_IDENTITY_EVIDENCE` in `$GITHUB_ENV`). The reusable
`.github/actions/emit-runner-identity` action exposes the same emission for
non-Python jobs. Image-drift debugging reads the recorded digest instead of
guessing.

### Zero `uv sync` on the happy path

With the env baked into the image and the canary symlink wiring active, the
happy-path job resolves **zero `uv sync`**: `UV_NO_SYNC=1` is published, the
workspace `.venv` is a symlink to the prebuilt env, and `uv run <tool>` is
redirected to `.venv/bin`. A `uv sync` on the happy path is a regression.

### Mutating jobs must opt out explicitly

Jobs that intentionally mutate the Python environment after setup (`uv sync`,
`uv pip install`, editable sibling installs) must set
`shared-env-enabled: "false"` so they get an isolated writable environment and
never write into the shared prebuilt env. Enumerated mutating jobs:

| Job | Why it mutates | Opt-out |
|-----|----------------|---------|
| `compliance` | runs sweeps that `uv sync` extra groups | `shared-env-enabled: "false"` |
| `schema-handshake` | editable sibling repo installs | `shared-env-enabled: "false"` |
| `kafka-boundary-compat` | editable sibling repo installs | `shared-env-enabled: "false"` |
| `contract-compliance` | pinned `onex_change_control` git+https install | isolated `setup-uv` (no shared env) |

Adding a new mutating job means adding the `shared-env-enabled: "false"` opt-out
and a row here in the same PR.

### Bump on release

The image/env version is bumped by editing `image_version` in
`runner-image.lock.json` and regenerating the lock. The `release` workflow runs
`runner_image_identity.py --mode verify` and fails the release if the committed
lock is stale — a release cannot ship an un-regenerated bound identity.
