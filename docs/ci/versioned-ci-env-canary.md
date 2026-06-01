# Versioned CI Environment Canary

OMN-12564 makes `omnibase_infra` the canary for host-local, immutable CI
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

The environment is built under `flock`, published atomically, and made
read-only. Jobs set `UV_PROJECT_ENVIRONMENT` to the published `.venv`,
`UV_NO_SYNC=1`, and `PYTHONPATH=<checkout>/src`. The dependency environment is
shared, but the authoritative source under test remains the current checkout.

Public fork PRs do not use the host-local environment and keep isolated setup.

## Operating Rules

- Do not run `uv sync` into an existing shared env from ordinary CI jobs.
- Do not install the checked-out project into the shared env; use
  `--no-install-project` so PR source cannot be stale.
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
