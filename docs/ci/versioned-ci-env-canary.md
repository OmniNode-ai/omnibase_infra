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

The environment is built at its final digest path under `flock`, marked ready
by writing `manifest.json` only after install succeeds, and made read-only.
Each checkout gets a local `.venv` symlink to the published env and a
runner-temp `uv` wrapper that turns ordinary `uv run <tool> ...` calls into
direct execution from `.venv/bin`. Commands with extra uv-run options still
delegate to the real uv binary. Jobs set `UV_PROJECT_ENVIRONMENT=<checkout>/.venv`,
`UV_NO_SYNC=1`, and `PYTHONPATH=<checkout>/src`. The dependency environment is
shared, but the authoritative source under test remains the current checkout.

Public fork PRs do not use the host-local environment and keep isolated setup.

## Operating Rules

- Do not run `uv sync` into an existing shared env from ordinary CI jobs.
- Jobs that intentionally run `uv sync` or `uv pip install` after setup must set
  `shared-env-enabled: "false"` and use an isolated writable environment.
- Do not install the checked-out project into the shared env; use
  `--no-install-project` so PR source cannot be stale.
- Build the shared env with bounded retries because a single transient git or
  wheel fetch failure would otherwise poison every job waiting on the digest.
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
