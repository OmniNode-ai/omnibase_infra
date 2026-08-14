> **Navigation**: [Home](../index.md) > [Patterns](README.md) > Service Catalog & Install Model

# Service Catalog Architecture & Install Model

Moved out of `CLAUDE.md` (OMN-15198). This document covers the two ways `omnibase_infra`
is consumed (pip package vs local clone) and the typed service-catalog system that
generates all Docker infrastructure.

## Install Model

`omnibase_infra` ships as both a **pip-installable package** and a **cloneable repository**.

### Pip Package (library + runtime CLIs)

Install via pip for using `omnibase_infra` as a library dependency in other ONEX services,
or for running the bundled runtime CLIs:

```bash
pip install omnibase-infra
# or
uv add omnibase-infra
```

The bundled CLI entry points (`omni-infra`, `onex-runtime`, `onex-infra-test`,
`onex-git-hook-relay`, `onex-linear-relay`, `onex-status`, ...) are declared in
`pyproject.toml` under `[project.scripts]` — that table is the authoritative list.

### Local Clone (operational scripts)

A **local clone is required** to run the operational scripts in `scripts/`. These scripts
are **not bundled** in the pip package — they live only in the repository source tree:

```bash
git clone https://github.com/OmniNode-ai/omnibase_infra.git
cd omnibase_infra
uv sync
```

**Why scripts require a clone:** they scan the repository source tree directly
(e.g., `seed-infisical.py` iterates over `src/omnibase_infra/nodes/*/contract.yaml`),
write back to `~/.omnibase/.env`, or depend on shell tooling co-located with the repo.
This applies to all of `scripts/` — including `seed-infisical.py`,
`bootstrap-infisical.sh`, `provision-infisical.py`, `setup-infisical-identity.sh`,
`create_kafka_topics.py`, and `validate.py`.

### Decision Summary

| Use Case | Install Method |
|----------|---------------|
| Add `omnibase_infra` as a library dependency | `pip install omnibase-infra` |
| Run ONEX runtime services | `pip install omnibase-infra` → `onex-runtime` |
| Bootstrap Infisical (first-time setup) | Clone + `scripts/bootstrap-infisical.sh` |
| Seed Infisical from contracts | Clone + `uv run python scripts/seed-infisical.py` |
| Provision machine identities | Clone + `uv run python scripts/provision-infisical.py` |
| Run CI validators | Clone + `uv run python scripts/validate.py` |
| Develop nodes and handlers | Clone (full dev environment) |

> **Note on `sync-omnibase-env.py`**: This script is **not** part of
> `omnibase_infra`. Use the separately installed environment-sync tooling
> available in your workspace.

## Service Catalog

The service catalog is the authoritative source for all Docker infrastructure.
Every deployable unit is a typed YAML manifest; the compose file is generated, not hand-edited.

### Concepts

| Term | Description |
|------|-------------|
| **Manifest** | Typed YAML declaration of a single deployable service (`docker/catalog/services/<name>.yaml`) |
| **Bundle** | Named group of manifests deployed together (`docker/catalog/bundles.yaml`) |
| **Resolver** | Loads manifests + bundles, resolves transitive `includes`, returns `ResolvedStack` |
| **Generator** | Renders `ResolvedStack` → `docker-compose.generated.yml` |
| **Validator** | Checks that all `required_env` vars are present before start |

### Bundles

`docker/catalog/bundles.yaml` is the single authoritative source for bundle names,
their service membership, `includes` composition, and env injection — read it rather
than any copied table (copied tables drift). Key structural facts:

- **Transitive resolution**: bundles compose via `includes`; the resolver expands all
  `includes` before collecting services (e.g. `runtime` pulls in `core` and its
  sub-bundles; `tracing` pulls in `observability`).
- **Incremental rollout**: the `runtime` platform is split into sub-bundles so operators
  can deploy `onex up runtime-core` (no new secret requirements) to pick up correctness
  fixes without also enabling integrations that need new secrets. Once secrets are seeded
  (Infisical or `~/.omnibase/.env`), bring up the remaining sub-bundles individually.
- **Env injection**: each bundle may declare `inject_env` (hardcoded values injected into
  generated compose) and `inject_required_env` (vars that must be present in the operator
  environment at start time).

### onex CLI Commands

The `onex` CLI (`src/omnibase_infra/docker/catalog/cli.py`) is the primary operator interface.

```bash
# Generate compose file for one or more bundles
uv run python -m omnibase_infra.docker.catalog.cli generate core
uv run python -m omnibase_infra.docker.catalog.cli generate runtime memgraph

# Validate env completeness before starting
uv run python -m omnibase_infra.docker.catalog.cli validate runtime

# Start a bundle (generate + validate + docker compose up)
uv run python -m omnibase_infra.docker.catalog.cli up core
uv run python -m omnibase_infra.docker.catalog.cli up runtime memgraph tracing

# Stop a running bundle
uv run python -m omnibase_infra.docker.catalog.cli down core
```

The shell functions `infra-up` (→ `onex up core`), `infra-up-runtime` (→ `onex up runtime`),
`infra-up-memory` (→ `onex up runtime memgraph`), and `infra-down` (defined in `~/.zshrc`)
are backwards-compatible wrappers around `onex up/down`. They remain the preferred operator
interface — do not bypass them with raw `docker compose -f <path>`.

### Adding a New Service

1. Create `docker/catalog/services/<name>.yaml` using an existing manifest as template.
2. Set `layer` to one of: `infrastructure`, `runtime`, `observability`, `auth`, `secrets`.
3. Declare all `required_env` vars that the container needs from the operator environment.
4. Add hardcoded container-internal addresses under `hardcoded_env` (never pass host-side env vars for internal addressing).
5. Add the service name to the appropriate bundle(s) in `docker/catalog/bundles.yaml`.
6. Run `uv run python -m omnibase_infra.docker.catalog.cli validate <bundle>` to confirm env contract.

### Env Var Contract

Three categories of env vars in the catalog:

| Category | Location | Behavior |
|----------|----------|----------|
| `required_env` | Per-manifest YAML | Must be set in operator env; validated before start |
| `hardcoded_env` | Per-manifest YAML | Container-internal addresses; never overrideable |
| `inject_env` | Per-bundle in `bundles.yaml` | Injected only when that bundle is selected |

**Rule**: Container-to-container addresses (e.g. `redpanda:9092`, `valkey:6379`) must live
in `hardcoded_env`, never in `required_env`. Operator-supplied secrets (`POSTGRES_PASSWORD`,
API keys) belong in `required_env`.
