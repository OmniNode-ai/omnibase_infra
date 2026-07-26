# CLAUDE.md - Omnibase Infrastructure

> **Python**: 3.12+ | **Framework**: ONEX Infrastructure
>
> **Shared standards**: See **`~/.claude/CLAUDE.md`** for Python/Git/Testing standards, PEP 604 type unions, architecture principles, environment configuration, infrastructure topology, PostgreSQL, Kafka/Redpanda, Docker networking, LLM endpoints, and environment variables. Those rules apply to this repo and are not repeated here.

---

## Repo Invariants

These are non-negotiable architectural truths:

- **Nodes are declarative** - `node.py` extends base class with NO custom logic
- **Handlers own logic** - Business logic lives in handlers, not nodes
- **Reducers are pure** - `delta(state, event) -> (new_state, intents[])` with no I/O
- **Orchestrators emit, never return** - ORCHESTRATOR nodes cannot return `result`
- **Contracts are source of truth** - YAML contracts define behavior, not code
- **Unidirectional flow** - EFFECT → COMPUTE → REDUCER → ORCHESTRATOR, never backwards
- **Container injection** - All services use `ModelONEXContainer` for DI

---

## Non-Goals

We explicitly do **NOT** optimize for:

- **Backwards compatibility** - This repo has no external consumers. Schemas, APIs, and interfaces may change without deprecation periods. No `_deprecated` suffixes, no shims, no compatibility layers.
- **Convenience over correctness** - Contract violations fail loudly
- **Business logic in nodes** - Nodes coordinate; handlers compute
- **Dynamic runtime behavior** - All behavior must be contract-declared
- **Versioned directories** - NEVER create `v1_0_0/`, `v2/` directories; version through `contract.yaml` fields only

**When you see deprecated or unused code: DELETE IT.** Do not leave it "for reference", comment it out, add deprecation warnings, create compatibility shims, or keep old function signatures with forwarding.

---

## Install Model & Service Catalog

`omnibase_infra` ships as both a pip package (library + runtime CLIs; entry points are
declared in `pyproject.toml [project.scripts]`) and a cloneable repo (the operational
`scripts/` are NOT bundled in the pip package — they need a clone). All Docker
infrastructure is generated from typed YAML manifests: `docker/catalog/services/*.yaml`
grouped by `docker/catalog/bundles.yaml`, driven by the `onex` CLI
(`src/omnibase_infra/docker/catalog/cli.py` — `generate` / `validate` / `up` / `down`).
Never hand-edit the generated compose file; never bypass the catalog with raw
`docker compose -f <path>`.

Full walkthrough (install decision table, bundle composition, adding a new service):
`docs/patterns/service_catalog.md`.

**Env var rule (trap)**: Container-to-container addresses (e.g. `redpanda:9092`,
`valkey:6379`) must live in `hardcoded_env`, never in `required_env`. Operator-supplied
secrets (`POSTGRES_PASSWORD`, API keys) belong in `required_env`.

---

## Quick Reference

```bash
uv sync && pre-commit install             # Setup
uv run pytest tests/ -n auto              # Tests (parallel)
uv run pytest tests/ -m unit              # Unit only; -m integration for integration
uv run mypy src/omnibase_infra/           # Type checking
uv run ruff check src/ tests/             # Linting
pre-commit run --all-files                # All hooks
```

Coverage minimum is enforced via `fail_under` in `pyproject.toml` — read it there, don't
trust a copied number.

## SPDX Headers

All source files in `src/`, `tests/`, `scripts/`, `examples/` require MIT SPDX headers.
Canonical spec: `omnibase_core/docs/conventions/FILE_HEADERS.md`

- Stamp missing headers: `onex spdx fix src tests scripts examples`
- Check without writing: `onex spdx fix --check src tests scripts examples`
- Bypass a file: add `# spdx-skip: <reason>` in the first 10 lines

---

### Git Commit Rules (repo-specific additions)

> `--no-verify` and hook rules: see `~/.claude/CLAUDE.md` Git Standards.

- **NEVER use `--no-gpg-sign`** unless explicitly requested
- **NEVER run git commits in background mode**

---

## Agent Behavioral Rules

### Autonomous mode safety rails

- Never disable pre-commit hooks, CI checks, or type checkers to make code pass. Fix the code instead.
- Never write state files to `~/.claude/`; use the workspace `.onex_state/` directory.
- Friction logs go under `.onex_state/friction/` for external observability.

### Contract-first topic definitions

Kafka topics and event schemas belong in contract YAML files, not hardcoded in
application code. This repo is the primary home of ONEX node contracts.

When adding a new Kafka topic:
1. Declare it in the node's contract YAML under `event_bus.publish_topics` or `subscribe_topics`
2. Add the topic to the relevant `topics.yaml` skill file if it is a skill-emitted topic
3. Reference the contract-declared topic name in code via the contract loader
4. Never hardcode topic strings like `"onex.evt.foo.bar.v1"` in Python modules — the
   `arch-invariants` job in `.github/workflows/ci.yml` fails CI on hardcoded topic strings

---

## Architecture: Four-Node Pattern

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   EFFECT    │───▶│   COMPUTE   │───▶│   REDUCER   │───▶│ORCHESTRATOR │
│ External I/O│    │  Transform  │    │  FSM State  │    │  Workflow   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Data Flow**: Unidirectional left-to-right. No backwards dependencies.

| Node | Contract Type | Purpose | Primary Output |
|------|--------------|---------|----------------|
| **EFFECT** | `EFFECT_GENERIC` | External I/O (APIs, DB, files) | `events[]` |
| **COMPUTE** | `COMPUTE_GENERIC` | Pure data transformation | `result` (required) |
| **REDUCER** | `REDUCER_GENERIC` | FSM state management | `projections[]` |
| **ORCHESTRATOR** | `ORCHESTRATOR_GENERIC` | Workflow coordination | `events[]`, `intents[]` |

Base classes: `from omnibase_core.nodes import NodeEffect, NodeCompute, NodeReducer, NodeOrchestrator`.
Layering: `omnibase_core` = archetypes/models/enums, `omnibase_spi` = protocols, `omnibase_infra` = implementations.

---

## Declarative Nodes

**ALL nodes MUST be declarative - no custom Python logic in node.py**

1. Extend the base class from `omnibase_core.nodes`
2. Use `container: ModelONEXContainer` for dependency injection (call `super().__init__(container)`)
3. Define all behavior in `contract.yaml` (handlers, routing, workflows)
4. `node.py` contains ONLY the class definition extending base - no custom logic

For the canonical node directory layout and required contract fields, copy an existing
node under `src/omnibase_infra/nodes/` rather than working from a prose description.

---

## Handler System

Two protocols: `ProtocolHandler` (request/response I/O: `ModelProtocolRequest` →
`ModelProtocolResponse`) and `ProtocolMessageHandler` (dispatch: `ModelEventEnvelope` →
`ModelHandlerOutput`).

Two routing strategies in `handler_routing` contract blocks:
- `payload_type_match` — routes on event payload model type (orchestrator handlers)
- `operation_match` — routes on envelope operation (infrastructure handlers)

See `docs/patterns/operation_routing.md` and existing node contracts for the YAML shape.
Handlers expose `handler_type` (`EnumHandlerType`) and `handler_category`
(`EnumHandlerTypeCategory`) classification properties.

### Handler No-Publish Constraint

**Handlers MUST NOT have direct event bus access** - only orchestrators may publish events.

| Constraint | Verification |
|------------|--------------|
| No bus parameters | `__init__`, `handle()` signatures |
| No bus attributes | No `_bus`, `_event_bus`, `_publisher` |
| No publish methods | No `publish()`, `emit()`, `send_event()` |

---

## Intent Model Architecture

Reducers emit intents that orchestrators route to Effect layer nodes.

- **Two layers**: a typed payload model (e.g. `ModelPayloadConsulRegister`, with its own
  `intent_type` literal field like `"consul.register"`) wrapped in the standard
  `ModelIntent` envelope with `intent_type="extension"`.
- **Routing**: the Effect layer routes on `payload.intent_type`, not the outer envelope.
- **Target URI convention**: `{protocol}://{resource}/{identifier}` (e.g.
  `postgres://node_registrations/{node_id}`, `consul://service/{service_name}`).
- **Trap**: infra payload models extend `BaseModel` directly (repo convention — see
  `docs/standards/ONEX_TERMINOLOGY.md`). Do NOT justify this as "`ModelIntentPayloadBase`
  was removed in omnibase_core 0.6.2" — that claim is false: the class exists in live
  core (`omnibase_core.models.reducer.payloads`, present at v0.6.2 and every version
  since) and bases core's own closed-set intent payloads. The real 0.6.2 change was
  `ModelIntent.payload: dict[str, Any]` → `ProtocolIntentPayload` (OMN-1256).

---

## Error Handling

The infra error hierarchy roots at `RuntimeHostError` (itself under `ModelOnexError`);
see `omnibase_infra.errors` for the concrete tree — pick the narrowest matching class
(`InfraConnectionError`, `InfraTimeoutError`, `RepositoryError`, `ContainerWiringError`, ...).

### Error Context Factory (MANDATORY)

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Auto-generates correlation_id; pass correlation_id=... to propagate an existing one
context = ModelInfraErrorContext.with_correlation(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
)
raise InfraConnectionError("Failed to connect", context=context) from e
```

### Error Sanitization

**NEVER include**: passwords, API keys, PII, connection strings with credentials.
Use `sanitize_error_message()` / `sanitize_secret_path()` / `sanitize_consul_key()` from
`omnibase_infra.utils.util_error_sanitization`.

---

## Infrastructure Patterns

Transport types are enumerated in `EnumInfraTransportType` — read the enum, not a copied
table.

### Circuit Breaker & Dispatcher Resilience

Use `MixinAsyncCircuitBreaker` for external service integrations (see
`docs/patterns/circuit_breaker_implementation.md`).

**Dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap
dispatchers with circuit breakers. Each dispatcher implements `MixinAsyncCircuitBreaker`
for external service calls, configures thresholds appropriate to its transport type, and
raises `InfraUnavailableError` when the circuit opens. See
`docs/patterns/dispatcher_resilience.md`.

### Correlation ID Rules

Always propagate from incoming requests; auto-generate with `uuid4()` if missing; include
in all error context.

---

## Pydantic Model Standards

File/class naming is mechanical: `model_<name>.py` → `Model<Name>`, `enum_<name>.py` →
`Enum<Name>`, and likewise for `adapter_`, `dispatcher_`, `mixin_`, `protocol_`,
`service_`, `store_`, `validator_` prefixes. Node registries: `registry_infra_<name>.py`
→ `RegistryInfra<Name>`.

```python
# Standard ConfigDict (most common)
model_config = ConfigDict(
    frozen=True,           # Immutability for thread safety
    extra="forbid",        # Strict validation
    from_attributes=True,  # ORM/pytest-xdist compatibility
)
```

Prefer empty string over `None` for optional strings; use `default_factory` for
collections; use `tuple` for collections on frozen models. A model overriding `__bool__`
must document the non-standard behavior in a `Warning` docstring section.

---

## Testing and CI

Test tree: `tests/{unit,integration,chaos,replay,performance}` — directory placement
auto-applies the matching pytest marker; the full marker list lives in `pyproject.toml`.
Service-dependent markers (`consul`, `postgres`, `kafka`) and `slow`/`serial` are manual.

```bash
uv run pytest tests/ -n auto     # Parallel
uv run pytest tests/ -n 0 -xvs   # Debug mode (no parallelism)
```

### Runtime Startup is a First-Class CI Gate

Any PR that touches `auto_wiring/`, `service_kernel.py`, handler `__init__` signatures, or kernel-level registration MUST include a test that:

1. Loads the real contract manifest from disk.
2. Runs `wire_from_manifest` with the same args the kernel passes in production.
3. Asserts zero failures for required handlers.

CI must additionally boot `omninode-runtime` in a compose sandbox and assert:

- the container reaches Docker healthy state within the compose `start_period` configured for `omninode-runtime` in `docker/docker-compose.infra.yml`, and
- `RestartCount == 0` at the health-ready checkpoint — not a fixed wall-clock window.

Ad-hoc short timeouts are forbidden: any PR that shortens the gate below `start_period` must also update the compose healthcheck in the same PR, with justification.

**Forbidden:** aspirational integration gates — "there's a test file but it uses fake handlers." The boot must actually happen against real handlers.

**Strict-mode invariants** that tighten acceptance land AFTER all downstream consumers are compliant, not before. If a strict gate must ship first, it ships behind an env flag (default OFF) and is flipped in a separate PR once compliance is merged.

---

## Contract-Driven Config Discovery

Infisical-backed configuration management: config requirements are extracted from ONEX
contract YAMLs (`metadata.transport_type`, handler-level transport types, and
`dependencies[].type == "environment"`) and resolved from Infisical at runtime.
Implementation lives under `src/omnibase_infra/runtime/config_discovery/`.

- **Path convention**: shared config at `/shared/<transport>/KEY`, per-service at
  `/services/<service>/<transport>/KEY`.
- **Opt-in (trap)**: config prefetch only runs when `INFISICAL_ADDR` is set in the
  environment. Without it, the runtime falls back to standard environment variable
  resolution — local development works without Infisical.
- **Bootstrap**: `scripts/bootstrap-infisical.sh` orchestrates the full first-time
  sequence (identity provisioning + seeding); `scripts/seed-infisical.py` is safe by
  default and supports `--dry-run`. The full pre-Infisical env reference is preserved in
  `docs/env-example-full.txt`.

---

## Common Pitfalls

### Do NOT

1. **Skip base class initialization** — node `__init__` without `super().__init__(container)` is wrong
2. **Add custom logic to declarative nodes** — no `process()`/business methods on node classes
3. **Return result from ORCHESTRATOR**
   ```python
   return ModelHandlerOutput.for_orchestrator(result={"status": "done"})  # ValueError!
   ```
4. **Base infra payload DTOs on ModelIntentPayloadBase** — extend `BaseModel` directly
   (repo convention). The class itself was never removed: it lives in
   `omnibase_core.models.reducer.payloads` and bases core's closed-set intent payloads.
   ```python
   class ModelPayloadExample(BaseModel):  # infra convention; do not subclass core's base
   ```

### DO

1. Always call `super().__init__(container)` in node constructors
2. Use protocol names for DI: `container.get_service("ProtocolEventBus")`
3. Keep nodes declarative - all logic in handlers
4. Use `ModelInfraErrorContext.with_correlation()` for error context

---

## Handler Plugin Loader

The runtime uses plugin-based handler loading from YAML contracts
(see `docs/patterns/handler_plugin_loader.md`).

- **Contract file precedence**: `handler_contract.yaml` (dedicated, preferred) vs
  `contract.yaml` (general contract with handler fields).
- **FAIL-FAST (trap)**: when both files exist in the same directory, the loader raises
  `AMBIGUOUS_CONTRACT_CONFIGURATION` — it does not silently pick one.
- **Security**: restrict loading with `HandlerPluginLoader(allowed_namespaces=[...])`
  in production.

---

## Release Process

`src/omnibase_infra/runtime/version_compatibility.py` checks at runtime that installed
`omnibase_core` / `omnibase_spi` versions match the constraints in `pyproject.toml`.
`VERSION_MATRIX` is derived automatically from `pyproject.toml` at import time;
`_FALLBACK_MATRIX` (used when no source tree is present) is kept in sync by
`scripts/update_version_matrix.py`.

**Dependency bump checklist:**

1. Update `pyproject.toml` bounds, then `uv sync`.
2. `uv run pytest tests/unit/runtime/test_version_compatibility.py` — `test_matrix_matches_pyproject` catches drift.
3. The release workflow runs `scripts/update_version_matrix.py --check` as a pre-build gate (run without `--check` to update the fallback in-place).

**What NOT to do:** Do not manually edit `VERSION_MATRIX` or `_FALLBACK_MATRIX` in
`version_compatibility.py`. Let `pyproject.toml` be the single source of truth.

---

## Documentation

| Topic | Document |
|-------|----------|
| Any Type Enforcement | `docs/decisions/adr-any-type-pydantic-workaround.md` |
| Container DI | `docs/patterns/container_dependency_injection.md` |
| Error Handling | `docs/patterns/error_handling_patterns.md` |
| Error Recovery | `docs/patterns/error_recovery_patterns.md` |
| Circuit Breaker | `docs/patterns/circuit_breaker_implementation.md` |
| Dispatcher Resilience | `docs/patterns/dispatcher_resilience.md` |
| Protocol Patterns | `docs/patterns/protocol_patterns.md` |
| Security Patterns | `docs/patterns/security_patterns.md` |
| Handler Plugin Loader | `docs/patterns/handler_plugin_loader.md` |
| Mixin Dependencies | `docs/patterns/mixin_dependencies.md` |
| Service Catalog & Install Model | `docs/patterns/service_catalog.md` |

---

## Branch Protection

Never assert branch-protection state from memory or docs — probe it. Before any
`gh api --method PUT .../branches/<branch>/protection` mutation, dry-run the audit:

```bash
bash scripts/audit-branch-protection.sh --repo <repo> --dry-run
```

For enforcement + merge-policy parity (MISSING gates, needs-closure, queue/strict drift):

```bash
uv run python scripts/audit_required_context_parity_cli.py report --owner OmniNode-ai
```

The declared policy lives in `scripts/enforcement_parity_manifest.yaml` (config-as-data:
`{repo → branch → {load_bearing_gates[], merge_policy}}`); assertion logic is in
`scripts/audit_branch_protection_lib.py`. Both audits run on a schedule via
`.github/workflows/branch-protection-audit.yml` (the parity ratchet is report-only and
never mutates protection).

---

**Bottom Line**: Declarative nodes, container injection, contracts as source of truth. No backwards compatibility, no custom node logic.
