# OMN-3988: Investigation — Non-Node Dirs Under `nodes/`

**Ticket:** OMN-3988
**Repo:** omnibase_infra
**Status:** Read-only investigation
**Date:** 2026-03-08

---

## 1. Scope

Three directories exist directly under `src/omnibase_infra/nodes/` that do **not** follow the canonical `node_<name>_<type>` naming convention:

| Directory | Type | Contents |
|---|---|---|
| `nodes/effects/` | Shared library | `NodeRegistryEffect` class + shared LLM/registry models |
| `nodes/handlers/` | Contract stubs | `contract.yaml` files only (no Python) |
| `nodes/reducers/` | Shared library | `RegistrationReducer` class + registration payload models |

This document investigates their import graph, loader implications, packaging, test dependencies, and proposed migration for OMN-3989.

---

## 2. Node Count (Before / After)

| Category | Count |
|---|---|
| Proper `node_*` prefixed dirs | 46 |
| Non-prefixed node dirs (`architecture_validator`, `contract_registry_reducer`) | 2 |
| Non-node dirs under `nodes/` (`effects/`, `handlers/`, `reducers/`) | 3 |
| **Total entries in `nodes/`** | **51** (plus `__init__.py`) |

**Node count does not change** in OMN-3989 because `effects/`, `handlers/`, and `reducers/` are not nodes — they are shared libraries and contract stubs living in the wrong location. Moving them is purely a relocation.

---

## 3. Detailed Content Inventory

### 3.1 `nodes/effects/`

**Python files (19):**

```
nodes/effects/__init__.py               — re-exports NodeRegistryEffect, ModelRegistryRequest, ModelRegistryResponse, ModelBackendResult
nodes/effects/contract.yaml             — ONEX contract for NodeRegistryEffect (name: "registry_effect")
nodes/effects/registry_effect.py        — NodeRegistryEffect class implementation
nodes/effects/protocol_postgres_adapter.py       — ProtocolPostgresAdapter protocol
nodes/effects/protocol_effect_idempotency_store.py — ProtocolEffectIdempotencyStore protocol
nodes/effects/store_effect_idempotency_inmemory.py — StoreEffectIdempotencyInmemory implementation
nodes/effects/README.md                 — documentation

nodes/effects/models/__init__.py        — re-exports all shared LLM + registry models
nodes/effects/models/model_effect_idempotency_config.py
nodes/effects/models/model_llm_inference_request.py   — canonical contract-level LLM request model
nodes/effects/models/model_llm_inference_response.py
nodes/effects/models/model_llm_message.py
nodes/effects/models/model_llm_tool_call.py
nodes/effects/models/model_llm_tool_choice.py
nodes/effects/models/model_llm_tool_definition.py
nodes/effects/models/model_llm_function_call.py
nodes/effects/models/model_llm_function_def.py
nodes/effects/models/model_llm_usage.py
nodes/effects/models/model_registry_request.py
nodes/effects/models/model_registry_response.py
nodes/effects/models/adapter_llm_usage_to_contract.py
```

**Key observation:** `nodes/effects/` is a **shared model library** used by multiple proper node dirs. It is not itself a self-contained node. The `NodeRegistryEffect` class here is a legacy holdover — there is a proper `node_registry_effect/` that imports protocols from `nodes/effects/` and re-exports the models through `node_registry_effect/models/__init__.py`.

### 3.2 `nodes/handlers/`

**No Python files.** Contains only YAML:

```
nodes/handlers/db/contract.yaml       — handler_db, EFFECT_GENERIC
nodes/handlers/graph/contract.yaml    — handler_graph, EFFECT_GENERIC
nodes/handlers/http/contract.yaml     — handler_http, EFFECT_GENERIC
nodes/handlers/intent/contract.yaml   — handler_intent, EFFECT_GENERIC
nodes/handlers/mcp/contract.yaml      — handler_mcp, EFFECT_GENERIC
```

These `contract.yaml` files describe **generic handler plugins** (not node-level contracts). They define `handler_routing` and `operation_bindings` sections that the `RuntimeContractConfigLoader` picks up via recursive glob.

No `__init__.py` files — this is a YAML-only directory.

### 3.3 `nodes/reducers/`

**Python files (9):**

```
nodes/reducers/__init__.py             — re-exports RegistrationReducer
nodes/reducers/registration_reducer.py — RegistrationReducer class (pure function reducer)

nodes/reducers/models/__init__.py
nodes/reducers/models/model_payload_postgres_upsert_registration.py
nodes/reducers/models/model_payload_postgres_update_registration.py
nodes/reducers/models/model_registration_ack_update.py
nodes/reducers/models/model_registration_heartbeat_update.py
nodes/reducers/models/model_registration_confirmation.py
nodes/reducers/models/model_registration_state.py
nodes/reducers/models/model_payload_ledger_append.py
```

**Key observation:** `RegistrationReducer` is a pure-function reducer. The canonical declarative version is `node_registration_reducer/`. The `nodes/reducers/` version is an older shared implementation used directly by multiple nodes and the runtime.

---

## 4. Import Graph

### 4.1 `nodes/effects/` — Importers

**Source files importing from `omnibase_infra.nodes.effects`:**

| Importer | What It Imports |
|---|---|
| `nodes/__init__.py` | `ModelBackendResult`, `ModelRegistryRequest`, `ModelRegistryResponse`, `NodeRegistryEffect` (via `nodes.effects`) |
| `nodes/effects/registry_effect.py` | Self-internal protocol imports |
| `nodes/node_llm_inference_effect/models/__init__.py` | All shared LLM models (re-exported) |
| `nodes/node_llm_inference_effect/models/model_llm_inference_request.py` | `ModelLlmToolChoice`, `ModelLlmToolDefinition` |
| `nodes/node_llm_inference_effect/handlers/handler_llm_ollama.py` | `ModelLlmInferenceResponse`, `ModelLlmMessage`, `ModelLlmUsage`, `ModelLlmToolCall` |
| `nodes/node_llm_inference_effect/handlers/handler_llm_openai_compatible.py` | All LLM shared models |
| `nodes/node_llm_inference_effect/handlers/bifrost/handler_bifrost_gateway.py` | `ModelLlmInferenceResponse` |
| `nodes/node_llm_inference_effect/handlers/bifrost/model_bifrost_response.py` | `ModelLlmInferenceResponse` |
| `nodes/node_llm_inference_effect/services/service_llm_metrics_publisher.py` | `ModelLlmInferenceResponse` |
| `nodes/node_llm_inference_effect/services/protocol_llm_handler.py` | `ModelLlmInferenceResponse` |
| `nodes/node_llm_embedding_effect/models/model_llm_embedding_response.py` | `ModelLlmUsage` |
| `nodes/node_llm_embedding_effect/handlers/handler_embedding_ollama.py` | `ModelLlmUsage` |
| `nodes/node_llm_embedding_effect/handlers/handler_embedding_openai_compatible.py` | `ModelLlmUsage` |
| `nodes/node_registry_effect/models/__init__.py` | `ModelRegistryRequest`, `ModelRegistryResponse` |
| `nodes/node_registry_effect/handlers/handler_postgres_upsert.py` | `ProtocolPostgresAdapter` (protocol) |
| `nodes/node_registry_effect/handlers/handler_postgres_deactivate.py` | `ProtocolPostgresAdapter` |
| `nodes/node_registry_effect/handlers/handler_partial_retry.py` | `ProtocolPostgresAdapter` |
| `handlers/registration_storage/handler_registration_storage_postgres.py` | `ModelRegistryRequest`, `ModelRegistryResponse` |

**Test files importing from `omnibase_infra.nodes.effects` (38 files):**

Key test paths:
- `tests/unit/registration/effect/` — `test_effect_idempotency_store.py`, `test_effect_partial_failure.py`
- `tests/unit/nodes/node_llm_inference_effect/` — all handler/model tests
- `tests/unit/models/effects/` — all LLM model unit tests (13 files)
- `tests/integration/registration/effect/` — protocol compliance, integration tests
- `tests/performance/registration/effect/` — performance benchmarks

### 4.2 `nodes/handlers/` — Importers

**Zero Python importers.** The `contract.yaml` files are picked up by:

1. `RuntimeContractConfigLoader.load_all_contracts()` via `search_path.glob("**/contract.yaml")`
2. `ContractConfigExtractor.extract_from_paths()` via the same recursive glob
3. `seed-infisical.py` with `--contracts-dir src/omnibase_infra/nodes` (recursive scan)

The `RuntimeContractConfigLoader` does NOT filter by directory naming convention. Any `contract.yaml` found under the search root is loaded.

### 4.3 `nodes/reducers/` — Importers

**Source files importing from `omnibase_infra.nodes.reducers`:**

| Importer | What It Imports |
|---|---|
| `nodes/__init__.py` | `RegistrationReducer` |
| `nodes/node_ledger_projection_compute/handlers/handler_ledger_projection.py` | `ModelPayloadLedgerAppend` |
| `nodes/node_ledger_write_effect/registry/registry_infra_ledger_write.py` | `ModelPayloadLedgerAppend` |
| `nodes/node_ledger_write_effect/protocols/protocol_ledger_persistence.py` | `ModelPayloadLedgerAppend` |
| `nodes/node_ledger_write_effect/handlers/handler_ledger_append.py` | `ModelPayloadLedgerAppend` |
| `nodes/node_registration_orchestrator/handlers/handler_node_registration_acked.py` | `ModelRegistrationAckUpdate`, `ModelRegistrationConfirmation` |
| `nodes/node_registration_orchestrator/services/registration_reducer_service.py` | `RegistrationReducer`, `ModelRegistrationState` |
| `nodes/node_registration_reducer/models/__init__.py` | All registration state models |
| `runtime/intent_effects/intent_effect_postgres_upsert.py` | `ModelPayloadPostgresUpsertRegistration` |
| `runtime/intent_effects/intent_effect_postgres_update.py` | `ModelPayloadPostgresUpdateRegistration` |

**Test files importing from `omnibase_infra.nodes.reducers` (31 files):**

Key test paths:
- `tests/unit/nodes/reducers/` — reducer unit tests, purity tests, performance
- `tests/unit/nodes/reducers/models/` — model tests
- `tests/unit/nodes/node_registration_orchestrator/` — orchestrator tests
- `tests/replay/` — all 9 replay tests (replay_utils.py helper)
- `tests/integration/registration/` — workflow and handler tests
- `tests/integration/ledger/` — ledger append idempotency

---

## 5. Loaders and Globs That Scan `nodes/`

| Loader | Pattern | Picks Up `effects/`? | Picks Up `handlers/`? | Picks Up `reducers/`? |
|---|---|---|---|---|
| `RuntimeContractConfigLoader` | `search_path.glob("**/contract.yaml")` | YES (`effects/contract.yaml`) | YES (5 `contract.yaml` files) | NO (no `contract.yaml`) |
| `ContractConfigExtractor` | `extract_from_paths([nodes_dir])` | YES | YES | NO |
| `seed-infisical.py` | `--contracts-dir src/omnibase_infra/nodes` → recursive glob | YES | YES | NO |
| Python `import` resolution | Module path matching | YES (`nodes.effects`) | NO (no `__init__.py`) | YES (`nodes.reducers`) |

**Implication for `nodes/handlers/`:** Since `handlers/` has no `__init__.py`, it is invisible to Python imports. It exists purely as a YAML contract location. The `RuntimeContractConfigLoader` picks up all 5 `contract.yaml` files there.

**Implication for `nodes/effects/contract.yaml`:** The `effects/contract.yaml` is currently being picked up by `RuntimeContractConfigLoader` and `ContractConfigExtractor` as if it were a node-level contract. This is an accidental inclusion — `effects/contract.yaml` belongs to `NodeRegistryEffect` which has its own proper home at `node_registry_effect/`.

---

## 6. Packaging and Distribution Implications

### 6.1 Python Package Structure

`pyproject.toml` declares:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/omnibase_infra"]
```

This includes the entire `omnibase_infra` package tree recursively. All three non-node dirs are currently included in the wheel:
- `omnibase_infra.nodes.effects` — included as Python package (has `__init__.py`)
- `omnibase_infra.nodes.handlers` — **not a Python package** (no `__init__.py`); YAML files included via `MANIFEST.in` or hatch data include
- `omnibase_infra.nodes.reducers` — included as Python package (has `__init__.py`)

### 6.2 Entry Points

No entry points reference `nodes/effects/`, `nodes/handlers/`, or `nodes/reducers/` directly. The one domain plugin entry point is:

```toml
[project.entry-points."onex.domain_plugins"]
registration = "omnibase_infra.nodes.node_registration_orchestrator.plugin:PluginRegistration"
```

This is unaffected by the migration.

### 6.3 Package Data (YAML files)

The `contract.yaml` files in `nodes/handlers/` are not in a Python package (no `__init__.py`). They are included in distribution only if hatch is configured to include non-Python files. This needs verification before migration.

---

## 7. Test Fixtures That Depend on Current Paths

### 7.1 Direct path-based fixtures

| Test File | Dependency |
|---|---|
| `tests/unit/registration/effect/test_effect_idempotency_store.py` | `omnibase_infra.nodes.effects` module imports |
| `tests/unit/registration/effect/test_effect_partial_failure.py` | `NodeRegistryEffect`, `ProtocolPostgresAdapter` from `nodes.effects` |
| `tests/unit/nodes/effects/models/test_model_llm_inference_response.py` | `omnibase_infra.nodes.effects.models` |
| `tests/unit/models/effects/` (13 files) | `omnibase_infra.nodes.effects.models.*` |
| `tests/unit/nodes/reducers/` (4 files) | `omnibase_infra.nodes.reducers.*` |
| `tests/unit/nodes/reducers/models/` | `omnibase_infra.nodes.reducers.models.*` |
| `tests/replay/replay_utils.py` | `RegistrationReducer` from `omnibase_infra.nodes.reducers` |
| `tests/replay/` (8 test files) | via `replay_utils.py` |

### 7.2 Conftest-based fixtures

- `tests/integration/registration/effect/conftest.py` — creates `NodeRegistryEffect`, `ProtocolPostgresAdapter` from `nodes.effects`
- `tests/integration/registration/workflow/conftest.py` — imports from both `nodes.effects` and `nodes.reducers`
- `tests/performance/registration/effect/conftest.py` — `nodes.effects` imports

---

## 8. Proposed Target Layout

The migration moves content to canonical locations without changing public module names (using re-exports for backward compatibility during transition, or direct path updates for OMN-3989).

### 8.1 `nodes/effects/` → Three canonical destinations

**Shared LLM models** (used across `node_llm_inference_effect`, `node_llm_embedding_effect`):

```
CURRENT:  src/omnibase_infra/nodes/effects/models/model_llm_*.py
TARGET:   src/omnibase_infra/models/llm/model_llm_*.py
```

Rationale: These are domain models, not node-specific. They belong in `omnibase_infra.models.llm` alongside other shared models.

**Registry models** (used by `node_registry_effect`):

```
CURRENT:  src/omnibase_infra/nodes/effects/models/model_registry_*.py
          src/omnibase_infra/nodes/effects/models/model_effect_idempotency_config.py
TARGET:   src/omnibase_infra/nodes/node_registry_effect/models/model_registry_*.py
          src/omnibase_infra/nodes/node_registry_effect/models/model_effect_idempotency_config.py
```

**Protocols and NodeRegistryEffect**:

```
CURRENT:  src/omnibase_infra/nodes/effects/protocol_postgres_adapter.py
          src/omnibase_infra/nodes/effects/protocol_effect_idempotency_store.py
          src/omnibase_infra/nodes/effects/store_effect_idempotency_inmemory.py
          src/omnibase_infra/nodes/effects/registry_effect.py
          src/omnibase_infra/nodes/effects/contract.yaml
TARGET:   src/omnibase_infra/nodes/node_registry_effect/protocols/protocol_postgres_adapter.py
          src/omnibase_infra/nodes/node_registry_effect/protocols/protocol_effect_idempotency_store.py
          src/omnibase_infra/nodes/node_registry_effect/store_effect_idempotency_inmemory.py
          src/omnibase_infra/nodes/node_registry_effect/node.py  (rename registry_effect.py → node.py)
          (contract.yaml already exists at node_registry_effect/ — deduplicate)
```

### 8.2 `nodes/handlers/` → `contracts/handlers/`

```
CURRENT:  src/omnibase_infra/nodes/handlers/db/contract.yaml
          src/omnibase_infra/nodes/handlers/graph/contract.yaml
          src/omnibase_infra/nodes/handlers/http/contract.yaml
          src/omnibase_infra/nodes/handlers/intent/contract.yaml
          src/omnibase_infra/nodes/handlers/mcp/contract.yaml
TARGET:   src/omnibase_infra/contracts/handlers/db/contract.yaml
          src/omnibase_infra/contracts/handlers/graph/contract.yaml
          src/omnibase_infra/contracts/handlers/http/contract.yaml
          src/omnibase_infra/contracts/handlers/intent/contract.yaml
          src/omnibase_infra/contracts/handlers/mcp/contract.yaml
```

Rationale: Handler plugin contracts are not nodes. They belong in a `contracts/` directory. The `RuntimeContractConfigLoader` search path would need to be updated to include `src/omnibase_infra/contracts/`.

Alternative: Keep them in `src/omnibase_infra/handlers/` (which already exists for handler implementations). Each handler already has a corresponding implementation dir under `src/omnibase_infra/handlers/`.

### 8.3 `nodes/reducers/` → Two canonical destinations

**RegistrationReducer** (pure function reducer):

```
CURRENT:  src/omnibase_infra/nodes/reducers/registration_reducer.py
TARGET:   src/omnibase_infra/nodes/node_registration_reducer/registration_reducer.py
```

The declarative `NodeRegistrationReducer` already lives at `node_registration_reducer/`. The pure function `RegistrationReducer` can co-locate there.

**Registration state models**:

```
CURRENT:  src/omnibase_infra/nodes/reducers/models/model_payload_*.py
          src/omnibase_infra/nodes/reducers/models/model_registration_*.py
TARGET:   src/omnibase_infra/nodes/node_registration_reducer/models/model_payload_*.py
          src/omnibase_infra/nodes/node_registration_reducer/models/model_registration_*.py
```

Note: `node_registration_reducer/models/__init__.py` already imports from `nodes.reducers.models` — this would become a local import after migration.

---

## 9. Expected Node Count Before and After

| Category | Before | After |
|---|---|---|
| `node_*` prefixed node dirs | 46 | 46 (unchanged) |
| Non-prefixed node dirs (`architecture_validator`, `contract_registry_reducer`) | 2 | 2 (unchanged) |
| Non-node dirs (`effects/`, `handlers/`, `reducers/`) | 3 | 0 (removed) |
| Total node dirs | 48 | 48 (unchanged) |

The migration eliminates the 3 non-node directories. The canonical node count (48 actual nodes) does not change.

---

## 10. Rollback Path

If OMN-3989 migration destabilizes packaging or discovery:

1. **Python imports break**: The safest rollback is re-adding `nodes/effects/` and `nodes/reducers/` as re-export shims pointing to the new locations. Because both have `__init__.py`, this is backward-compatible.

2. **`RuntimeContractConfigLoader` misses handler contracts**: Rollback by reverting `search_paths` configuration to include `nodes/` again, or by restoring `nodes/handlers/` in parallel with the new `contracts/handlers/` location.

3. **`seed-infisical.py` misses contracts**: Update `--contracts-dir` default path or pass both directories. No data loss — Infisical keys are not deleted by the script.

4. **Wheel distribution**: If YAML files are missing from the wheel, add explicit `include` patterns to `pyproject.toml`.

---

## 11. Explicit Migration Plan for OMN-3989

### 11.1 Migrate `nodes/effects/models/` LLM models → `models/llm/`

1. Create `src/omnibase_infra/models/llm/__init__.py` with re-exports
2. Move all `model_llm_*.py` and `adapter_llm_usage_to_contract.py` files
3. Update all 14 source importers (see Section 4.1)
4. Update 38 test files
5. Add temporary re-export shim in `nodes/effects/models/` pointing to new locations (preserve backward compat for 1 release cycle)

### 11.2 Migrate `nodes/effects/` registry content → `node_registry_effect/`

1. Move `protocol_postgres_adapter.py` → `node_registry_effect/protocols/`
2. Move `protocol_effect_idempotency_store.py` → `node_registry_effect/protocols/`
3. Move `store_effect_idempotency_inmemory.py` → `node_registry_effect/`
4. Rename `registry_effect.py` → `node_registry_effect/node.py` (or keep as `node_registry_effect/registry_effect.py` if preferred)
5. Move `model_registry_*.py` and `model_effect_idempotency_config.py` → `node_registry_effect/models/`
6. Deduplicate `contract.yaml` — `effects/contract.yaml` and `node_registry_effect/contract.yaml` must be reconciled (keep one)
7. Update `node_registry_effect/models/__init__.py` to use local imports
8. Update `handlers/registration_storage/handler_registration_storage_postgres.py`
9. Update `nodes/__init__.py`
10. Update all test files (integration/unit/performance)

### 11.3 Migrate `nodes/handlers/` → `contracts/handlers/`

1. Create `src/omnibase_infra/contracts/handlers/` directory tree
2. Move all 5 `contract.yaml` files
3. Update `RuntimeContractConfigLoader` call sites to include the new path in `search_paths`
4. Update `seed-infisical.py` default `--contracts-dir` value or add second default path
5. Verify `ContractConfigExtractor` scans the new location

### 11.4 Migrate `nodes/reducers/` → `node_registration_reducer/`

1. Move `registration_reducer.py` → `node_registration_reducer/registration_reducer.py`
2. Move all `models/model_*.py` → `node_registration_reducer/models/`
3. Update `node_registration_reducer/models/__init__.py` to use local imports
4. Update all 10 source importers (see Section 4.3)
5. Update 31 test files
6. Add re-export shim in `nodes/reducers/__init__.py` pointing to new location

### 11.5 Remove `nodes/effects/`, `nodes/handlers/`, `nodes/reducers/`

1. After all import updates verified passing: delete the three dirs
2. Run full test suite: `uv run pytest -m "not slow"`
3. Run type checking: `uv run mypy src/ --strict`
4. Run `pre-commit run --all-files`

### 11.6 Update `nodes/__init__.py`

Remove imports from `nodes.effects` and `nodes.reducers`. Replace with direct imports from canonical locations.

---

## 12. Blast Radius Summary

| Risk Area | Severity | Notes |
|---|---|---|
| Python imports broken | HIGH | 69 files import from `nodes.effects` or `nodes.reducers` |
| Runtime contract loading broken | MEDIUM | 5 handler contracts in `nodes/handlers/` move; update search paths |
| Wheel distribution (YAML missing) | LOW | Verify hatch includes YAML data files |
| Type checking errors | MEDIUM | Protocol references and model types may require path updates |
| Test suite failures | HIGH | 69+ test files need import updates |
| seed-infisical.py coverage gap | LOW | Update default `--contracts-dir` or add second path |

**Total files requiring import changes:**
- Source files: ~29 files
- Test files: ~69 files
- Config/scripts: 2 files (`seed-infisical.py`, runtime search path config)
