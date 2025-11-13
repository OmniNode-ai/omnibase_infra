# OmniNode Bridge → OmniBase Infra Migration Guide

## 1. Purpose
- Transform `omninode_bridge` bridge workflows into canonical Omninode nodes hosted in `omnibase_infra`.
- Preserve operational parity for coordination, registry, metadata stamping, and workflow orchestration while eliminating ad-hoc bridge adapters.
- Deliver automation and validation steps suitable for Claude Code Web execution without requiring direct GitHub access.

## 2. Source Staging (No PAT Required)
- **Prepare working copy**  
  ```bash
  export SOURCE_ROOT="/Volumes/PRO-G40/Code/omninode/repos"
  export MIGRATION_ROOT="/tmp/omninode_bridge_migration"
  mkdir -p "${MIGRATION_ROOT}"
  rsync -a --delete "${SOURCE_ROOT}/omninode_bridge/" "${MIGRATION_ROOT}/omninode_bridge/"
  ```
- **Lock revision**: record `git -C "${SOURCE_ROOT}/omninode_bridge" rev-parse HEAD` inside `MIGRATION_ROOT/reference.txt` for provenance.
- **Do not add as submodule**: all commands operate on the copied tree to avoid PAT exposure.

## 3. Target Repository Preparation (`omnibase_infra`)
- Pull latest `main` in `/Volumes/PRO-G40/Code/omninode/repos/omnibase_infra` (create repo if absent).
- Ensure `poetry install` succeeds and pytest baseline passes before migration.
- Confirm availability of canonical templates:
  - `docs/guides/templates/ORCHESTRATOR_NODE_TEMPLATE.md`
  - `docs/guides/templates/REDUCER_NODE_TEMPLATE.md`
  - `docs/guides/templates/COMPUTE_NODE_TEMPLATE.md`
  - `docs/guides/templates/EFFECT_NODE_TEMPLATE.md`
- Establish new documentation folder `docs/migrations/`.
- Review prep branch `claude/infrastructure-foundation-prep-011CUqGGKpP3SVMdyMaRJWM1`: it lands canonical Postgres/Kafka/resilience/observability utilities with a large test suite—reuse those patterns and avoid duplicating infrastructure code during migration.

## 4. Architecture Mapping

| Legacy Scope | New Node Stack | Responsibilities | Canonical Reference |
|--------------|----------------|------------------|---------------------|
| `agents/workflows/CodeGenerationWorkflow` (async coordinator) | `nodes/code_generation_orchestrator` (orchestrator) | Stage transitions, template selection, metrics aggregation | `ONEX_FOUR_NODE_ARCHITECTURE.md`, orchestrator template |
| `agents/workflows` sub-steps (template load, validation) | `nodes/code_generation_reducer` | State, retries, FSM transitions | reducer template |
| `services/metadata_stamping` + `clients/metadata_stamping_client` | `nodes/metadata_stamping_effect` | External metadata stamping service calls, event emission | effect template |
| `agents/metrics`, `monitoring/codegen_dlq_monitor.py` | `nodes/code_generation_compute` | Aggregations, SLA enforcement, metrics summarization | compute template |
| `registry/` and `config/environment_config.py` | `nodes/registry_reducer` + shared `tools/` | Registry synchronization, config validation | `CONTAINER_TYPES.md`, contract templates |

- Align contract schemas with omnibase_core linked-document architecture (contract.yaml + contracts/ + models/).
- Event bus topics from `EVENT_PUBLISHING_QUICK_REFERENCE.md` should migrate into orchestrator node contracts as shared enums.

## 5. Migration Workflow
1. **Generate node scaffolds**
   - Use omnibase_core scripting entry point:
     ```bash
     cd /Volumes/PRO-G40/Code/omninode/repos/omnibase_infra
     poetry run python -m omnibase_core.scripts.generate_node --node-type orchestrator \
       --node-name code_generation_orchestrator --domain bridge --output src/omnibase_infra/nodes
     ```
   - Repeat for reducer, compute, and effect nodes; maintain versioned directories (`v1_0_0/`).
2. **Port contracts**
   - From `MIGRATION_ROOT/omninode_bridge/src/omninode_bridge/contracts`, extract YAML definitions.
   - Rewrite into canonical layout:
     - `contract.yaml` referencing `contracts/contract_actions.yaml`, `contract_models.yaml`, etc.
     - Map existing pydantic models to generated equivalents; remove manual models once generation completes.
3. **Transplant implementation logic**
   - Map workflows:
     - `CodeGenerationWorkflow.initialize` → orchestrator `setup` path.
     - Template loading/pipelines → compute node methods.
     - External resource calls (Kafka, Redis, Consul) → effect node tool classes with dependency injection.
   - Shared utilities move into `src/omnibase_infra/tools/` prefixed with `tool_` and registered via enums.
4. **Reducer state & orchestration**
   - Translate `ThreadSafeState`, `SignalCoordinator`, and `StateManager` patterns into reducer node state models.
   - Use canonical FSM pattern from `docs/guides/templates/REDUCER_NODE_TEMPLATE.md` for transitions.
5. **Environment & configuration**
   - Convert `config/*.py` to YAML-backed config consumed via `node_config.yaml`.
   - Document service endpoints in `deployment_config.yaml`.
   - Secrets remain external; reference `.env` keys only.
6. **Event handling**
   - Move topic names to `enums/enum_code_generation_topic.py`.
   - Implement event publication through effect node using `omnibase_core.events`.
7. **Delete deprecated bridge adapters**
   - After parity validation, create follow-up tickets to remove redundant `metadata_stamping` clients from original repo.

## 6. Testing Strategy
- **Unit tests**: migrate `tests/unit/**` to `src/omnibase_infra/nodes/.../node_tests/`.
  - Update imports to new module paths.
  - Enforce canonical naming: `test_code_generation_orchestrator.py`.
- **Integration tests**:
  - Port `tests/integration/` workflows; run via `pytest -m "integration and code_generation"`.
  - Provide fixtures for Kafka/Redpanda via `tests/fixtures/event_bus.py` referencing local docker-compose.
- **New tests**:
  - Add contract validation tests using `omnibase_core.validation.contract_validator`.
  - Include reducer FSM regression covering success, retry, DLQ scenarios.
- **CI hook**: Update `pyproject.toml` and `.github/workflows/ci.yml` (if present) to execute `pytest --maxfail=1 --disable-warnings -q`.

## 7. Automation Scripts
- **Node generation helper** (`scripts/generate_bridge_nodes.py`):
  - Wrap `poetry run python -m omnibase_core.scripts.generate_node` for orchestrator, reducer, compute, effect nodes with consistent naming.
- **Contract sync script** (`scripts/sync_bridge_contracts.py`):
  - Reads legacy contract definitions, outputs canonical YAML files, and reruns model generation (`poetry run omnibase_core tools regenerate-models --path ...`).
- **Test harness** (`scripts/run_bridge_migration_tests.sh`):
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  poetry run pytest src/omnibase_infra/nodes/code_generation_orchestrator/v1_0_0/node_tests
  poetry run pytest src/omnibase_infra/nodes/code_generation_reducer/v1_0_0/node_tests
  poetry run pytest tests/integration/code_generation
  ```
- Document script usage inside `README.md`.

## 8. Validation Checklist
- [ ] Contracts linted via `poetry run omnibase_core tools validate-contract --path <node>/v1_0_0/contract.yaml`.
- [ ] Generated models committed; no manual models remain.
- [ ] Event schemas verified against `EVENT_PUBLISHING_QUICK_REFERENCE.md`.
- [ ] All pytest suites pass with coverage ≥ baseline (record baseline from legacy tests).
- [ ] Deployment config references new node entrypoints.

## 9. Risks & Mitigations
- **Template regression**: ensure template cache logic preserved; add integration tests with cold and warm cache scenarios.
- **Concurrency differences**: reducer FSM must cover previously implicit thread-safe behavior; document lease/lock strategy.
- **Metrics continuity**: replicate Prometheus metrics names; verify via `tests/integration/metrics/test_metrics_emission.py`.
- **Rollback path**: retain original bridge repository snapshot until production bake-off completes.

## 10. Follow-Up Documentation
- Update `docs/architecture/` in `omnibase_infra` with migration rationale.
- Cross-link to `docs/FUTURE_FUNCTIONALITY_PHASE_5_6.md` for orchestrated workflows.
- Archive this guide in repo root once migration completes and reference final node documentation.

