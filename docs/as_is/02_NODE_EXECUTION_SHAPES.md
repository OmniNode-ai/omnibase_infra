> **Navigation**: [Home](../index.md) > [As-Is](INDEX.md) > Node Execution Shapes

## Node execution shapes (As-Is)

This document captures the **current execution surface area** for node types in Core, and
how Infra3 nodes mirror (or partially mirror) those patterns.

### Core: node taxonomy

Core defines “kinds” and “types”:

- **Kind**: architectural role (effect/compute/reducer/orchestrator/runtime_host)
  - Reference: `omnibase_core/src/omnibase_core/enums/enum_node_kind.py`
- **Type**: specific implementation classification (e.g., `*_GENERIC`)
  - Reference: `omnibase_core/src/omnibase_core/enums/enum_node_type.py`

### Core: minimal node lifecycle shape

`NodeCoreBase` defines the common lifecycle shape:

- **`initialize()`**: loads contract (if present), resolves dependencies from the container, sets state/metrics.
- **`process(input_data)`**: abstract; implemented by each node type.

Reference: `omnibase_core/src/omnibase_core/infrastructure/node_core_base.py`.

### Core: contract-driven “generic” nodes

Core provides “contract-driven” generic nodes with stable interfaces:

#### Effect nodes

- Primary class: `NodeEffect`
  - Reference: `omnibase_core/src/omnibase_core/nodes/node_effect.py`
- **Input/Output models**:
  - `ModelEffectInput`: `omnibase_core/src/omnibase_core/models/effect/model_effect_input.py`
  - `ModelEffectOutput`: `omnibase_core/src/omnibase_core/models/effect/model_effect_output.py`
- Driven by a subcontract:
  - `ModelEffectSubcontract`: `omnibase_core/src/omnibase_core/models/contracts/subcontracts/model_effect_subcontract.py`

**As-is semantics**:
- Effect is the component that performs side effects.
- Contracts can describe operations, retry policies, circuit breakers, etc.
- Core documents explicit “v1.0 limitations” around per-operation configs (kept here as fact, not critique).

#### Reducer nodes

Core’s reducer node is an FSM-driven pattern:
- Reference: `omnibase_core/src/omnibase_core/nodes/node_reducer.py`

**As-is semantics**:
- Reducer maintains mutable FSM state (MVP tradeoff) but is architecturally described as:
  \((state, input)\rightarrow(new\_state, intents/actions)\)
- Side effects are expressed as intents to be executed elsewhere.

#### Orchestrator nodes

Core orchestrator coordinates multi-step workflows:
- Reference: `omnibase_core/src/omnibase_core/nodes/node_orchestrator.py`
- Execution engine in mixin: `MixinWorkflowExecution`
  - Reference: `omnibase_core/src/omnibase_core/mixins/mixin_workflow_execution.py`

**As-is semantics**:
- Orchestrator maintains mutable workflow execution state (MVP tradeoff).
- Workflow definition can be injected manually or loaded from contract metadata mixins.

### Core: service wrappers (production wiring pattern)

Core also provides “service” wrappers that pre-compose mixins:

- `ModelServiceEffect`:
  - Reference: `omnibase_core/src/omnibase_core/models/services/model_service_effect.py`
- `ModelServiceOrchestrator`:
  - Reference: `omnibase_core/src/omnibase_core/models/services/model_service_orchestrator.py`

These represent an **as-is production wiring idiom**: “service mode + health + event bus + metrics”
wrapped around a node role.

### Core: container + contract-driven dependency resolution

Core container:
- `ModelONEXContainer`: `omnibase_core/src/omnibase_core/models/container/model_onex_container.py`

**As-is behavior** (high-level):
- Provides `get_service_async(protocol_type, ...)` with caching.
- Can resolve through a service registry if enabled.

Also note there is a `NodeBase` class that demonstrates contract loading patterns and
contract-driven dependency processing:
- Reference: `omnibase_core/src/omnibase_core/infrastructure/node_base.py`

### Infra3: node implementations used in 2-way registration

Infra3 contains concrete nodes that align to the Effect/Reducer split:

#### Infra3 reducer (pure intent emission)

- `NodeDualRegistrationReducer`
  - Reference: `omnibase_infra3/src/omnibase_infra/nodes/reducers/node_dual_registration_reducer.py`

**As-is semantics**:
- It emits **Core typed intents**:
  - `ModelConsulRegisterIntent`
  - `ModelPostgresUpsertRegistrationIntent`
  - Defined in Core: `omnibase_core/src/omnibase_core/models/intents/`

This is an important “shape alignment” point: Infra3 reducers are already using
Core’s intent system rather than inventing a new intent representation.

#### Infra3 effect (executes I/O and can publish events)

- `node_registry_effect` (contracted “effect node”)
  - Contract: `omnibase_infra3/src/omnibase_infra/nodes/node_registry_effect/v1_0_0/contract.yaml`
  - Implementation: `omnibase_infra3/src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py`

**As-is semantics**:
- Executes Consul + DB operations via resolved handler dependencies.
- Uses an event bus dependency to publish (at least some) registry-related messages.

### Gap to keep in mind (purely descriptive)

Core provides a unified “contract-driven NodeEffect / NodeReducer / NodeOrchestrator”
framework, while Infra3 also contains nodes that implement similar roles but are not
always wired through the same execution runtime. This is a key reason we need the
runtime-dispatch map (`05_RUNTIME_DISPATCH_SHAPES.md`) before evaluating changes.
