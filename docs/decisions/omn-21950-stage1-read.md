# OMN-21950 AC3: stage-1 read of omnibase_core validation

Read at omnibase_core 5bb91ae7, the pin in this repo's `uv.lock`. The question: does an
existing core validation script, or the two stage-1 checks, fail a config model or handler
constructor that names a concrete bus type?

| File(s) read | What it judges | Fails a concrete-bus-typed config model or handler constructor? |
| -- | -- | -- |
| `src/omnibase_core/validation/validator_event_chain_walker.py`, `checker_event_chain_walker.py` (OMN-21558) | Contract YAML relations: unconsumed or unproduced topic, FSM state with no exit, undeclared trigger | No. It reads contract documents only |
| `src/omnibase_core/validation/checker_plan_invariant.py`, `validator_plan_invariant.py` (OMN-21559) | Plan text: a task with no falsifiable criterion, literal values, an invariant with no enforcing task | No. It reads plan text only |
| `scripts/validation/validate-no-direct-io.py` | Imports of `kafka`/`aiokafka`/`confluent_kafka` and other I/O in core's declarative nodes | No. It checks module imports, not annotations, and does not scan infra |
| `scripts/validation/check_no_fallbacks.py` (pattern 6, `_INJECTABLE_PARAM_NAMES`) | `event_bus`/`container` parameters defaulting to `None` in a handler `__init__` | No. It checks the default, not the type |
| `scripts/validation/validate-no-kafka-listener-apis.py`, `validate-no-infra-imports.py`, `validate-hardcoded-env-vars.py`, `validate_no_none_guard_publish.py` | Listener APIs, infra imports, `KAFKA_*` env names, None-guarded publish | No |

Result: none of these fails a config model or a handler constructor typed to a concrete bus
config such as `ModelKafkaEventBusConfig`. The contract-aware check is still needed in
omnibase_core, with this forwarder as its positive control (a later pull request).
