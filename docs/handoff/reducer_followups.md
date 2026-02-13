# Reducer-Authoritative Registration Workflow: Follow-Up Tasks

**Context**: The reducer-authoritative refactoring is complete. `RegistrationReducerService` now
owns all state transition decisions, and all 4 handlers delegate to it. Three follow-ups remain to
close the E2E registration loop.

**Baseline**: 11,559 unit tests pass, mypy clean, ruff clean.

---

## Follow-Up 1: Extend `publish_envelope` to Accept a Partition Key

### Problem

`DispatchResultApplier._resolve_partition_key()` (line 123 of
`src/omnibase_infra/runtime/service_dispatch_result_applier.py`) already extracts a partition key
from output events, but it cannot pass the key to `publish_envelope` because the method signature
does not accept one. The key is currently logged at DEBUG level and discarded. Without partition
keys, output events land on random Kafka partitions and per-entity ordering is lost.

The lower-level `publish()` method **does** accept `key: bytes | None` on every implementation
(see `EventBusInmemory.publish` at line 278 and `MixinKafkaBroadcast`/`EventBusKafka` which
delegate to `self.publish`).

### Files to Modify

| # | File | What to change |
|---|------|----------------|
| 1 | `src/omnibase_infra/protocols/protocol_event_bus_like.py` | Add `key: bytes \| None = None` parameter to `publish_envelope` |
| 2 | `src/omnibase_infra/event_bus/event_bus_inmemory.py` | Update `publish_envelope` signature and pass `key` to `self.publish()` |
| 3 | `src/omnibase_infra/event_bus/mixin_kafka_broadcast.py` | Update `publish_envelope` signature and pass `key` to `self.publish()` |
| 4 | `src/omnibase_infra/runtime/service_dispatch_result_applier.py` | Pass the resolved partition key to `publish_envelope` |
| 5 | All test files with mock `publish_envelope` implementations | Update signatures to accept the new parameter |

### Step-by-Step Changes

#### 1. Protocol definition (`protocol_event_bus_like.py`, line 74)

Current signature:

```python
async def publish_envelope(
    self,
    envelope: object,
    topic: str,
) -> None:
```

New signature:

```python
async def publish_envelope(
    self,
    envelope: object,
    topic: str,
    *,
    key: bytes | None = None,
) -> None:
```

The `key` parameter is keyword-only with a default of `None` so all existing callers remain
compatible.

#### 2. In-memory implementation (`event_bus_inmemory.py`, line 387)

Current code (line 469):

```python
await self.publish(topic, None, value, headers)
```

Change the method signature on line 387 to accept `key: bytes | None = None` as a keyword-only
parameter, and change line 469 to:

```python
await self.publish(topic, key, value, headers)
```

#### 3. Kafka mixin (`mixin_kafka_broadcast.py`, line 143)

Same pattern. Current code (line 177):

```python
await self.publish(topic, None, value, headers)
```

Update the method signature on line 143 to accept `key: bytes | None = None` as a keyword-only
parameter, and change line 177 to:

```python
await self.publish(topic, key, value, headers)
```

Also update the `ProtocolKafkaBroadcastHost` protocol class at line 44 if it declares
`publish_envelope` (it does not currently, so no change needed there).

#### 4. DispatchResultApplier (`service_dispatch_result_applier.py`, line 279)

Current code:

```python
await self._event_bus.publish_envelope(
    envelope=output_envelope,
    topic=self._output_topic,
)
```

Change to:

```python
await self._event_bus.publish_envelope(
    envelope=output_envelope,
    topic=self._output_topic,
    key=partition_key,
)
```

The variable `partition_key` is already computed on line 269. Remove the TODO comment on
line 149 and the "currently logged only" note in the class docstring (lines 75-77).

#### 5. Test mock implementations

The following test files define mock `publish_envelope` methods that must be updated to accept the
new `key` keyword argument. Since `key` has a default value, only update the signature -- no
behavioral changes are needed:

- `tests/integration/registration/test_introspection_event_bus_e2e.py` (line 69)
- `tests/integration/mixins/test_mixin_node_introspection_contract_integration.py` (line 88)
- `tests/integration/timeouts/conftest.py` (line 96)
- `tests/unit/runtime/test_runtime_host_process.py` (line 224)
- `tests/unit/runtime/test_registry_race_conditions.py` (line 101)
- `tests/unit/runtime/test_registry_runtime_type_validation.py` (line 149)
- `tests/unit/mixins/test_mixin_node_introspection.py` (line 133 and line 941)
- `tests/unit/errors/test_registry_errors_correlation_id.py` (line 129)
- `tests/unit/errors/test_event_bus_registry_error.py` (lines 194 and 316)

For each, change `async def publish_envelope(self, envelope: object, topic: str) -> None:` to
`async def publish_envelope(self, envelope: object, topic: str, *, key: bytes | None = None) -> None:`.

### Validation

```bash
# Type check
poetry run mypy src/omnibase_infra/

# Lint
poetry run ruff check src/ tests/

# Unit tests (verify no regressions)
poetry run pytest tests/unit/ -n auto

# Specifically test dispatch result applier
poetry run pytest tests/unit/runtime/test_service_dispatch_result_applier.py -xvs

# Integration tests for event bus
poetry run pytest tests/integration/ -k "event_bus or introspection" -n auto
```

---

## Follow-Up 2: Create `IntentEffectPostgresUpdate` Effect Handler

### Problem

The reducer now emits `postgres.update_registration` intents via `ModelPayloadPostgresUpdateRegistration`
(from `RegistrationReducerService.decide_ack` and `decide_heartbeat`), but no effect handler exists to
execute them. The `IntentExecutor` will raise `RuntimeHostError` with
`"No effect handler registered for intent_type='postgres.update_registration'"`.

### Template to Follow

Use `IntentEffectPostgresUpsert` (`src/omnibase_infra/runtime/intent_effects/intent_effect_postgres_upsert.py`)
as the structural template. The new effect is simpler because it performs a plain UPDATE instead of an
UPSERT with record normalization.

### Files to Create

| # | File | Description |
|---|------|-------------|
| 1 | `src/omnibase_infra/runtime/intent_effects/intent_effect_postgres_update.py` | New effect handler |
| 2 | `tests/unit/runtime/test_intent_effect_postgres_update.py` | Unit tests |

### Files to Modify

| # | File | What to change |
|---|------|----------------|
| 1 | `src/omnibase_infra/runtime/intent_effects/__init__.py` | Export the new class |

### Implementation: `intent_effect_postgres_update.py`

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent effect adapter for PostgreSQL registration UPDATE operations.

This module provides the IntentEffectPostgresUpdate adapter, which bridges
ModelPayloadPostgresUpdateRegistration intent payloads to actual PostgreSQL
UPDATE operations via raw SQL on the ProjectorShell's connection pool.

Architecture:
    RegistrationReducerService
        -> ModelPayloadPostgresUpdateRegistration (intent payload)
        -> IntentExecutor
        -> IntentEffectPostgresUpdate.execute()
        -> asyncpg UPDATE query (with monotonic heartbeat guard)

The adapter performs a conditional UPDATE WHERE with a monotonic guard
on last_heartbeat_at to ensure idempotent heartbeat processing:

    UPDATE registration_projections
    SET col1 = $1, col2 = $2, ...
    WHERE entity_id = $N AND domain = $M
      AND (last_heartbeat_at IS NULL OR last_heartbeat_at < $heartbeat_param)

The heartbeat guard is only applied when the updates dict contains a
`last_heartbeat_at` key. For non-heartbeat updates (e.g., ACK state
transitions), the guard is omitted.

Related:
    - ModelPayloadPostgresUpdateRegistration: Intent payload model
    - RegistrationReducerService: Emits this intent from decide_ack / decide_heartbeat
    - IntentEffectPostgresUpsert: Sibling effect for INSERT...ON CONFLICT

.. versionadded:: 0.8.0
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ContainerWiringError, RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.projectors.util_sql_identifiers import quote_identifier
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


# TIMESTAMPTZ columns that need str -> datetime conversion for asyncpg.
# Must stay in sync with schema_registration_projection.sql.
_TIMESTAMP_COLUMNS: frozenset[str] = frozenset(
    {
        "ack_deadline",
        "liveness_deadline",
        "last_heartbeat_at",
        "ack_timeout_emitted_at",
        "liveness_timeout_emitted_at",
        "registered_at",
        "updated_at",
    }
)

# UUID columns that need str -> UUID conversion for asyncpg.
_UUID_COLUMNS: frozenset[str] = frozenset(
    {"entity_id", "last_applied_event_id", "correlation_id"}
)


class IntentEffectPostgresUpdate:
    """Intent effect adapter for PostgreSQL registration UPDATE operations.

    Bridges ModelPayloadPostgresUpdateRegistration intent payloads to
    plain UPDATE queries on the registration_projections table. Includes
    a monotonic guard on last_heartbeat_at for idempotent heartbeat
    processing.

    Thread Safety:
        This class is designed for single-threaded async use. The underlying
        asyncpg pool handles connection concurrency.

    Attributes:
        _pool: asyncpg connection pool for executing UPDATE queries.

    .. versionadded:: 0.8.0
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize the PostgreSQL UPDATE intent effect.

        Args:
            pool: asyncpg connection pool. Must be fully initialized.

        Raises:
            ContainerWiringError: If pool is None.
        """
        if pool is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=uuid4(),
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_init",
            )
            raise ContainerWiringError(
                "asyncpg pool is required for IntentEffectPostgresUpdate",
                context=context,
            )
        self._pool = pool

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute a PostgreSQL UPDATE from an intent payload.

        Builds and executes:
            UPDATE registration_projections
            SET <updates>
            WHERE entity_id = $e AND domain = $d
              [AND (last_heartbeat_at IS NULL OR last_heartbeat_at < $hb)]

        The heartbeat monotonic guard is applied only when the updates
        dict contains a ``last_heartbeat_at`` key.

        Args:
            payload: The ModelPayloadPostgresUpdateRegistration intent payload.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            RuntimeHostError: If the payload type is wrong, updates are empty,
                or the UPDATE query fails.
        """
        effective_correlation_id = (
            correlation_id or getattr(payload, "correlation_id", None) or uuid4()
        )

        if not isinstance(payload, ModelPayloadPostgresUpdateRegistration):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_update",
            )
            raise RuntimeHostError(
                f"Expected ModelPayloadPostgresUpdateRegistration, "
                f"got {type(payload).__name__}",
                context=context,
            )

        updates = payload.updates
        if not updates:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_update",
            )
            raise RuntimeHostError(
                "Intent payload has empty updates dict -- UPDATE would be a no-op",
                context=context,
            )

        try:
            # Normalize types for asyncpg
            normalized_updates = self._normalize_for_asyncpg(dict(updates))
            entity_id = payload.entity_id
            domain = payload.domain

            # Build SET clause
            set_parts: list[str] = []
            params: list[object] = []
            idx = 1
            for col, val in normalized_updates.items():
                set_parts.append(f"{quote_identifier(col)} = ${idx}")
                params.append(val)
                idx += 1

            set_clause = ", ".join(set_parts)

            # WHERE clause: entity_id + domain
            where_parts = [
                f"{quote_identifier('entity_id')} = ${idx}",
            ]
            params.append(entity_id)
            idx += 1

            where_parts.append(f"{quote_identifier('domain')} = ${idx}")
            params.append(domain)
            idx += 1

            # Monotonic guard for heartbeat idempotency
            heartbeat_val = normalized_updates.get("last_heartbeat_at")
            if heartbeat_val is not None:
                where_parts.append(
                    f"({quote_identifier('last_heartbeat_at')} IS NULL "
                    f"OR {quote_identifier('last_heartbeat_at')} < ${idx})"
                )
                params.append(heartbeat_val)
                idx += 1

            where_clause = " AND ".join(where_parts)

            # S608: Safe -- identifiers quoted, values parameterized
            sql = (
                f"UPDATE {quote_identifier('registration_projections')} "  # noqa: S608
                f"SET {set_clause} "
                f"WHERE {where_clause}"
            )

            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, *params, timeout=30.0)

            # Parse row count (e.g., "UPDATE 1" -> 1)
            rows_affected = 0
            parts = result.split()
            if parts and parts[-1].isdigit():
                rows_affected = int(parts[-1])

            logger.info(
                "PostgreSQL UPDATE executed: entity_id=%s rows=%d correlation_id=%s",
                str(entity_id),
                rows_affected,
                str(effective_correlation_id),
            )

        except RuntimeHostError:
            raise
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_update",
            )
            logger.warning(
                "PostgreSQL UPDATE intent failed: error=%s correlation_id=%s",
                sanitize_error_message(e),
                str(effective_correlation_id),
                extra={"error_type": type(e).__name__},
            )
            raise RuntimeHostError(
                "Failed to execute PostgreSQL UPDATE intent",
                context=context,
            ) from e

    @staticmethod
    def _normalize_for_asyncpg(
        record: dict[str, object],
    ) -> dict[str, object]:
        """Normalize values from JSON-serializable types to asyncpg-native types.

        Converts string UUIDs to UUID objects and ISO datetime strings to
        datetime objects, matching the schema column types.

        Args:
            record: Dict of column name -> value.

        Returns:
            New dict with UUID and datetime columns converted to native types.
        """
        normalized: dict[str, object] = {}
        for key, value in record.items():
            if value is None:
                normalized[key] = value
            elif key in _UUID_COLUMNS:
                normalized[key] = (
                    UUID(str(value)) if not isinstance(value, UUID) else value
                )
            elif key in _TIMESTAMP_COLUMNS:
                if isinstance(value, str):
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    normalized[key] = dt
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        return normalized


__all__: list[str] = ["IntentEffectPostgresUpdate"]
```

### Key design decisions

1. **Takes `pool` directly, not `ProjectorShell`**: Unlike the UPSERT effect which delegates to
   `ProjectorShell.upsert_partial()`, the UPDATE effect runs raw SQL because `ProjectorShell`'s
   `partial_update()` does not support composite primary keys (it raises
   `ProtocolConfigurationError` for composite PKs at `mixin_projector_sql_operations.py` line 697).
   The registration schema has composite PK `(entity_id, domain)`, so we need direct SQL with a
   two-column WHERE clause.

2. **Monotonic heartbeat guard**: When the `updates` dict contains `last_heartbeat_at`, an
   additional WHERE condition prevents stale heartbeats from overwriting newer ones:
   `WHERE ... AND (last_heartbeat_at IS NULL OR last_heartbeat_at < $param)`. This makes
   heartbeat processing idempotent under redelivery.

3. **No guard for non-heartbeat updates**: When `RegistrationReducerService.decide_ack` emits
   an update intent to transition state to ACTIVE, there is no `last_heartbeat_at` in the
   updates dict, so the monotonic guard is not applied. State transitions are protected by
   the reducer's own state validation.

### Modify `__init__.py` (`src/omnibase_infra/runtime/intent_effects/__init__.py`)

Add the import and export:

```python
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_update import (
    IntentEffectPostgresUpdate,
)

__all__: list[str] = [
    "IntentEffectConsulDeregister",
    "IntentEffectConsulRegister",
    "IntentEffectPostgresUpdate",
    "IntentEffectPostgresUpsert",
]
```

### Unit Test Structure (`test_intent_effect_postgres_update.py`)

Follow the same structure as `tests/unit/runtime/test_intent_effect_postgres_upsert.py`.
Key test cases:

```python
class TestIntentEffectPostgresUpdateInit:
    def test_init_with_valid_pool(self) -> None: ...
    def test_init_raises_on_none_pool(self) -> None: ...

class TestIntentEffectPostgresUpdateExecute:
    async def test_execute_heartbeat_update(self) -> None:
        """Verify SET clause, WHERE with monotonic guard, and params."""
        ...

    async def test_execute_state_transition_update(self) -> None:
        """Verify no monotonic guard for non-heartbeat updates."""
        ...

    async def test_execute_rejects_wrong_payload_type(self) -> None: ...
    async def test_execute_rejects_empty_updates(self) -> None: ...
    async def test_execute_wraps_db_error(self) -> None: ...
    async def test_normalize_timestamps(self) -> None: ...
    async def test_normalize_uuids(self) -> None: ...
    async def test_execute_uses_payload_correlation_id_as_fallback(self) -> None: ...
```

Mock the asyncpg pool using `MagicMock` with `conn.execute = AsyncMock(return_value="UPDATE 1")`.
Verify the SQL string and parameters passed to `conn.execute` using `call_args`.

### Validation

```bash
# Run the new tests
poetry run pytest tests/unit/runtime/test_intent_effect_postgres_update.py -xvs

# Full unit suite
poetry run pytest tests/unit/ -n auto

# Type check
poetry run mypy src/omnibase_infra/

# Lint
poetry run ruff check src/ tests/
```

---

## Follow-Up 3: Wire `postgres.update_registration` in Plugin and Contract

### Problem

The `IntentExecutor` routes intents based on the contract's `intent_routing_table`, and effect
handlers are registered in `PluginRegistration._wire_intent_effects()`. The new
`postgres.update_registration` intent type must be added to both so the executor can find the
`IntentEffectPostgresUpdate` handler at runtime.

### Files to Modify

| # | File | What to change |
|---|------|----------------|
| 1 | `src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml` | Add to `intent_routing_table` and `subscribed_intents` |
| 2 | `src/omnibase_infra/nodes/node_registration_orchestrator/plugin.py` | Add wiring in `_wire_intent_effects()` |

### Step-by-Step Changes

#### 1. Contract YAML (`contract.yaml`, lines 426-434)

Current `intent_consumption` section:

```yaml
intent_consumption:
  subscribed_intents:
    - "consul.register"
    - "consul.deregister"
    - "postgres.upsert_registration"
  intent_routing_table:
    "consul.register": "node_registry_effect"
    "consul.deregister": "node_registry_effect"
    "postgres.upsert_registration": "node_registry_effect"
```

Change to:

```yaml
intent_consumption:
  subscribed_intents:
    - "consul.register"
    - "consul.deregister"
    - "postgres.upsert_registration"
    - "postgres.update_registration"
  intent_routing_table:
    "consul.register": "node_registry_effect"
    "consul.deregister": "node_registry_effect"
    "postgres.upsert_registration": "node_registry_effect"
    "postgres.update_registration": "node_registry_effect"
```

#### 2. Plugin wiring (`plugin.py`, method `_wire_intent_effects`, starting line 1095)

The current wiring logic in `_wire_intent_effects` (line 1164) uses a protocol-prefix dispatch:

```python
if protocol == "postgres" and self._projector is not None:
    from omnibase_infra.runtime.intent_effects import (
        IntentEffectPostgresUpsert,
    )

    pg_effect = IntentEffectPostgresUpsert(projector=self._projector)
    intent_executor.register_handler(intent_type, pg_effect)
```

This code matches **all** `postgres.*` intent types and registers them with `IntentEffectPostgresUpsert`.
This is wrong for `postgres.update_registration` which needs `IntentEffectPostgresUpdate`.

**Change the protocol-prefix dispatch** to distinguish between the two postgres intent types.
Replace the `if protocol == "postgres"` block (lines 1164-1180) with:

```python
if intent_type == "postgres.upsert_registration" and self._projector is not None:
    from omnibase_infra.runtime.intent_effects import (
        IntentEffectPostgresUpsert,
    )

    pg_upsert_effect = IntentEffectPostgresUpsert(projector=self._projector)
    intent_executor.register_handler(intent_type, pg_upsert_effect)
    await self._register_effect_in_container(
        config, IntentEffectPostgresUpsert, pg_upsert_effect, correlation_id
    )
    registered_count += 1
    logger.debug(
        "Registered IntentEffectPostgresUpsert for intent_type=%s "
        "(correlation_id=%s)",
        intent_type,
        correlation_id,
    )

elif intent_type == "postgres.update_registration" and self._pool is not None:
    from omnibase_infra.runtime.intent_effects import (
        IntentEffectPostgresUpdate,
    )

    pg_update_effect = IntentEffectPostgresUpdate(pool=self._pool)
    intent_executor.register_handler(intent_type, pg_update_effect)
    await self._register_effect_in_container(
        config, IntentEffectPostgresUpdate, pg_update_effect, correlation_id
    )
    registered_count += 1
    logger.debug(
        "Registered IntentEffectPostgresUpdate for intent_type=%s "
        "(correlation_id=%s)",
        intent_type,
        correlation_id,
    )
```

**Also update the `_protocol_resources` check** (lines 1138-1146) to account for the pool-based
resource availability. Currently:

```python
_protocol_resources = {
    "postgres": self._projector is not None,
    "consul": self._consul_handler is not None,
}
```

The UPDATE effect needs a pool, not a projector. Change to check availability per intent type:

```python
_protocol_resources = {
    "postgres.upsert_registration": self._projector is not None,
    "postgres.update_registration": self._pool is not None,
    "consul.register": self._consul_handler is not None,
    "consul.deregister": self._consul_handler is not None,
}
```

And update the wirable_intent_types computation (lines 1142-1146) to use exact keys instead
of protocol prefix matching:

```python
wirable_intent_types: set[str] = set()
for it in routing_table:
    if _protocol_resources.get(it, False):
        wirable_intent_types.add(it)
```

This replaces the prefix-based `protocol = it.split(".", 1)[0]` logic with exact key lookup.
The change is backward-compatible because the `_protocol_resources` dict now contains all
known intent types.

### Validation

```bash
# Test the intent routing loader
poetry run pytest tests/unit/runtime/test_intent_routing_loader.py -xvs

# Test the full registration orchestrator integration
poetry run pytest tests/integration/nodes/test_registration_orchestrator_integration.py -xvs

# Test the dispatch result applier (exercises IntentExecutor)
poetry run pytest tests/unit/runtime/test_service_dispatch_result_applier.py -xvs

# Full unit suite
poetry run pytest tests/unit/ -n auto

# Full integration suite
poetry run pytest tests/integration/ -n auto

# Type check
poetry run mypy src/omnibase_infra/

# Lint
poetry run ruff check src/ tests/
```

### Integration Smoke Test

After all three follow-ups are implemented, run the E2E registration loop:

```bash
# E2E registration tests (requires PostgreSQL)
poetry run pytest tests/integration/registration/ -xvs -m "not slow"

# Specifically test heartbeat -> UPDATE path
poetry run pytest tests/integration/registration/handlers/test_handler_node_heartbeat_integration.py -xvs
```

---

## Dependency Order

Follow-ups can be implemented independently, but the recommended order is:

1. **Follow-Up 2** (create `IntentEffectPostgresUpdate`) -- no dependencies, pure addition
2. **Follow-Up 3** (wire in plugin + contract) -- depends on Follow-Up 2 existing
3. **Follow-Up 1** (partition key in `publish_envelope`) -- fully independent, can be done in
   parallel with 2+3

---

## Key Reference Files

| Purpose | Path |
|---------|------|
| Protocol for `publish_envelope` | `src/omnibase_infra/protocols/protocol_event_bus_like.py` |
| In-memory event bus | `src/omnibase_infra/event_bus/event_bus_inmemory.py` |
| Kafka broadcast mixin | `src/omnibase_infra/event_bus/mixin_kafka_broadcast.py` |
| Kafka event bus class | `src/omnibase_infra/event_bus/event_bus_kafka.py` |
| DispatchResultApplier | `src/omnibase_infra/runtime/service_dispatch_result_applier.py` |
| IntentExecutor + ProtocolIntentEffect | `src/omnibase_infra/runtime/service_intent_executor.py` |
| ProtocolIntentPayload | `src/omnibase_infra/runtime/protocols/protocol_intent_payload.py` |
| IntentEffectPostgresUpsert (template) | `src/omnibase_infra/runtime/intent_effects/intent_effect_postgres_upsert.py` |
| IntentEffectConsulRegister (template) | `src/omnibase_infra/runtime/intent_effects/intent_effect_consul_register.py` |
| Intent effects `__init__` | `src/omnibase_infra/runtime/intent_effects/__init__.py` |
| ModelPayloadPostgresUpdateRegistration | `src/omnibase_infra/nodes/reducers/models/model_payload_postgres_update_registration.py` |
| Registration orchestrator contract | `src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml` |
| Plugin (intent wiring) | `src/omnibase_infra/nodes/node_registration_orchestrator/plugin.py` |
| Intent routing loader | `src/omnibase_infra/runtime/service_intent_routing_loader.py` |
| ProjectorShell | `src/omnibase_infra/runtime/projector_shell.py` |
| SQL operations mixin | `src/omnibase_infra/runtime/mixins/mixin_projector_sql_operations.py` |
| SQL schema | `src/omnibase_infra/schemas/schema_registration_projection.sql` |
| RegistrationReducerService | `src/omnibase_infra/nodes/node_registration_orchestrator/services/registration_reducer_service.py` |
| Upsert effect tests (template) | `tests/unit/runtime/test_intent_effect_postgres_upsert.py` |
| Reducer models `__init__` | `src/omnibase_infra/nodes/reducers/models/__init__.py` |
| `quote_identifier` utility | `src/omnibase_infra/models/projectors/util_sql_identifiers.py` |
