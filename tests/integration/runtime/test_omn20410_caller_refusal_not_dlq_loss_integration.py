# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20410 -- a keyless tenant's refusals do not trip ``projection_dlq_saturation``.

The unit test beside the change pins the counter row. This one asserts the
user-visible fact: it drives records through the real consume boundary
(``_make_event_bus_callback``), the real ``MessageDispatchEngine`` and the real
consumer-flow counters, drains the flow window, and feeds the retained windows
to the real saturation computation the runtime health monitor publishes
(``evaluate_projection_liveness`` -> ``dlq_saturation_status``).

With N answered caller refusals as the only traffic the dimension is HEALTHY;
with N ``ValueError`` failures (positive control) it is DEGRADED and names the
projection. The only stand-ins are the handler that raises -- mirroring
omnimarket's ``CustomerKeyRefusedError`` by name and ``error_code`` -- and a
recording bus with the two duck-typed methods the boundary calls. No Kafka or
Postgres is needed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
    _make_dispatch_callback,
    _make_event_bus_callback,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.health.projection_liveness import (
    DLQ_SATURATION_MIN_MESSAGES,
    describe_dlq_saturation,
    dlq_saturation_status,
    evaluate_projection_liveness,
    select_projection_contracts,
    select_projection_group_suffixes,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.observability import (
    get_consumer_flow_counters,
    reset_consumer_flow_counters,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

_IN_TOPIC = "onex.evt.platform.node-heartbeat.v1"  # onex-topic-allow: test fixture topic shared with the OMN-16777 seam test
_OUT_TOPIC = "onex.cmd.omnibase-infra.gateway-link-health-upsert.v1"  # onex-topic-allow: test fixture topic shared with the OMN-16777 seam test
_PROJECTION = "delegation-routing-reducer"
_GROUP = f"onex-dev.omnimarket.{_PROJECTION}.consume.1.0.0"
_KEY_ABSENT = "ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT"
_REFUSAL_MESSAGE = (
    f"[{_KEY_ABSENT}] delegation.customer_provider_key.absent: delegation "
    "refused for tenant 'omn20410-probe' (task_type='document', surface=cloud): "
    "no provider key is registered for this tenant. Register a provider key "
    "for this tenant and retry."
)
# Above the dimension's minimum-sample floor, so a positive control can trip it.
_N = DLQ_SATURATION_MIN_MESSAGES + 2


class CustomerKeyRefusedError(Exception):
    """Mirror of omnimarket's class: same NAME and ``error_code`` (layering)."""

    def __init__(self, message: str, code: str = _KEY_ABSENT) -> None:
        super().__init__(message)
        self.error_code = code


class _ModelIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    node_id: UUID


class _HandlerRaises:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def handle(self, request: _ModelIn) -> None:
        raise self._exc


class _RecordingBus:
    """The two duck-typed bus methods the consume boundary calls."""

    def __init__(self) -> None:
        self.dlq_records: list[tuple[object, ...]] = []

    async def publish_envelope(self, *args: object, **kwargs: object) -> None:
        return None

    async def _publish_raw_to_dlq(self, *args: object, **kwargs: object) -> bool:
        self.dlq_records.append(args)
        return True


@pytest.fixture(autouse=True)
def _clean_counters(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "true")
    reset_consumer_flow_counters()
    yield
    reset_consumer_flow_counters()


def _engine_for(handler: _HandlerRaises) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id="omn20410-dispatcher",
        dispatcher=_make_dispatch_callback(handler, None),
        category=EnumMessageCategory.EVENT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="omn20410-route",
            topic_pattern="*.evt.platform.node-heartbeat.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="omn20410-dispatcher",
        )
    )
    engine.freeze()
    return engine


def _projection_manifest() -> ModelAutoWiringManifest:
    contract = ModelDiscoveredContract(
        name=_PROJECTION,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=_PROJECTION,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(_IN_TOPIC,), publish_topics=()),
        db_io=ModelDbOwnershipSubcontract(
            db_tables=[
                ModelDbTableDeclaration(
                    name="projection_target",
                    database_ref="application",
                    schema="omninode_internal",
                    migration="0001_init.sql",
                    access="read_write",
                    role="projection_target",
                )
            ]
        ),
    )
    return ModelAutoWiringManifest(contracts=(contract,), errors=())


async def _drive_and_read_dimension(exc: Exception, count: int) -> tuple[str, str, int]:
    """Drive ``count`` failing records, drain, and read the health dimension."""
    t0 = datetime(2026, 10, 3, 18, 58, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)
    bus = _RecordingBus()
    callback = _make_event_bus_callback(
        _IN_TOPIC,
        _engine_for(_HandlerRaises(exc)),
        DispatchResultApplier(event_bus=bus, output_topic=_OUT_TOPIC),
        event_bus=bus,
        allowed_dispatcher_ids={"omn20410-dispatcher"},
        consumer_group=_GROUP,
    )
    for _ in range(count):
        message = MagicMock()
        message.value = json.dumps({"node_id": str(uuid4())}).encode("utf-8")
        await callback(message)
    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=60))
    assert window is not None

    manifest = _projection_manifest()
    projections = select_projection_contracts(manifest)
    verdict = evaluate_projection_liveness(
        projections=projections,
        attached_topics=frozenset({_IN_TOPIC}),
        flow_windows=counters.retained_windows.snapshot(),
        projection_group_suffixes=select_projection_group_suffixes(
            manifest, projections
        ),
    )
    assert verdict.saturation_evaluated
    return (
        dlq_saturation_status(verdict),
        describe_dlq_saturation(verdict),
        len(bus.dlq_records),
    )


@pytest.mark.asyncio
async def test_keyless_tenant_refusals_alone_do_not_trip_dlq_saturation() -> None:
    status, detail, parked = await _drive_and_read_dimension(
        CustomerKeyRefusedError(_REFUSAL_MESSAGE), _N
    )

    assert status == "HEALTHY", detail
    assert _PROJECTION not in detail
    # The change is to the accounting, not the route: every record is still parked.
    assert parked == _N


@pytest.mark.asyncio
async def test_value_error_failures_trip_dlq_saturation() -> None:
    """Positive control: the same drive with a real failure saturates the dimension."""
    status, detail, parked = await _drive_and_read_dimension(
        ValueError("projection row rejected"), _N
    )

    assert status == "DEGRADED", detail
    assert _PROJECTION in detail
    assert parked == _N
