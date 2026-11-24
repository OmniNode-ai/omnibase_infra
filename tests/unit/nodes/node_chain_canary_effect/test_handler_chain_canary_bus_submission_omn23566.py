# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23566: the chain canary submits its delegation over the bus.

The canary used to submit through the separate HTTP API's
``POST /v1/workflows`` with a gateway credential, and fell back to the
runtime's tenant-less ``/skill`` edge when no gateway was declared. The
separate HTTP API is being removed, so the canary now publishes the delegate
request itself, on the lane's broker, stamped with the lane's tenant id.

What these tests pin:

* the delegate request is published once, on the delegate-skill command topic,
  on the lane's declared broker, carrying the probe's minted correlation id and
  the lane tenant on the envelope and in the payload;
* a missing tenant is RED (``SUBMISSION_ROUTE_NOT_CONFIGURED``) with nothing
  published -- there is no fallback to a tenant-less submission;
* no HTTP submission surface is left: the request model carries no gateway
  field and the handler module defines no gateway post.
"""

from __future__ import annotations

import sys
from uuid import UUID, uuid4

import pydantic
import pytest

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.handlers import handler_chain_canary
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_projection_readback_status import (
    EnumProjectionReadbackStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_projection_readback_outcome import (
    ModelProjectionReadbackOutcome,
)

_PROBE_URL = "http://runtime.invalid:8085"
_BOOTSTRAP = "broker.invalid:19092"
_TENANT = "820272f9-4aaf-5add-a2df-0af942852ab2"
_COMMAND_TOPIC = EnumOmnimarketTopic.CMD_DELEGATE_SKILL_V1.value
_SUCCESS_TOPIC = EnumOmnimarketTopic.EVT_DELEGATE_SKILL_COMPLETED_V1.value
_PROJECTION_DSN_ENV = "CHAIN_CANARY_PROJECTION_DSN"
_LEDGER_DSN_ENV = "CHAIN_CANARY_LEDGER_DSN_FOR_TESTS"
_FULL_CHAIN = ("received", "routed", "inference_completed", "terminal")


@pytest.fixture(autouse=True)
def _clean_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pytest"])


def _request(**overrides: object) -> ModelChainCanaryRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": _PROBE_URL,
        "tenant_id": _TENANT,
        "budget_ms": 5_000,
        "terminal_bootstrap_servers": _BOOTSTRAP,
        "projection_dsn_env": _PROJECTION_DSN_ENV,
        "ledger_source_env": _LEDGER_DSN_ENV,
        "expected_ledger_hops": _FULL_CHAIN,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest.model_validate(fields)


class _RecordingPublisher:
    """Injected bus publish. Records what it was handed."""

    def __init__(
        self,
        response: dict[str, object] | None = None,
        error: str = "",
        elapsed_ms: int = 42,
    ) -> None:
        self.response = response if response is not None else {"ok": True}
        self.error = error
        self.elapsed_ms = elapsed_ms
        self.calls: list[tuple[str, str, dict[str, object], float]] = []

    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        envelope: dict[str, object],
        timeout_s: float,
    ) -> tuple[dict[str, object] | None, str, int]:
        self.calls.append((bootstrap, topic, envelope, timeout_s))
        if self.error:
            return None, self.error, self.elapsed_ms
        return self.response, "", self.elapsed_ms


class _Quarantine:
    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[bool | None, int, str]:
        return False, 500, ""


class _Terminal:
    async def __call__(
        self,
        bootstrap: str,
        topics: tuple[str, ...],
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[str | None, int, str]:
        return _SUCCESS_TOPIC, 120, ""


class _Projection:
    async def __call__(
        self, dsn: str, correlation_id: str, timeout_s: float
    ) -> ModelProjectionReadbackOutcome:
        return ModelProjectionReadbackOutcome(
            status=EnumProjectionReadbackStatus.TERMINAL,
            state="COMPLETED",
            traffic_class="synthetic",
        )


class _Ledger:
    async def __call__(
        self, source: str, correlation_id: str, timeout_s: float
    ) -> tuple[tuple[str, ...] | None, bool, str, str, str]:
        return _FULL_CHAIN, True, "pass", "complete", ""


def _handler(
    publisher: _RecordingPublisher | None = None,
) -> tuple[HandlerChainCanary, _RecordingPublisher]:
    publisher = publisher or _RecordingPublisher()
    handler = HandlerChainCanary(
        bus_publish=publisher,
        quarantine_scan=_Quarantine(),
        terminal_readback=_Terminal(),
        projection_readback=_Projection(),
        projection_dsn_lookup=lambda name: (
            "postgresql://p@db.invalid:5436/omnibase_infra"
            if name == _PROJECTION_DSN_ENV
            else ""
        ),
        ledger_replay=_Ledger(),
        ledger_dsn_lookup=lambda name: (
            "postgresql://p@db.invalid:5436/omnibase_infra"
            if name == _LEDGER_DSN_ENV
            else ""
        ),
    )
    return handler, publisher


class TestTheDelegateRequestIsPublishedOnTheBus:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_publishes_once_on_the_command_topic_on_the_lane_broker(
        self,
    ) -> None:
        handler, publisher = _handler()
        result = await handler.handle(_request())
        assert len(publisher.calls) == 1
        bootstrap, topic, _envelope, _timeout = publisher.calls[0]
        assert bootstrap == _BOOTSTRAP
        assert topic == _COMMAND_TOPIC
        assert result.verdict != EnumChainCanaryVerdict.SUBMISSION_ROUTE_NOT_CONFIGURED

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_envelope_carries_the_lane_tenant_and_the_probe_correlation(
        self,
    ) -> None:
        handler, publisher = _handler()
        result = await handler.handle(_request())
        _bootstrap, _topic, envelope, _timeout = publisher.calls[0]
        probe_id = str(result.probe_correlation_id)
        assert envelope["tenant_id"] == _TENANT
        assert envelope["correlation_id"] == probe_id
        assert envelope["event_type"] == "omnimarket.delegate-skill"
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        assert payload["tenant_id"] == _TENANT
        assert payload["correlation_id"] == probe_id
        assert set(payload) == {
            "prompt",
            "task_type",
            "source",
            "max_tokens",
            "provenance",
            "correlation_id",
            "tenant_id",
        }
        assert payload["provenance"]["traffic_class"] == "synthetic"
        # Every run mints its own id; the run's id is never the probe's.
        assert UUID(probe_id) != result.correlation_id

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_publish_failure_is_red_and_names_the_broker_error(self) -> None:
        handler, _publisher = _handler(_RecordingPublisher(error="broker refused"))
        result = await handler.handle(_request())
        assert result.success is False
        assert result.verdict == EnumChainCanaryVerdict.INGRESS_UNREACHABLE
        assert "broker refused" in result.detail


class TestAMissingTenantIsRedWithNoFallback:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_tenant_publishes_nothing_and_names_the_tenant(self) -> None:
        handler, publisher = _handler()
        result = await handler.handle(_request(tenant_id=""))
        assert publisher.calls == []
        assert result.success is False
        assert result.verdict == EnumChainCanaryVerdict.SUBMISSION_ROUTE_NOT_CONFIGURED
        assert "tenant" in result.detail

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_broker_publishes_nothing(self) -> None:
        handler, publisher = _handler()
        result = await handler.handle(_request(terminal_bootstrap_servers=""))
        assert publisher.calls == []
        assert result.success is False
        assert result.verdict == EnumChainCanaryVerdict.SUBMISSION_ROUTE_NOT_CONFIGURED

    @pytest.mark.unit
    def test_a_tenant_that_is_not_a_uuid_is_refused_at_the_model(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="tenant_id"):
            _request(tenant_id="not-a-uuid")


class TestNoHttpSubmissionSurfaceIsLeft:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "removed_field",
        ["gateway_url", "gateway_api_key_env", "gateway_workflow_type"],
    )
    def test_the_request_model_has_no_gateway_field(self, removed_field: str) -> None:
        assert removed_field not in ModelChainCanaryRequest.model_fields
        with pytest.raises(pydantic.ValidationError):
            _request(**{removed_field: "x"})

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "removed_name",
        [
            "_post_workflow_via_gateway",
            "_post_skill_via_httpx",
            "_lookup_gateway_api_key",
        ],
    )
    def test_the_handler_module_defines_no_http_submission(
        self, removed_name: str
    ) -> None:
        assert not hasattr(handler_chain_canary, removed_name)
