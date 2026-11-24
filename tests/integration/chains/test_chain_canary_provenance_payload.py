# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The delegate command the canary publishes carries the synthetic source.

OMN-23566 replaced the ``/skill`` POST (whose body carried a structured
``payload.provenance`` block) with a bus publish on the delegate-skill command
topic. The envelope's payload no longer carries the governed
``ModelDelegationProvenance``; it carries the provenance's ``source``
identifier, which is the marker the delegate-skill consumer classifies the
traffic from. This test pins that the marker on the wire is the synthetic
provenance's source, stamped with the lane's tenant.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
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
_LEDGER_SOURCE_ENV = "CHAIN_CANARY_LEDGER_DSN_FOR_TESTS"
_FULL_CHAIN = ("received", "routed", "inference_completed", "terminal")
# The synthetic provenance's registered adapter source, the one field of the
# governed block that still travels on the wire (OMN-23566).
_SYNTHETIC_SOURCE = "external-client"


class _RecordingPublisher:
    """Injected bus publish: records the envelope, answers a bare broker ACK.

    The ACK carries ``ok`` and never a terminal-event key, so the verdict comes
    from the readback legs and this fixture stays about the envelope's shape.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object], float]] = []

    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        envelope: dict[str, object],
        timeout_s: float,
    ) -> tuple[dict[str, object] | None, str, int]:
        self.calls.append((bootstrap, topic, envelope, timeout_s))
        return {"ok": True}, "", 42


async def _quarantine_clean(
    bootstrap: str,
    topic: str,
    correlation_id: str,
    max_records: int,
    timeout_s: float,
) -> tuple[bool, int, str]:
    return False, 0, ""


async def _terminal_present(
    bootstrap: str,
    topics: tuple[str, ...],
    correlation_id: str,
    max_records: int,
    timeout_s: float,
) -> tuple[str, int, str]:
    return _SUCCESS_TOPIC, 1, ""


async def _projection_terminal(
    dsn: str, correlation_id: str, timeout_s: float
) -> ModelProjectionReadbackOutcome:
    return ModelProjectionReadbackOutcome(
        status=EnumProjectionReadbackStatus.TERMINAL,
        state="COMPLETED",
        traffic_class="synthetic",
    )


async def _ledger_verified(
    source: str, correlation_id: str, timeout_s: float
) -> tuple[tuple[str, ...], bool, str, str, str]:
    return _FULL_CHAIN, True, "pass", "complete", ""


def _ledger_dsn_lookup(name: str) -> str:
    return (
        "postgresql://probe@db.invalid:5436/omnibase_infra"
        if name == _LEDGER_SOURCE_ENV
        else ""
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chain_canary_marks_delegated_skill_payload_as_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        _PROJECTION_DSN_ENV, "postgresql://probe@db.invalid:5436/omnibase_infra"
    )
    publisher = _RecordingPublisher()
    handler = HandlerChainCanary(
        bus_publish=publisher,
        quarantine_scan=_quarantine_clean,
        terminal_readback=_terminal_present,
        projection_readback=_projection_terminal,
        ledger_replay=_ledger_verified,
        ledger_dsn_lookup=_ledger_dsn_lookup,
        kill_switch_disabled=False,
    )

    result = await handler.handle(
        ModelChainCanaryRequest(
            correlation_id=uuid4(),
            probe_url=_PROBE_URL,
            budget_ms=5_000,
            tenant_id=_TENANT,
            terminal_bootstrap_servers=_BOOTSTRAP,
            projection_dsn_env=_PROJECTION_DSN_ENV,
            ledger_source_env=_LEDGER_SOURCE_ENV,
            expected_ledger_hops=_FULL_CHAIN,
        )
    )

    bootstrap, topic, envelope, _timeout = publisher.calls[0]
    assert bootstrap == _BOOTSTRAP
    assert topic == _COMMAND_TOPIC
    # The synthetic marker: the governed provenance's source identifier, on
    # the delegate payload, stamped with the lane's tenant.
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    assert payload["provenance"] == {
        "source": _SYNTHETIC_SOURCE,
        "traffic_class": "synthetic",
        "source_surface": "scheduled-chain-canary",
        "requested_by": "chain-canary",
    }
    assert payload["source"] == payload["provenance"]["source"]
    assert payload["tenant_id"] == _TENANT
    assert envelope["tenant_id"] == _TENANT
    assert envelope["correlation_id"] == str(result.probe_correlation_id)
    # The classification the marker produces, read back through link 2.
    assert result.projection_traffic_class == "synthetic"
