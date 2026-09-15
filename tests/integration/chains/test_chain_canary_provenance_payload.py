# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

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
_SUCCESS_TOPIC = EnumOmnimarketTopic.EVT_DELEGATE_SKILL_COMPLETED_V1.value
_PROJECTION_DSN_ENV = "CHAIN_CANARY_PROJECTION_DSN"
_LEDGER_SOURCE_ENV = "CHAIN_CANARY_LEDGER_DSN_FOR_TESTS"
_FULL_CHAIN = ("received", "routed", "inference_completed", "terminal")


class _RecordingIngress:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], float]] = []

    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object], str, int]:
        self.calls.append((url, body, timeout_s))
        return (
            {
                "ok": True,
                "command_name": "node_delegate_skill_orchestrator",
                "terminal_event": "omnimarket.delegate-skill-completed",
                "output_payloads": [{"status": "completed"}],
            },
            "",
            42,
        )


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
) -> tuple[tuple[str, ...], bool, str, str]:
    return _FULL_CHAIN, True, "pass", ""


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
    ingress = _RecordingIngress()
    handler = HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=_quarantine_clean,
        terminal_readback=_terminal_present,
        projection_readback=_projection_terminal,
        ledger_replay=_ledger_verified,
        ledger_dsn_lookup=_ledger_dsn_lookup,
        kill_switch_disabled=False,
    )

    await handler.handle(
        ModelChainCanaryRequest(
            correlation_id=uuid4(),
            probe_url=_PROBE_URL,
            budget_ms=5_000,
            terminal_bootstrap_servers="broker.invalid:19092",
            projection_dsn_env=_PROJECTION_DSN_ENV,
            ledger_source_env=_LEDGER_SOURCE_ENV,
            expected_ledger_hops=_FULL_CHAIN,
        )
    )

    _, body, _ = ingress.calls[0]
    payload = body["payload"]
    assert isinstance(payload, dict)
    assert payload["provenance"] == {
        "source": "external-client",
        "traffic_class": "synthetic",
        "source_surface": "scheduled-chain-canary",
        "requested_by": "chain-canary",
    }
    assert payload["source"] == payload["provenance"]["source"]
