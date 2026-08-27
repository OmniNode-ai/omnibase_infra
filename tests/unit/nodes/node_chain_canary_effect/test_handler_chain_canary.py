# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for HandlerChainCanary — OMN-16773.

The canary's job is to notice that the live delegation chain is dead. The
load-bearing fixture here is ``test_reproduces_omn_16767_signature``, which
replays the EXACT observed shape of the 2026-08-27 incident: the ingress
returns ``ok=false`` with ``error.code == "dispatch_timeout"`` after the full
budget, and the run's own correlation id turns up in the platform quarantine
sink. The canary must call that RED, and must call it ``QUARANTINED`` rather
than the vaguer ``TERMINAL_MISSING`` — those are different pages, and the
quarantine one names the actual defect.

Every test drives the real handler with injected transport. No network.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_quarantine_check_status import (
    EnumQuarantineCheckStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)

_PROBE_URL = "http://runtime.invalid:8085"
_BOOTSTRAP = "broker.invalid:19092"


def _request(**overrides: object) -> ModelChainCanaryRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": _PROBE_URL,
        "budget_ms": 5_000,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest(**fields)  # type: ignore[arg-type]


class _RecordingIngress:
    """Captures the outbound body so tests can assert the recipe shape."""

    def __init__(
        self,
        response: dict[str, object] | None = None,
        error: str = "",
        elapsed_ms: int = 42,
    ) -> None:
        self.response = response
        self.error = error
        self.elapsed_ms = elapsed_ms
        self.calls: list[tuple[str, dict[str, object], float]] = []

    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        self.calls.append((url, body, timeout_s))
        return self.response, self.error, self.elapsed_ms


class _RecordingQuarantine:
    """Injected quarantine tail scanner."""

    def __init__(self, found: bool | None = False, error: str = "", scanned: int = 500):
        self.found = found
        self.error = error
        self.scanned = scanned
        self.calls: list[tuple[str, str, str, int, float]] = []

    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[bool | None, int, str]:
        self.calls.append((bootstrap, topic, correlation_id, max_records, timeout_s))
        return self.found, self.scanned, self.error


def _terminal_response(correlation_id_echo: str = "") -> dict[str, object]:
    """A successful ingress response carrying a terminal event."""
    return {
        "ok": True,
        "command_name": "node_delegate_skill_orchestrator",
        "correlation_id": correlation_id_echo,
        "terminal_event": "omnimarket.delegate-skill-completed",
        "output_payloads": [{"status": "completed"}],
    }


# -- AC1: fresh correlation id, recorded recipe shape ----------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mints_a_fresh_correlation_id_per_run() -> None:
    """AC1 — the probe correlation id is minted per run, never reused."""
    ingress = _RecordingIngress(response=_terminal_response())
    handler = HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=_RecordingQuarantine(),
        kill_switch_disabled=False,
    )

    first = await handler.handle(_request())
    second = await handler.handle(_request())

    assert first.probe_correlation_id != second.probe_correlation_id
    # A real UUID4, not a placeholder.
    assert UUID(str(first.probe_correlation_id)).version == 4
    # And it is the id actually put on the wire, not a decorative field.
    sent_ids = [body["correlation_id"] for _, body, _ in ingress.calls]
    assert sent_ids == [
        str(first.probe_correlation_id),
        str(second.probe_correlation_id),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_posts_the_recorded_delegation_recipe() -> None:
    """AC1 — the body matches the recorded omnidash dispatch shape."""
    ingress = _RecordingIngress(response=_terminal_response())
    handler = HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=_RecordingQuarantine(),
        kill_switch_disabled=False,
    )

    await handler.handle(_request(task_type="test", prompt="ping", max_tokens=32))

    url, body, timeout_s = ingress.calls[0]
    assert url == f"{_PROBE_URL}/skill"
    assert body["command_name"] == "node_delegate_skill_orchestrator"
    assert body["timeout_ms"] == 5_000
    payload = body["payload"]
    assert isinstance(payload, dict)
    assert payload["prompt"] == "ping"
    assert payload["task_type"] == "test"
    assert payload["wait"] is True
    assert payload["max_tokens"] == 32
    # The client budget must exceed the runtime budget or the canary times
    # itself out before the runtime can answer, manufacturing a false RED.
    assert timeout_s > 5.0


# -- AC2 / AC3: verdicts ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_green_when_terminal_lands_and_quarantine_is_clean() -> None:
    quarantine = _RecordingQuarantine(found=False)
    handler = HandlerChainCanary(
        ingress=_RecordingIngress(response=_terminal_response()),
        quarantine_scan=quarantine,
        kill_switch_disabled=False,
    )

    result = await handler.handle(_request(quarantine_bootstrap_servers=_BOOTSTRAP))

    assert result.verdict is EnumChainCanaryVerdict.GREEN
    assert result.success is True
    assert result.quarantine_status is EnumQuarantineCheckStatus.CLEAN
    assert result.terminal_event == "omnimarket.delegate-skill-completed"
    # The quarantine scan was asked about THIS run's correlation id.
    assert quarantine.calls[0][2] == str(result.probe_correlation_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reproduces_omn_16767_signature() -> None:
    """The incident fixture: dispatch_timeout + our correlation in quarantine.

    Observed 2026-08-27T15:32Z on the .201 dev lane. Both symptoms are true
    at once; the canary must report the quarantine one, because that is the
    symptom that names the defect.
    """
    handler = HandlerChainCanary(
        ingress=_RecordingIngress(
            response={
                "ok": False,
                "command_name": "node_delegate_skill_orchestrator",
                "error": {
                    "code": "dispatch_timeout",
                    "message": "Local runtime ingress timed out after 5000 ms",
                    "retryable": True,
                },
            },
            elapsed_ms=5_001,
        ),
        quarantine_scan=_RecordingQuarantine(found=True),
        kill_switch_disabled=False,
    )

    result = await handler.handle(_request(quarantine_bootstrap_servers=_BOOTSTRAP))

    assert result.verdict is EnumChainCanaryVerdict.QUARANTINED
    assert result.success is False
    assert result.quarantine_status is EnumQuarantineCheckStatus.FOUND
    assert result.ingress_error_code == "dispatch_timeout"
    assert "quarantine" in result.detail.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_missing_when_ingress_times_out_without_quarantine() -> None:
    handler = HandlerChainCanary(
        ingress=_RecordingIngress(
            response={
                "ok": False,
                "error": {"code": "dispatch_timeout", "message": "timed out"},
            }
        ),
        quarantine_scan=_RecordingQuarantine(found=False),
        kill_switch_disabled=False,
    )

    result = await handler.handle(_request(quarantine_bootstrap_servers=_BOOTSTRAP))

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.success is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_missing_when_ok_true_but_no_terminal_event() -> None:
    """ok=true is not proof. A 202-shaped success with no terminal is RED.

    This is the OMN-16027 fail-open lesson: the accepting side answering
    cheerfully proves nothing about the chain behind it.
    """
    handler = HandlerChainCanary(
        ingress=_RecordingIngress(response={"ok": True, "terminal_event": ""}),
        quarantine_scan=_RecordingQuarantine(found=False),
        kill_switch_disabled=False,
    )

    result = await handler.handle(_request(quarantine_bootstrap_servers=_BOOTSTRAP))

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.success is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingress_unreachable_is_its_own_verdict() -> None:
    handler = HandlerChainCanary(
        ingress=_RecordingIngress(response=None, error="connection refused"),
        quarantine_scan=_RecordingQuarantine(found=False),
        kill_switch_disabled=False,
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.INGRESS_UNREACHABLE
    assert result.success is False


# -- AC3: the unconfigured quarantine leg must never read as clean ---------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unconfigured_quarantine_reports_skipped_not_clean() -> None:
    quarantine = _RecordingQuarantine(found=False)
    handler = HandlerChainCanary(
        ingress=_RecordingIngress(response=_terminal_response()),
        quarantine_scan=quarantine,
        kill_switch_disabled=False,
    )

    result = await handler.handle(_request(quarantine_bootstrap_servers=""))

    assert result.quarantine_status is EnumQuarantineCheckStatus.SKIPPED_NOT_CONFIGURED
    assert quarantine.calls == []
    # A skipped leg does not block a green terminal — but the result says so
    # out loud rather than implying a check that never ran.
    assert result.verdict is EnumChainCanaryVerdict.GREEN
    assert "not configured" in result.detail.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quarantine_probe_failure_fails_closed() -> None:
    """A configured-but-broken quarantine leg is RED, never a silent pass."""
    handler = HandlerChainCanary(
        ingress=_RecordingIngress(response=_terminal_response()),
        quarantine_scan=_RecordingQuarantine(found=None, error="broker unreachable"),
        kill_switch_disabled=False,
    )

    result = await handler.handle(_request(quarantine_bootstrap_servers=_BOOTSTRAP))

    assert result.quarantine_status is EnumQuarantineCheckStatus.ERROR
    assert result.verdict is EnumChainCanaryVerdict.QUARANTINE_PROBE_FAILED
    assert result.success is False


# -- AC6: kill switch ------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kill_switch_performs_zero_io() -> None:
    ingress = _RecordingIngress(response=_terminal_response())
    quarantine = _RecordingQuarantine()
    handler = HandlerChainCanary(
        ingress=ingress, quarantine_scan=quarantine, kill_switch_disabled=True
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.SKIPPED_DISABLED
    assert result.success is True
    assert ingress.calls == []
    assert quarantine.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kill_switch_read_from_env_at_handle_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-arg contract-driven construction can never miss the switch."""
    ingress = _RecordingIngress(response=_terminal_response())
    handler = HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=_RecordingQuarantine(),
        kill_switch_disabled=False,
    )
    monkeypatch.setenv("ONEX_CHAIN_CANARY_DISABLED", "1")

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.SKIPPED_DISABLED
    assert ingress.calls == []


# -- request validation ----------------------------------------------------


@pytest.mark.unit
def test_probe_url_is_required_and_must_be_http() -> None:
    """Rule 8 — fail fast on a missing target, never guess a default lane."""
    with pytest.raises(ValueError):
        ModelChainCanaryRequest(correlation_id=uuid4(), probe_url="")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ModelChainCanaryRequest(correlation_id=uuid4(), probe_url="192.168.0.1:8085")  # type: ignore[call-arg]


@pytest.mark.unit
def test_probe_url_trailing_slash_is_normalised() -> None:
    req = ModelChainCanaryRequest(
        correlation_id=uuid4(), probe_url="http://runtime.invalid:8085/"
    )
    assert req.probe_url == "http://runtime.invalid:8085"
