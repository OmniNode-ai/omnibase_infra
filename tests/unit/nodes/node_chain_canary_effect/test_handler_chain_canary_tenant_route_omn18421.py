# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18421: the canary submits through the TENANT-BEARING route.

WHAT WAS WRONG
--------------
This probe fired one real delegation every two hours into the runtime's generic
``/skill`` ingress, which reads ``X-Correlation-ID`` and nothing else — no
authorization header, no tenant. So every event the resulting chain published
was unattributed, and omnimarket's delegation projection writer refused every
one of them fail-closed into
``onex.dlq.omnimarket.projection-delegation-malformed.v1``.

The refusal is correct and is not what changes. A projection writer under FORCE
ROW LEVEL SECURITY cannot discover a row's tenant by reading, so attribution is
producer-recorded or it does not exist; stamping a house identity instead would
put rows under a tenant nobody submitted, where the submitting tenant's reader
could never see them, and would block the correctly-attributed terminal behind a
USING-clause refusal. The runtime will not source a tenant either — carrying one
across a hop is its job, inventing one is explicitly forbidden.

So the arrivals stop when the SUBMISSION carries a verified tenant, and the only
surface with an authenticated identity is the onex-api gateway.

THE PART THAT IS EASY TO GET WRONG
----------------------------------
A fallback. If the authenticated route is unavailable and the probe quietly
submits the tenant-less way instead, it goes on reporting PROBE-GREEN on a chain
that projects nothing — which is precisely the false-clean that let the
malformed sink take continuous arrivals while this probe said the chain was
alive. The tests below pin the absence of that fallback as hard as they pin the
happy path, because the fallback is the more tempting code and the worse one.
"""

from __future__ import annotations

import sys
from uuid import uuid4

import pytest

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
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
_GATEWAY_URL = "http://runtime.invalid:8090"
_GATEWAY_KEY_ENV = "CHAIN_CANARY_GATEWAY_API_KEY"
_GATEWAY_KEY = "onex_test_key_never_a_real_credential"
_BOOTSTRAP = "broker.invalid:19092"
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
        "gateway_url": _GATEWAY_URL,
        "gateway_api_key_env": _GATEWAY_KEY_ENV,
        "budget_ms": 5_000,
        "terminal_bootstrap_servers": _BOOTSTRAP,
        "projection_dsn_env": _PROJECTION_DSN_ENV,
        "ledger_source_env": _LEDGER_DSN_ENV,
        "expected_ledger_hops": _FULL_CHAIN,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest(**fields)  # type: ignore[arg-type]


class _RecordingGateway:
    """Injected authenticated submission. Records the key it was handed."""

    def __init__(
        self,
        response: dict[str, object] | None = None,
        error: str = "",
        elapsed_ms: int = 42,
    ) -> None:
        self.response = response if response is not None else {"ok": True}
        self.error = error
        self.elapsed_ms = elapsed_ms
        self.calls: list[tuple[str, dict[str, object], str, float]] = []

    async def __call__(
        self, url: str, body: dict[str, object], api_key: str, timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        self.calls.append((url, body, api_key, timeout_s))
        return self.response, self.error, self.elapsed_ms


class _RecordingSkillIngress:
    """The LEGACY tenant-less route. Present so the tests can prove it is
    never called on the gateway path — an unused stub proves nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], float]] = []

    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        self.calls.append((url, body, timeout_s))
        return {"ok": True}, "", 7


class _RecordingQuarantine:
    def __init__(self, found: bool | None = False) -> None:
        self.found = found

    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[bool | None, int, str]:
        return self.found, 500, ""


class _RecordingTerminal:
    def __init__(self, found: str | None = _SUCCESS_TOPIC) -> None:
        self.found = found

    async def __call__(
        self,
        bootstrap: str,
        topics: tuple[str, ...],
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[str | None, int, str]:
        return self.found, 120, ""


class _RecordingProjection:
    async def __call__(
        self, dsn: str, correlation_id: str, timeout_s: float
    ) -> ModelProjectionReadbackOutcome:
        return ModelProjectionReadbackOutcome(
            status=EnumProjectionReadbackStatus.TERMINAL,
            state="COMPLETED",
            traffic_class="synthetic",
        )


class _RecordingLedger:
    async def __call__(
        self, source: str, correlation_id: str, timeout_s: float
    ) -> tuple[tuple[str, ...] | None, bool, str, str]:
        return _FULL_CHAIN, True, "pass", ""


def _handler(
    *,
    gateway: _RecordingGateway | None = None,
    skill: _RecordingSkillIngress | None = None,
    key: str = _GATEWAY_KEY,
) -> tuple[HandlerChainCanary, _RecordingGateway, _RecordingSkillIngress]:
    gateway = gateway or _RecordingGateway()
    skill = skill or _RecordingSkillIngress()
    handler = HandlerChainCanary(
        ingress=skill,
        gateway_ingress=gateway,
        gateway_key_lookup=lambda name: (key if name == _GATEWAY_KEY_ENV else ""),
        quarantine_scan=_RecordingQuarantine(),
        terminal_readback=_RecordingTerminal(),
        projection_readback=_RecordingProjection(),
        projection_dsn_lookup=lambda name: (
            "postgresql://p@db.invalid:5436/omnibase_infra"
            if name == _PROJECTION_DSN_ENV
            else ""
        ),
        ledger_replay=_RecordingLedger(),
        ledger_dsn_lookup=lambda name: (
            "postgresql://p@db.invalid:5436/omnibase_infra"
            if name == _LEDGER_DSN_ENV
            else ""
        ),
    )
    return handler, gateway, skill


class TestTheProbeTakesTheAuthenticatedRoute:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_submits_to_the_gateway_workflows_endpoint(self) -> None:
        handler, gateway, _skill = _handler()
        await handler.handle(_request())
        assert len(gateway.calls) == 1
        url, _body, _key, _timeout = gateway.calls[0]
        assert url == f"{_GATEWAY_URL}/v1/workflows"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_tenant_less_skill_ingress_is_never_called(self) -> None:
        """AC3's falsifier, stated as a test: a run that still posts to a
        surface reading no authorization header has not taken this route."""
        handler, _gateway, skill = _handler()
        await handler.handle(_request())
        assert skill.calls == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_credential_is_passed_to_the_transport_not_the_body(
        self,
    ) -> None:
        """The key rides in a header argument. A key inside the submission body
        would be serialised into the node payload, onto the bus and into the
        event log."""
        handler, gateway, _skill = _handler()
        await handler.handle(_request())
        _url, body, api_key, _timeout = gateway.calls[0]
        assert api_key == _GATEWAY_KEY
        assert _GATEWAY_KEY not in str(body)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_body_is_exactly_the_gateway_submission_shape(self) -> None:
        """``ModelWorkflowSubmitRequest`` is extra="forbid" and the catalog
        schema is additionalProperties:false. The ``/skill`` keys
        (``command_name``, ``timeout_ms``, ``provenance``, ``wait``,
        ``metadata``) are a 400 here, not fields the gateway ignores."""
        handler, gateway, _skill = _handler()
        await handler.handle(_request())
        _url, body, _key, _timeout = gateway.calls[0]
        assert set(body) == {"workflow_type", "correlation_id", "payload"}
        assert set(body["payload"]) == {  # type: ignore[arg-type]
            "prompt",
            "task_type",
            "source",
            "max_tokens",
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_submits_the_head_hop_workflow_type_not_the_second_hop(
        self,
    ) -> None:
        """``delegation-inference`` would also be accepted and would enter the
        chain at hop 2, leaving the declared head hop non-existent and link 5
        reporting an incomplete chain that was never started."""
        handler, gateway, _skill = _handler()
        await handler.handle(_request())
        _url, body, _key, _timeout = gateway.calls[0]
        assert body["workflow_type"] == "delegate-skill"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_minted_correlation_id_rides_at_the_top_level(self) -> None:
        """Where ``ModelWorkflowSubmitRequest`` declares it. The gateway copies
        it onto both the envelope and the payload, so the correlation-scoped
        readbacks key on the same value they always have."""
        handler, gateway, _skill = _handler()
        result = await handler.handle(_request())
        _url, body, _key, _timeout = gateway.calls[0]
        assert body["correlation_id"] == str(result.probe_correlation_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_receipt_names_the_route_that_was_taken(self) -> None:
        """AC3's other falsifier: a receipt naming the port-8085 ingress."""
        handler, _gateway, _skill = _handler()
        result = await handler.handle(_request())
        assert result.probe_url == _GATEWAY_URL
        ingress_link = result.link_verdicts[0]
        assert ingress_link.status is EnumChainLinkStatus.PASS
        assert "tenant-bearing" in ingress_link.detail


class TestThereIsNoFallbackToTheTenantLessRoute:
    """The tempting code, pinned absent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_credential_is_red_and_publishes_nothing(self) -> None:
        handler, gateway, skill = _handler(key="")
        result = await handler.handle(_request())
        assert result.verdict is EnumChainCanaryVerdict.SUBMISSION_ROUTE_NOT_CONFIGURED
        assert result.success is False
        assert gateway.calls == []
        assert skill.calls == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_credential_names_the_variable_it_looked_for(self) -> None:
        """A red whose detail does not say what is missing teaches people to
        ignore the canary."""
        handler, _gateway, _skill = _handler(key="")
        result = await handler.handle(_request())
        assert _GATEWAY_KEY_ENV in result.detail

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_credential_claims_nothing_about_the_chain(self) -> None:
        """Every link unevaluated, not FAIL. FAIL on link 1 would be a claim
        about the lane; this is a claim about the canary's own wiring."""
        handler, _gateway, _skill = _handler(key="")
        result = await handler.handle(_request())
        assert result.links_proven == 0
        assert result.chain_proof_complete is False
        assert all(
            link.status is EnumChainLinkStatus.NOT_EVALUATED
            for link in result.link_verdicts
        )

    @pytest.mark.unit
    def test_a_gateway_url_without_a_credential_name_is_refused_at_the_model(
        self,
    ) -> None:
        """Refused before the run can publish, so a 401 never becomes a
        misleading ingress-shaped failure in the receipt."""
        with pytest.raises(ValueError, match="gateway_api_key_env"):
            _request(gateway_api_key_env="")

    @pytest.mark.unit
    def test_an_api_key_passed_where_a_variable_name_belongs_is_refused(self) -> None:
        """The transposition is silent and durable: the flag lands in argv, the
        dispatch step echoes its arguments into the run log, and the request is
        serialised into the event log."""
        with pytest.raises(ValueError, match="NAME of the environment variable"):
            _request(gateway_api_key_env="onex_live_abc.def-ghi")


class TestTheLegacyRouteStillWorksWhenDeliberatelyChosen:
    """The negative control for the whole change: an empty ``gateway_url`` is
    still the old behaviour, so the two routes are a choice rather than a
    half-finished migration that silently broke the old one."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_gateway_url_uses_the_skill_ingress(self) -> None:
        handler, gateway, skill = _handler()
        await handler.handle(_request(gateway_url="", gateway_api_key_env=""))
        assert gateway.calls == []
        assert len(skill.calls) == 1
        assert skill.calls[0][0] == f"{_PROBE_URL}/skill"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_legacy_route_receipt_says_it_carries_no_tenant(self) -> None:
        """An honest receipt on the old route. A reader quoting a colour at
        somebody should be able to see, from the receipt alone, that a green
        here is a green about an unattributed chain."""
        handler, _gateway, _skill = _handler()
        result = await handler.handle(_request(gateway_url="", gateway_api_key_env=""))
        assert "NO tenant" in result.link_verdicts[0].detail
