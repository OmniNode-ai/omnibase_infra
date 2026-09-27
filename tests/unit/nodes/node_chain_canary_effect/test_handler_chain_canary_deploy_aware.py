# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19811 -- the canary asks the deploy agent before blaming the chain.

The incident
------------
chain-canary run 36202173467 (2026-09-25T23:44Z, .201 dev lane) reported
``terminal_missing``: deploy-agent job 193cfeda recreated ``omninode-runtime``
inside the probe's 120 s budget. The chain was healthy; the lane was being
replaced under the probe, and nothing in the canary knew to ask.

What these tests pin
--------------------
* A TERMINAL_MISSING whose window overlapped a deploy job waits for the lane to
  converge, retries ONCE, and the receipt names the deploy job (AC1).
* A TERMINAL_MISSING with no deploy in its window stays RED with no retry, and
  so does one where the agent could not be read, one whose deploy never
  converged, and a retry that itself ends TERMINAL_MISSING (AC2). These are the
  cases that stop the retry from becoming a way for a dead chain to go green.
* A probe that starts while a deploy is in flight waits for it before firing
  (AC3), and every other verdict is left alone.

Every test drives the real handler with injected transports, a fake clock and a
fake sleep. No network, no wall-clock waits.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.deploy_agent_window import (
    deploys_in_window,
    queued_commands_from_payload,
    snapshot_from_payloads,
)
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_deploy_window_status import (
    EnumDeployWindowStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_projection_readback_status import (
    EnumProjectionReadbackStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_deploy_agent_snapshot import (
    ModelDeployAgentSnapshot,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_projection_readback_outcome import (
    ModelProjectionReadbackOutcome,
)

_PROBE_URL = "http://runtime.invalid:8085"
_AGENT_URL = "http://agent.invalid:8098"
_BOOTSTRAP = "broker.invalid:19092"
_SUCCESS_TOPIC = EnumOmnimarketTopic.EVT_DELEGATE_SKILL_COMPLETED_V1.value
_PROJECTION_DSN_ENV = "CHAIN_CANARY_PROJECTION_DSN"
_LEDGER_SOURCE_ENV = "CHAIN_CANARY_LEDGER_DSN_FOR_TESTS"
_FULL_CHAIN = ("received", "routed", "inference_completed", "terminal")
# The incident's deploy job, spelled as a full correlation id.
_DEPLOY_JOB = UUID("193cfeda-0000-4000-8000-000000000001")
_T0 = datetime(2026, 9, 25, 23, 44, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PROJECTION_DSN_ENV, "postgresql://probe@db.invalid:5436/x")
    monkeypatch.setattr(sys, "argv", ["pytest"])


class _Clock:
    """A clock the probe and the fake sleep both move."""

    def __init__(self) -> None:
        self.now = _T0
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


class _Ingress:
    """Each submission takes ``elapsed_s`` of fake wall clock."""

    def __init__(
        self,
        clock: _Clock,
        elapsed_s: float = 125.0,
        unreachable_calls: frozenset[int] = frozenset(),
    ) -> None:
        self.clock = clock
        self.elapsed_s = elapsed_s
        self.calls = 0
        # 1-based call numbers whose connection is refused, the way onex-api
        # answered while deploy 3d3f4b1a recreated it (lab run 36279784915).
        self.unreachable_calls = unreachable_calls

    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        self.calls += 1
        if self.calls in self.unreachable_calls:
            self.clock.now += timedelta(seconds=1)
            return None, "ConnectError: All connection attempts failed", 1000
        self.clock.now += timedelta(seconds=self.elapsed_s)
        # The ingress answer is recorded, never trusted (OMN-16931): the
        # verdict comes from the terminal readback below.
        return {"ok": True}, "", int(self.elapsed_s * 1000)


class _TerminalReadback:
    """Answers per attempt: ``""`` is scanned-and-absent, a topic is found."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def __call__(
        self,
        bootstrap: str,
        topics: tuple[str, ...],
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[str | None, int, str]:
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return answer, 120, ""


async def _quarantine(
    bootstrap: str, topic: str, correlation_id: str, max_records: int, timeout_s: float
) -> tuple[bool | None, int, str]:
    return False, 500, ""


async def _projection(
    dsn: str, correlation_id: str, timeout_s: float
) -> ModelProjectionReadbackOutcome:
    return ModelProjectionReadbackOutcome(
        status=EnumProjectionReadbackStatus.TERMINAL, state="COMPLETED"
    )


async def _ledger(
    source: str, correlation_id: str, timeout_s: float
) -> tuple[tuple[str, ...] | None, bool, str, str]:
    return _FULL_CHAIN, True, "pass", ""


def _ledger_dsn_lookup(name: str) -> str:
    return "postgresql://probe@db.invalid:5436/x" if name == _LEDGER_SOURCE_ENV else ""


class _Agent:
    """Scripted deploy agent. Each read takes the next snapshot factory; the
    last one repeats. A factory receives the read's ``observed_at``."""

    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.reads: list[datetime] = []

    async def __call__(
        self, agent_url: str, timeout_s: float, observed_at: datetime
    ) -> ModelDeployAgentSnapshot:
        assert agent_url == _AGENT_URL
        factory = self.script[min(len(self.reads), len(self.script) - 1)]
        self.reads.append(observed_at)
        assert callable(factory)
        snapshot = factory(observed_at)
        assert isinstance(snapshot, ModelDeployAgentSnapshot)
        return snapshot


def _idle(
    last_cid: UUID | None = None,
    accepted: datetime | None = None,
    completed: datetime | None = None,
) -> object:
    def build(observed_at: datetime) -> ModelDeployAgentSnapshot:
        return ModelDeployAgentSnapshot(
            observed_at=observed_at,
            readable=True,
            state="idle",
            last_correlation_id=last_cid,
            last_accepted_at=accepted,
            last_completed_at=completed,
        )

    return build


def _deploying(cid: UUID, accepted: datetime) -> object:
    def build(observed_at: datetime) -> ModelDeployAgentSnapshot:
        return ModelDeployAgentSnapshot(
            observed_at=observed_at,
            readable=True,
            state="deploying",
            active_correlation_id=cid,
            active_accepted_at=accepted,
        )

    return build


def _unreadable(observed_at: datetime) -> ModelDeployAgentSnapshot:
    return ModelDeployAgentSnapshot(
        observed_at=observed_at, readable=False, error="connection refused"
    )


def _request(**overrides: object) -> ModelChainCanaryRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": _PROBE_URL,
        "budget_ms": 120_000,
        "terminal_bootstrap_servers": _BOOTSTRAP,
        "quarantine_bootstrap_servers": _BOOTSTRAP,
        "projection_dsn_env": _PROJECTION_DSN_ENV,
        "ledger_source_env": _LEDGER_SOURCE_ENV,
        "expected_ledger_hops": _FULL_CHAIN,
        "settle_seconds": 0,
        "deploy_agent_url": _AGENT_URL,
        "deploy_wait_seconds": 1200,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest(**fields)  # type: ignore[arg-type]


class _Ready:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def __call__(self, url: str, timeout_s: float) -> bool:
        self.calls.append(url)
        return self.answer


def _handler(
    clock: _Clock,
    agent: _Agent,
    readback: _TerminalReadback,
    ingress: _Ingress | None = None,
    ready: _Ready | None = None,
) -> tuple[HandlerChainCanary, _Ingress]:
    ingress = ingress or _Ingress(clock)
    handler = HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=_quarantine,
        terminal_readback=readback,
        projection_readback=_projection,
        ledger_replay=_ledger,
        ledger_dsn_lookup=_ledger_dsn_lookup,
        kill_switch_disabled=False,
        deploy_agent_read=agent,
        lane_ready=ready or _Ready(),
        clock=clock,
        sleep=clock.sleep,
    )
    return handler, ingress


# -- AC1: a deploy in the window earns exactly one retry ---------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deploy_in_window_retries_once_and_names_the_job() -> None:
    """The run-36202173467 shape, and the reason for this ticket.

    The agent was idle at pre-fire; job 193cfeda was accepted 30 s into the
    probe and was still running when the attempt ended TERMINAL_MISSING. The
    run waits for it, retries once, and the retry's GREEN is the verdict --
    with the first attempt and the deploy job on the receipt.
    """
    clock = _Clock()
    accepted = _T0 + timedelta(seconds=30)
    finished = _T0 + timedelta(seconds=400)
    agent = _Agent(
        _idle(),  # pre-fire
        _deploying(_DEPLOY_JOB, accepted),  # after the attempt
        _deploying(_DEPLOY_JOB, accepted),  # first convergence poll
        _idle(_DEPLOY_JOB, accepted, finished),  # converged
    )
    readback = _TerminalReadback("", _SUCCESS_TOPIC)
    handler, ingress = _handler(clock, agent, readback)

    result = await handler.handle(_request())

    assert ingress.calls == 2
    assert result.verdict is EnumChainCanaryVerdict.GREEN
    assert result.success is True
    window = result.deploy_window
    assert window.status is EnumDeployWindowStatus.DEPLOY_IN_WINDOW_RETRIED
    assert window.retried is True
    assert window.deploy_correlation_ids == (_DEPLOY_JOB,)
    assert window.first_attempt_verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert window.first_attempt_probe_correlation_id is not None
    assert window.first_attempt_probe_correlation_id != result.probe_correlation_id
    assert window.convergence_waited_seconds > 0
    assert str(_DEPLOY_JOB) in result.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deploy_that_completed_inside_the_window_also_retries() -> None:
    """A job accepted before the window and completed inside it recreated the
    runtime mid-probe just as surely as one still running."""
    clock = _Clock()
    agent = _Agent(
        _idle(),
        _idle(
            _DEPLOY_JOB,
            _T0 - timedelta(seconds=600),
            _T0 + timedelta(seconds=60),
        ),
    )
    handler, ingress = _handler(clock, agent, _TerminalReadback("", _SUCCESS_TOPIC))

    result = await handler.handle(_request())

    assert ingress.calls == 2
    assert (
        result.deploy_window.status is EnumDeployWindowStatus.DEPLOY_IN_WINDOW_RETRIED
    )
    assert result.deploy_window.deploy_correlation_ids == (_DEPLOY_JOB,)
    assert result.verdict is EnumChainCanaryVerdict.GREEN


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_retry_that_is_still_terminal_missing_stays_red_and_is_not_retried_again() -> (
    None
):
    """One retry, never two. A chain dead on both sides of a deploy is dead."""
    clock = _Clock()
    agent = _Agent(
        _idle(),
        _idle(_DEPLOY_JOB, _T0 + timedelta(seconds=10), _T0 + timedelta(seconds=90)),
    )
    handler, ingress = _handler(clock, agent, _TerminalReadback(""))

    result = await handler.handle(_request())

    assert ingress.calls == 2
    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.success is False
    assert (
        result.deploy_window.status is EnumDeployWindowStatus.DEPLOY_IN_WINDOW_RETRIED
    )


# -- AC2: no deploy, no retry ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_in_window_terminal_missing_stays_red_without_retry() -> None:
    """The dead chain this canary exists to catch. The agent's last job
    completed an hour before the window, so nothing explains the silence."""
    clock = _Clock()
    agent = _Agent(
        _idle(
            UUID("01d00000-0000-4000-8000-000000000000"),
            _T0 - timedelta(hours=2),
            _T0 - timedelta(hours=1),
        )
    )
    handler, ingress = _handler(clock, agent, _TerminalReadback("", _SUCCESS_TOPIC))

    result = await handler.handle(_request())

    assert ingress.calls == 1
    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.success is False
    assert result.deploy_window.status is EnumDeployWindowStatus.NO_DEPLOY_IN_WINDOW
    assert result.deploy_window.retried is False
    assert result.deploy_window.deploy_correlation_ids == ()
    assert clock.slept == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_unreadable_agent_never_turns_red_into_a_retry() -> None:
    """An agent nobody could read is no evidence of a deploy. Fail closed."""
    clock = _Clock()
    agent = _Agent(_unreadable)
    handler, ingress = _handler(clock, agent, _TerminalReadback("", _SUCCESS_TOPIC))

    result = await handler.handle(_request())

    assert ingress.calls == 1
    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.deploy_window.status is EnumDeployWindowStatus.AGENT_UNREADABLE
    assert result.deploy_window.preflight_agent_error == "connection refused"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_configured_behaves_as_before() -> None:
    """With no agent URL the run makes no deploy claim and never retries."""
    clock = _Clock()
    agent = _Agent(_unreadable)
    handler, ingress = _handler(clock, agent, _TerminalReadback("", _SUCCESS_TOPIC))

    result = await handler.handle(_request(deploy_agent_url=""))

    assert ingress.calls == 1
    assert agent.reads == []
    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.deploy_window.status is EnumDeployWindowStatus.NOT_CONFIGURED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_convergence_inside_the_budget_means_no_retry() -> None:
    """A deploy overlapped, but the lane was still being replaced when the wait
    budget ran out. Firing again would only repeat the first attempt."""
    clock = _Clock()
    accepted = _T0 + timedelta(seconds=20)
    agent = _Agent(_idle(), _deploying(_DEPLOY_JOB, accepted))
    handler, ingress = _handler(clock, agent, _TerminalReadback("", _SUCCESS_TOPIC))

    result = await handler.handle(_request(deploy_wait_seconds=60))

    assert ingress.calls == 1
    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    window = result.deploy_window
    assert window.status is EnumDeployWindowStatus.DEPLOY_IN_WINDOW_NOT_CONVERGED
    assert window.deploy_correlation_ids == (_DEPLOY_JOB,)
    assert window.convergence_waited_seconds >= 60


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_retry_when_readiness_never_answers() -> None:
    """Agent idle is half of convergence; the lane's readiness route is the
    other half. A runtime that never answers /health is not converged."""
    clock = _Clock()
    agent = _Agent(
        _idle(),
        _idle(_DEPLOY_JOB, _T0 + timedelta(seconds=10), _T0 + timedelta(seconds=90)),
    )
    handler, ingress = _handler(
        clock,
        agent,
        _TerminalReadback("", _SUCCESS_TOPIC),
        ready=_Ready(answer=False),
    )

    result = await handler.handle(_request(deploy_wait_seconds=45))

    assert ingress.calls == 1
    assert (
        result.deploy_window.status
        is EnumDeployWindowStatus.DEPLOY_IN_WINDOW_NOT_CONVERGED
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_retry_for_a_green_first_attempt() -> None:
    """A GREEN first attempt is never re-fired, even with a deploy running."""
    clock = _Clock()
    agent = _Agent(_idle(), _deploying(_DEPLOY_JOB, _T0 + timedelta(seconds=5)))
    handler, ingress = _handler(clock, agent, _TerminalReadback(_SUCCESS_TOPIC))

    result = await handler.handle(_request())

    assert ingress.calls == 1
    assert result.verdict is EnumChainCanaryVerdict.GREEN
    assert result.deploy_window.status is EnumDeployWindowStatus.NOT_NEEDED


# -- AC3: a deploy in flight at start is waited out -----------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_flight_deploy_at_start_is_waited_out_before_firing() -> None:
    clock = _Clock()
    accepted = _T0 - timedelta(seconds=300)
    agent = _Agent(
        _deploying(_DEPLOY_JOB, accepted),  # pre-fire read
        _deploying(_DEPLOY_JOB, accepted),  # first wait poll
        _deploying(_DEPLOY_JOB, accepted),  # second wait poll
        _idle(_DEPLOY_JOB, accepted, _T0 + timedelta(seconds=20)),
    )
    ready = _Ready()
    readback = _TerminalReadback(_SUCCESS_TOPIC)
    ingress = _Ingress(clock)
    handler, _ = _handler(clock, agent, readback, ingress=ingress, ready=ready)

    result = await handler.handle(_request())

    assert ingress.calls == 1
    assert result.verdict is EnumChainCanaryVerdict.GREEN
    window = result.deploy_window
    assert window.preflight_deploy_correlation_id == _DEPLOY_JOB
    assert window.preflight_converged is True
    assert window.preflight_waited_seconds >= 30
    # The probe fired only after the agent went idle and readiness answered.
    assert ready.calls == [_PROBE_URL]
    assert window.window_started_at >= (_T0 + timedelta(seconds=30)).isoformat()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_flight_wait_counts_against_the_retry_budget() -> None:
    """The pre-fire wait and the retry wait share one budget, so a run that
    spent it all waiting cannot also wait out a second deploy."""
    clock = _Clock()
    accepted = _T0 - timedelta(seconds=30)
    agent = _Agent(_deploying(_DEPLOY_JOB, accepted))
    handler, ingress = _handler(clock, agent, _TerminalReadback("", _SUCCESS_TOPIC))

    result = await handler.handle(_request(deploy_wait_seconds=60))

    assert ingress.calls == 1
    window = result.deploy_window
    assert window.preflight_converged is False
    assert window.status is EnumDeployWindowStatus.DEPLOY_IN_WINDOW_NOT_CONVERGED
    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_flight_gateway_readiness_is_checked_when_the_gateway_route_is_used() -> (
    None
):
    clock = _Clock()
    accepted = _T0 - timedelta(seconds=30)
    agent = _Agent(
        _deploying(_DEPLOY_JOB, accepted),
        _idle(_DEPLOY_JOB, accepted, _T0),
    )
    ready = _Ready()
    handler = HandlerChainCanary(
        gateway_ingress=_gateway_ingress(clock),
        gateway_key_lookup=lambda name: "key-for-tests",
        quarantine_scan=_quarantine,
        terminal_readback=_TerminalReadback(_SUCCESS_TOPIC),
        projection_readback=_projection,
        ledger_replay=_ledger,
        ledger_dsn_lookup=_ledger_dsn_lookup,
        kill_switch_disabled=False,
        deploy_agent_read=agent,
        lane_ready=ready,
        clock=clock,
        sleep=clock.sleep,
    )

    await handler.handle(
        _request(
            gateway_url="http://gateway.invalid:8090",
            gateway_api_key_env="CHAIN_CANARY_GATEWAY_API_KEY",
        )
    )

    assert ready.calls == [_PROBE_URL, "http://gateway.invalid:8090"]


def _gateway_ingress(clock: _Clock) -> object:
    async def post(
        url: str, body: dict[str, object], api_key: str, timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        clock.now += timedelta(seconds=5)
        return {"ok": True}, "", 5000

    return post


# -- the window arithmetic, against the agent's real payload shapes -------------


@pytest.mark.unit
def test_snapshot_reads_the_agents_health_and_job_payloads() -> None:
    """Payload shapes copied from the .201 agent's /health and /job on
    2026-09-26 (scripts/deploy-agent/deploy_agent/health.py)."""
    snapshot = snapshot_from_payloads(
        observed_at=_T0,
        health={
            "state": "deploying",
            "active_job": {
                "correlation_id": "656c0f2d-202a-4c2c-b21f-d67ea0353871",
                "current_phase": "seed",
                "started_at": "2026-09-26T19:23:31.029559+00:00",
            },
            "last_result": {
                "correlation_id": "6fed211f-8fcb-4f56-840e-028d847ee16a",
                "status": "success",
                "completed_at": "2026-09-26T19:10:22.282056+00:00",
                "settling": False,
            },
        },
        last_job={"accepted_at": "2026-09-26T18:55:00+00:00"},
    )

    assert snapshot.busy is True
    assert snapshot.active_correlation_id == UUID(
        "656c0f2d-202a-4c2c-b21f-d67ea0353871"
    )
    assert snapshot.last_accepted_at == datetime(2026, 9, 26, 18, 55, tzinfo=UTC)
    window = deploys_in_window(
        snapshot,
        datetime(2026, 9, 26, 19, 24, tzinfo=UTC),
        datetime(2026, 9, 26, 19, 26, tzinfo=UTC),
    )
    assert window == (UUID("656c0f2d-202a-4c2c-b21f-d67ea0353871"),)


@pytest.mark.unit
def test_a_job_accepted_after_the_window_is_not_in_it() -> None:
    snapshot = ModelDeployAgentSnapshot(
        observed_at=_T0,
        readable=True,
        state="deploying",
        active_correlation_id=_DEPLOY_JOB,
        active_accepted_at=_T0 + timedelta(minutes=10),
    )
    assert deploys_in_window(snapshot, _T0, _T0 + timedelta(minutes=2)) == ()


@pytest.mark.unit
def test_an_unreadable_snapshot_has_no_jobs_and_is_not_idle_evidence() -> None:
    snapshot = ModelDeployAgentSnapshot(observed_at=_T0, readable=False, error="x")
    assert deploys_in_window(snapshot, _T0, _T0 + timedelta(minutes=2)) == ()


# -- /queue: commands waiting behind an idle agent -------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_flight_queued_command_behind_an_idle_agent_is_waited_out() -> None:
    """An idle agent with a deploy command already queued is about to replace
    the lane; firing now would land the probe on that deploy."""
    clock = _Clock()

    def queued(observed_at: datetime) -> ModelDeployAgentSnapshot:
        return ModelDeployAgentSnapshot(
            observed_at=observed_at,
            readable=True,
            state="idle",
            queued_commands=1,
        )

    started = _T0 + timedelta(seconds=5)
    agent = _Agent(
        queued,  # pre-fire read
        queued,  # first wait poll
        _deploying(_DEPLOY_JOB, started),  # the queued command started
        _idle(_DEPLOY_JOB, started, _T0 + timedelta(seconds=40)),
    )
    ingress = _Ingress(clock)
    handler, _ = _handler(
        clock, agent, _TerminalReadback(_SUCCESS_TOPIC), ingress=ingress
    )

    result = await handler.handle(_request())

    assert ingress.calls == 1
    assert result.verdict is EnumChainCanaryVerdict.GREEN
    window = result.deploy_window
    assert window.preflight_queued_commands == 1
    # The wait was caused by the queue, not by a job, so no job is named.
    assert window.preflight_deploy_correlation_id is None
    assert window.preflight_converged is True
    assert window.preflight_waited_seconds >= 30


@pytest.mark.unit
def test_queue_depth_is_read_only_when_it_is_a_current_count() -> None:
    assert queued_commands_from_payload({"commands_ahead": 2}) == 2
    assert (
        queued_commands_from_payload(
            {"commands_ahead": 0, "control_topic_lag_age_seconds": 3.0}
        )
        == 0
    )
    # Unknown, not zero: a stale lag sample, an unknown depth, a non-count,
    # and an unread /queue each leave the depth unread.
    assert (
        queued_commands_from_payload(
            {"commands_ahead": 0, "control_topic_lag_age_seconds": 900.0}
        )
        is None
    )
    assert queued_commands_from_payload({"commands_ahead": None}) is None
    assert queued_commands_from_payload({"commands_ahead": True}) is None
    assert queued_commands_from_payload({"commands_ahead": -1}) is None
    assert queued_commands_from_payload(None) is None


@pytest.mark.unit
def test_an_unread_queue_does_not_make_an_idle_agent_busy() -> None:
    snapshot = snapshot_from_payloads(
        observed_at=_T0,
        health={"state": "idle", "active_job": None, "last_result": None},
        last_job=None,
        queue={"commands_ahead": None, "control_topic_lag_reason": "no sample"},
    )
    assert snapshot.queued_commands is None
    assert snapshot.busy is False


# -- INGRESS_UNREACHABLE: the other face of a redeploy ---------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deploy_in_window_ingress_unreachable_retries_once() -> None:
    """Lab run 36279784915 (2026-09-26T23:54Z): fired while deploy 3d3f4b1a
    recreated onex-api, so the submission route refused the connection. The
    deploy explains it exactly as it explains a missing terminal."""
    clock = _Clock()
    started = _T0 - timedelta(seconds=60)
    agent = _Agent(
        _idle(),  # pre-fire
        _deploying(_DEPLOY_JOB, started),  # after the attempt
        _idle(_DEPLOY_JOB, started, _T0 + timedelta(seconds=200)),
    )
    ingress = _Ingress(clock, unreachable_calls=frozenset({1}))
    handler, _ = _handler(
        clock, agent, _TerminalReadback(_SUCCESS_TOPIC), ingress=ingress
    )

    result = await handler.handle(_request())

    assert ingress.calls == 2
    assert result.verdict is EnumChainCanaryVerdict.GREEN
    window = result.deploy_window
    assert window.status is EnumDeployWindowStatus.DEPLOY_IN_WINDOW_RETRIED
    assert window.first_attempt_verdict is EnumChainCanaryVerdict.INGRESS_UNREACHABLE
    assert window.deploy_correlation_ids == (_DEPLOY_JOB,)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_ingress_unreachable_stays_red_without_retry() -> None:
    """An ingress that is down with no deploy behind it is a real outage."""
    clock = _Clock()
    agent = _Agent(
        _idle(
            UUID("01d00000-0000-4000-8000-000000000000"),
            _T0 - timedelta(hours=2),
            _T0 - timedelta(hours=1),
        )
    )
    ingress = _Ingress(clock, unreachable_calls=frozenset({1, 2}))
    handler, _ = _handler(
        clock, agent, _TerminalReadback(_SUCCESS_TOPIC), ingress=ingress
    )

    result = await handler.handle(_request())

    assert ingress.calls == 1
    assert result.verdict is EnumChainCanaryVerdict.INGRESS_UNREACHABLE
    assert result.success is False
    assert result.deploy_window.status is EnumDeployWindowStatus.NO_DEPLOY_IN_WINDOW
    assert "ingress_unreachable" in result.deploy_window.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_deploy_retry_for_a_red_a_redeploy_does_not_explain() -> None:
    """A deploy running in the window does not earn a retry for a verdict it
    cannot cause: an unreadable bus is not a missing terminal."""
    clock = _Clock()
    agent = _Agent(_idle(), _deploying(_DEPLOY_JOB, _T0 + timedelta(seconds=5)))
    handler, ingress = _handler(clock, agent, _TerminalReadback(None))  # type: ignore[arg-type]

    result = await handler.handle(_request())

    assert ingress.calls == 1
    assert result.success is False
    assert result.verdict not in (
        EnumChainCanaryVerdict.TERMINAL_MISSING,
        EnumChainCanaryVerdict.INGRESS_UNREACHABLE,
    )
    assert result.deploy_window.status is EnumDeployWindowStatus.NOT_NEEDED
    assert result.deploy_window.retried is False
