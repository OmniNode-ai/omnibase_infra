# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The link-2 DSN is DECLARED by name, resolved from env, and refused otherwise.

OMN-18060. Chain-canary run 34281968883 proved 3 of the 5 OMN-16025 links and
failed closed on link 2:

    projection_readback_not_configured -- no DSN was configured for the
    projection readback

That refusal was correct and the gate was unclosable: nothing in the repo said
where the DSN comes from. The broker had exactly this problem eight days
earlier (OMN-17926) and it was solved by DECLARING it in omnimarket's
``config/ci_bus_lanes.yaml``. This file pins the same treatment for the DSN,
with the one difference that makes a DSN not a broker: a broker address is
config and can be committed, a DSN is a credential and cannot. So what is
declared is the NAME of the environment variable the value arrives in, and
every path by which a VALUE could get where a NAME belongs is refused.

The refusals, and what each one is actually protecting
-----------------------------------------------------
* **No declaration / a lane that is not ``dev``.** The canary publishes a live
  delegation and then reads a database. Both halves are dev-lane-only by the
  workflow's own declared scope; stability-test, judge and prod are read-only
  surfaces it has no ticket for.
* **A ``dsn_env`` that parses as a DSN.** The overlay is committed and
  CODEOWNERS-reviewed. A value pasted there is a credential in git history.
* **A DSN anywhere in argv.** ``/proc/<pid>/cmdline`` is world-readable, and
  the dispatch step echoes its own configuration into the run log.
* **A DSN whose role is SUPERUSER or BYPASSRLS.** This one is not about
  disclosure. Such a role is exempt from row-level security unconditionally,
  so a green read through it cannot distinguish "the projection is readable"
  from "the projection is readable by an identity nothing else has". The probe
  is un-forgeable — it asks the SERVER who ``current_user`` is, rather than
  trusting anything the caller supplied.

Every one of them reports a NON-passing link 2. None of them falls back to the
bus terminal: OMN-14843 measured 26 of 38 correlations stranded mid-FSM while
the topic layer was healthy at that same moment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
    _readback_projection_via_asyncpg,
)
from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
    PROJECTION_READBACK_DSN_ENV_NAME_VAR,
    PROJECTION_READBACK_LANES,
    dsn_shaped_argv_flags,
    load_lane_projection_readback,
    looks_like_a_dsn,
    projection_readback_env,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
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
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_projection_readback_outcome import (
    ModelProjectionReadbackOutcome,
)

pytestmark = pytest.mark.unit

_PROBE_URL = "http://runtime.invalid:8085"
_BOOTSTRAP = "broker.invalid:19092"
_SUCCESS_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_DSN_ENV = "CHAIN_CANARY_PROJECTION_DSN"
# `.invalid` is RFC 2606 reserved, so a fixture that accidentally escaped its
# stub cannot reach a real Postgres and pass by luck.
_DSN = "postgresql://chain_canary_reader:hunter2@db.invalid:5436/omnibase_infra"

#: Shaped after the live omnimarket `dev` lane after this ticket.
_DECLARED_OVERLAY = f"""
default: inmemory

lanes:
  dev:
    broker: "broker.example.internal:19092"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
    projection_readback:
      dsn_env: {_DSN_ENV}
  stability:
    broker: inmemory
"""


def _overlay(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Declaration resolution
# ---------------------------------------------------------------------------


def test_the_dev_lane_declaration_resolves_to_a_name(tmp_path: Path) -> None:
    declaration = load_lane_projection_readback(
        _overlay(tmp_path, _DECLARED_OVERLAY), "dev"
    )

    assert declaration.lane == "dev"
    assert declaration.dsn_env == _DSN_ENV
    assert not looks_like_a_dsn(declaration.dsn_env)


def test_only_the_name_is_exported_to_the_workflow(tmp_path: Path) -> None:
    """The exported env carries a NAME and nothing that could be a value.

    The asymmetry with ``lane_transport_env`` is deliberate and is the whole
    security property: that function exports a broker, a protocol and a
    mechanism because all three are config. This one exports the NAME of the
    variable the credential arrives in, and the credential itself is injected
    by the job's own secret block, never by this code.
    """
    declaration = load_lane_projection_readback(
        _overlay(tmp_path, _DECLARED_OVERLAY), "dev"
    )

    exported = projection_readback_env(declaration)

    assert exported == {PROJECTION_READBACK_DSN_ENV_NAME_VAR: _DSN_ENV}
    assert not any(looks_like_a_dsn(value) for value in exported.values())


# ---------------------------------------------------------------------------
# 2. Every refusal
# ---------------------------------------------------------------------------


def test_a_lane_with_no_projection_block_is_refused(tmp_path: Path) -> None:
    """Absent means absent. There is no default DSN and no fallback."""
    body = """
lanes:
  dev:
    broker: "broker.example.internal:19092"
    security_protocol: PLAINTEXT
"""
    with pytest.raises(ValueError, match="declares no 'projection_readback' block"):
        load_lane_projection_readback(_overlay(tmp_path, body), "dev")


@pytest.mark.parametrize("lane", ["stability", "prod", "judge", "lakshman"])
def test_a_declaration_on_a_non_dev_lane_is_refused(tmp_path: Path, lane: str) -> None:
    """Lane scope is enforced, not documented.

    Refused BEFORE the file is even read, so a declaration on a read-only lane
    cannot be honoured by an overlay that happens to carry one.
    """
    assert lane not in PROJECTION_READBACK_LANES
    body = f"""
lanes:
  {lane}:
    broker: "broker.example.internal:19092"
    security_protocol: PLAINTEXT
    projection_readback:
      dsn_env: {_DSN_ENV}
"""
    with pytest.raises(ValueError, match="may not declare a projection readback"):
        load_lane_projection_readback(_overlay(tmp_path, body), lane)


@pytest.mark.parametrize(
    "pasted",
    [
        "postgresql://chain_canary_reader:secret@db.invalid:5436/omnibase_infra",
        "postgres://u:p@h/db",
        "host=db.invalid dbname=omnibase_infra password=secret",
    ],
)
def test_a_dsn_value_in_the_declaration_is_refused(tmp_path: Path, pasted: str) -> None:
    """A VALUE where a NAME belongs is a credential in git history."""
    body = f"""
lanes:
  dev:
    broker: "broker.example.internal:19092"
    security_protocol: PLAINTEXT
    projection_readback:
      dsn_env: "{pasted}"
"""
    with pytest.raises(ValueError) as excinfo:
        load_lane_projection_readback(_overlay(tmp_path, body), "dev")

    # The refusal must not reprint the thing it is refusing.
    assert "parses as a connection string" in str(excinfo.value)
    assert "secret" not in str(excinfo.value)
    assert pasted not in str(excinfo.value)


@pytest.mark.parametrize("bad", ["lower_case_name", "9LEADING_DIGIT", "HAS-DASH"])
def test_a_dsn_env_that_is_not_a_shell_name_is_refused(
    tmp_path: Path, bad: str
) -> None:
    """A name the shell cannot export is a readback that silently never runs."""
    body = f"""
lanes:
  dev:
    broker: "broker.example.internal:19092"
    security_protocol: PLAINTEXT
    projection_readback:
      dsn_env: "{bad}"
"""
    with pytest.raises(ValueError, match="not a POSIX environment variable name"):
        load_lane_projection_readback(_overlay(tmp_path, body), "dev")


@pytest.mark.parametrize(
    "argv",
    [
        ["onex", "--projection-dsn", _DSN],
        [f"--projection-dsn={_DSN}"],
        ["onex", "skill", _DSN],
    ],
    ids=["separate-flag", "joined-flag", "positional"],
)
def test_a_dsn_in_argv_is_detected_and_never_echoed(argv: list[str]) -> None:
    offenders = dsn_shaped_argv_flags(argv)

    assert offenders, f"a DSN in {argv[0]!r}-style argv must be detected"
    # The detector names the FLAG, never the value.
    assert all(_DSN not in offender for offender in offenders)


def test_a_dsn_passed_as_the_name_field_is_refused_by_the_model() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ModelChainCanaryRequest(
            correlation_id=uuid4(), probe_url=_PROBE_URL, projection_dsn_env=_DSN
        )

    assert "takes the NAME" in str(excinfo.value)


def test_there_is_no_field_that_accepts_a_dsn_value() -> None:
    """The removed field cannot come back by accident.

    ``extra="forbid"`` turns the old spelling into a hard error rather than a
    silently ignored kwarg, which is what makes this assertion worth making.
    """
    assert "projection_dsn" not in ModelChainCanaryRequest.model_fields
    with pytest.raises(ValidationError):
        ModelChainCanaryRequest(
            correlation_id=uuid4(),
            probe_url=_PROBE_URL,
            projection_dsn=_DSN,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# 3. The privileged-role refusal, at the transport
# ---------------------------------------------------------------------------


class _FakeConnection:
    def __init__(
        self, *, role_row: dict[str, object] | None, row: dict[str, object] | None
    ) -> None:
        self._role_row = role_row
        self._row = row
        self.state_queries = 0

    async def fetchrow(self, query: str, *args: object) -> object:
        if "pg_roles" in query:
            return self._role_row
        self.state_queries += 1
        return self._row

    async def close(self) -> None:
        return None


class _FakeAsyncpg:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def connect(self, dsn: str) -> _FakeConnection:
        return self._connection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role_row", "expected_fragment"),
    [
        (
            {"rolname": "postgres", "rolsuper": True, "rolbypassrls": True},
            "rolsuper, rolbypassrls",
        ),
        (
            {"rolname": "omninodeadmin", "rolsuper": True, "rolbypassrls": False},
            "rolsuper",
        ),
        (
            {"rolname": "omn15919_rls_writer", "rolsuper": False, "rolbypassrls": True},
            "rolbypassrls",
        ),
    ],
    ids=["superuser-and-bypassrls", "superuser", "bypassrls"],
)
async def test_a_privileged_dsn_is_refused_before_it_reads_anything(
    monkeypatch: pytest.MonkeyPatch,
    role_row: dict[str, object],
    expected_fragment: str,
) -> None:
    """The refusal happens BEFORE the row read, not after.

    Discovering the privilege afterwards would still have exercised the read
    with the wrong identity, and on a lane with RLS it would have read rows the
    role was never supposed to see.
    """
    connection = _FakeConnection(role_row=role_row, row={"state": "COMPLETED"})
    monkeypatch.setitem(sys.modules, "asyncpg", _FakeAsyncpg(connection))

    outcome = await _readback_projection_via_asyncpg(_DSN, str(uuid4()), 5.0)

    assert outcome.status is EnumProjectionReadbackStatus.REFUSED
    assert expected_fragment in outcome.error
    assert connection.state_queries == 0
    assert outcome.state == ""


@pytest.mark.asyncio
async def test_a_least_privilege_reader_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control for the refusal above.

    Without it, "REFUSED on every input" would pass all three cases above and
    the probe would be a check that can never succeed.
    """
    connection = _FakeConnection(
        role_row={
            "rolname": "chain_canary_reader",
            "rolsuper": False,
            "rolbypassrls": False,
        },
        row={"state": "COMPLETED"},
    )
    monkeypatch.setitem(sys.modules, "asyncpg", _FakeAsyncpg(connection))

    outcome = await _readback_projection_via_asyncpg(_DSN, str(uuid4()), 5.0)

    assert outcome.status is EnumProjectionReadbackStatus.TERMINAL
    assert outcome.state == "COMPLETED"
    assert connection.state_queries == 1


@pytest.mark.asyncio
async def test_an_unresolvable_identity_is_refused_not_assumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No row for ``current_user`` fails closed.

    The alternative -- treating an unanswerable identity probe as
    least-privilege -- is the shape of every gate that reports green because it
    could not run.
    """
    connection = _FakeConnection(role_row=None, row={"state": "COMPLETED"})
    monkeypatch.setitem(sys.modules, "asyncpg", _FakeAsyncpg(connection))

    outcome = await _readback_projection_via_asyncpg(_DSN, str(uuid4()), 5.0)

    assert outcome.status is EnumProjectionReadbackStatus.REFUSED
    assert "could not be established" in outcome.error
    assert connection.state_queries == 0


# ---------------------------------------------------------------------------
# 4. The handler: the DSN is resolved from env, under the declared NAME
# ---------------------------------------------------------------------------


class _Ingress:
    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        return {"ok": True}, "", 42


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


class _TerminalReadback:
    async def __call__(
        self,
        bootstrap: str,
        topics: tuple[str, ...],
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[str | None, int, str]:
        return _SUCCESS_TOPIC, 120, ""


class _ProjectionStore:
    """A fake projection keyed by correlation id, plus what it was handed.

    Records the DSN it received so the test can assert the handler resolved it
    out of the ENVIRONMENT rather than out of the request.
    """

    def __init__(self, rows: dict[str, str]) -> None:
        self._rows = rows
        self.seen_dsns: list[str] = []
        self.seen_correlation_ids: list[str] = []

    async def __call__(
        self, dsn: str, correlation_id: str, timeout_s: float
    ) -> ModelProjectionReadbackOutcome:
        self.seen_dsns.append(dsn)
        self.seen_correlation_ids.append(correlation_id)
        state = self._rows.get(correlation_id, "")
        if not state:
            return ModelProjectionReadbackOutcome(
                status=EnumProjectionReadbackStatus.ROW_ABSENT
            )
        return ModelProjectionReadbackOutcome(
            status=EnumProjectionReadbackStatus.TERMINAL, state=state
        )


class _AnswersEverything(_ProjectionStore):
    """A projection that holds a terminal row for whatever it is asked.

    This is the POSITIVE CONTROL for the row-absent case below. Without it, a
    readback that was silently never wired would produce the same red as a
    genuinely missing row, and the test would be pinning nothing.
    """

    async def __call__(
        self, dsn: str, correlation_id: str, timeout_s: float
    ) -> ModelProjectionReadbackOutcome:
        self.seen_dsns.append(dsn)
        self.seen_correlation_ids.append(correlation_id)
        return ModelProjectionReadbackOutcome(
            status=EnumProjectionReadbackStatus.TERMINAL, state="COMPLETED"
        )


def _handler(projection: _ProjectionStore) -> HandlerChainCanary:
    return HandlerChainCanary(
        ingress=_Ingress(),
        quarantine_scan=_Quarantine(),
        terminal_readback=_TerminalReadback(),
        projection_readback=projection,
        kill_switch_disabled=False,
    )


def _request(**overrides: object) -> ModelChainCanaryRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": _PROBE_URL,
        "budget_ms": 5_000,
        "terminal_bootstrap_servers": _BOOTSTRAP,
        "quarantine_bootstrap_servers": _BOOTSTRAP,
        "projection_dsn_env": _DSN_ENV,
        "settle_seconds": 0,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest(**fields)  # type: ignore[arg-type]


def _link(result: ModelChainCanaryResult, link: EnumChainLink) -> EnumChainLinkStatus:
    for verdict in result.link_verdicts:
        if verdict.link is link:
            return verdict.status
    raise AssertionError(f"receipt carries no verdict for {link}")


def _detail(result: ModelChainCanaryResult, link: EnumChainLink) -> str:
    for verdict in result.link_verdicts:
        if verdict.link is link:
            return verdict.detail
    raise AssertionError(f"receipt carries no verdict for {link}")


@pytest.fixture(autouse=True)
def _clean_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """pytest's own argv is test paths; pin it so no invocation leaks in."""
    monkeypatch.setattr(sys, "argv", ["pytest"])


@pytest.mark.asyncio
async def test_the_dsn_comes_from_the_environment_under_the_declared_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_DSN_ENV, _DSN)
    projection = _AnswersEverything({})

    result = await _handler(projection).handle(_request())

    assert projection.seen_dsns == [_DSN]
    assert _link(result, EnumChainLink.ROUTING_PROJECTED) is EnumChainLinkStatus.PASS


@pytest.mark.asyncio
async def test_an_unset_variable_is_not_configured_and_never_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared name pointing at nothing makes no claim, and says which name.

    This is the OTHER half of the run-34281968883 fix. Closing the gate by
    declaring a name would be worthless if a name that resolves to nothing
    quietly behaved like a configured readback.
    """
    monkeypatch.delenv(_DSN_ENV, raising=False)
    projection = _AnswersEverything({})

    result = await _handler(projection).handle(_request())

    assert projection.seen_dsns == [], "the readback ran without a DSN"
    assert (
        _link(result, EnumChainLink.ROUTING_PROJECTED)
        is EnumChainLinkStatus.NOT_CONFIGURED
    )
    assert _DSN_ENV in _detail(result, EnumChainLink.ROUTING_PROJECTED)
    assert result.verdict is EnumChainCanaryVerdict.PROJECTION_READBACK_NOT_CONFIGURED
    assert result.success is False
    assert result.chain_proof_complete is False


@pytest.mark.asyncio
async def test_a_dsn_on_the_command_line_refuses_the_whole_leg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with everything else correct, a DSN in argv is refused.

    The environment holds the right DSN and the projection would answer. The
    run is red anyway, because the disclosure already happened by the time the
    handler could notice it, and reporting green would reward it.
    """
    monkeypatch.setenv(_DSN_ENV, _DSN)
    monkeypatch.setattr(sys, "argv", ["onex", "skill", "--projection-dsn", _DSN])
    projection = _AnswersEverything({})

    result = await _handler(projection).handle(_request())

    assert projection.seen_dsns == [], "the readback ran despite the refusal"
    assert result.verdict is EnumChainCanaryVerdict.PROJECTION_READBACK_REFUSED
    assert result.success is False
    assert _link(result, EnumChainLink.ROUTING_PROJECTED) is EnumChainLinkStatus.ERROR
    detail = _detail(result, EnumChainLink.ROUTING_PROJECTED)
    assert "--projection-dsn" in detail
    assert _DSN not in detail, "the refusal reprinted the credential"


# ---------------------------------------------------------------------------
# 5. The readback assertion itself, with its positive control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_two_passes_only_for_this_runs_own_correlation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row under somebody else's id is not this run's evidence.

    The projection here holds a perfectly good terminal row -- under a
    FABRICATED correlation id that no run ever used. The probe mints its own
    id internally and asks for that one, finds nothing, and link 2 FAILS.

    The positive control is ``_AnswersEverything``, exercised in the same
    assertion below: the identical wiring, with a store that answers for
    whatever id it is asked, PASSES. Without that half, a readback that was
    never called at all would produce this same red and the test would prove
    nothing about correlation scoping.
    """
    monkeypatch.setenv(_DSN_ENV, _DSN)
    fabricated = str(uuid4())

    scoped = _ProjectionStore({fabricated: "COMPLETED"})
    scoped_result = await _handler(scoped).handle(_request())

    control = _AnswersEverything({})
    control_result = await _handler(control).handle(_request())

    # The store WAS asked -- so the red below is a missing row, not a leg that
    # never ran.
    assert len(scoped.seen_correlation_ids) == 1
    asked_for = scoped.seen_correlation_ids[0]
    assert asked_for != fabricated, "the probe must mint its own correlation id"

    assert _link(scoped_result, EnumChainLink.ROUTING_PROJECTED) is (
        EnumChainLinkStatus.FAIL
    )
    assert scoped_result.verdict is EnumChainCanaryVerdict.PROJECTION_ROW_ABSENT
    assert scoped_result.success is False
    assert "no row" in _detail(scoped_result, EnumChainLink.ROUTING_PROJECTED)

    # Positive control: same wiring, a store that answers, link 2 passes.
    # The scalar verdict is deliberately NOT asserted GREEN here -- this
    # fixture leaves link 5's ledger source unset, so the run is honestly red
    # for a reason that has nothing to do with link 2. Asserting the LINK is
    # the assertion this test is making; asserting GREEN would require
    # configuring a leg the test does not care about and would couple it to
    # OMN-16964's fixture.
    assert _link(control_result, EnumChainLink.ROUTING_PROJECTED) is (
        EnumChainLinkStatus.PASS
    )
    assert control_result.verdict not in (
        EnumChainCanaryVerdict.PROJECTION_ROW_ABSENT,
        EnumChainCanaryVerdict.PROJECTION_STRANDED,
        EnumChainCanaryVerdict.PROJECTION_READBACK_FAILED,
        EnumChainCanaryVerdict.PROJECTION_READBACK_REFUSED,
        EnumChainCanaryVerdict.PROJECTION_READBACK_NOT_CONFIGURED,
    )


@pytest.mark.asyncio
async def test_two_runs_never_share_a_correlation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scoping above is only meaningful if the id is fresh every run."""
    monkeypatch.setenv(_DSN_ENV, _DSN)
    projection = _AnswersEverything({})

    await _handler(projection).handle(_request())
    await _handler(projection).handle(_request())

    assert len(set(projection.seen_correlation_ids)) == 2
