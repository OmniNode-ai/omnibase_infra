# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every refusal reaches the rejection topic, not just two of eight (OMN-17079).

WHAT WAS MEASURED, AND WHY IT MATTERS NOW
-----------------------------------------
``EnumRejectionReason`` has EIGHT members. Before this change exactly TWO of them ever
reached ``onex.evt.deploy.rebuild-rejected.v1``:

* ``IN_PROGRESS`` — ``agent.py`` publishes it when a second command arrives mid-deploy.
* ``SUPERSEDED`` — added by OMN-18143 AC7 when coalescing folds a command away.

The other six are resolved in ``consumer._process_message``, which commits the offset,
logs a line and returns ``(None, "<reason>")``. ``agent.py``'s loop then logs
``"Rejected command: %s"`` and nothing else. No event, ever:

* ``UNDECODABLE_PAYLOAD`` — the value deserializer handed back a marker
* ``INVALID_SIGNATURE`` — HMAC verification failed
* ``INVALID_PAYLOAD`` — a signed command the contract refuses
* ``LANE_NOT_ALLOWED`` — the lane fence refused it
* ``BUSY`` — a job is already active
* ``DUPLICATE`` — the correlation id was already seen

This is LD-11, filed 2026-08-30 and still true. It matters now because the topic is
about to have a reader: the Lab observability errors widget. **A widget wired to a
topic that carries two of eight refusal reasons would look quiet while six kinds of
refusal happened, and a quiet errors widget is read as "nothing was refused".** That is
the exact class of invisibility the observability work exists to remove, so shipping the
reader without this would install a new one.

WHAT THIS MODULE REFUSES TO DO, AND WHY IT IS NOT AN OMISSION
--------------------------------------------------------------
Three of the six are resolved BEFORE a valid command exists, so there is no guaranteed
correlation id and no guaranteed scope to put on the wire. ``ModelRebuildRejected``
requires both, and a rejection carrying a fabricated correlation id is worse than no
rejection: it is a durable, queryable record pointing at a command that never existed.

So the rule these tests pin is: **publish every reason whose identifiers the agent can
actually prove, and on the ones it cannot, record a named non-publish rather than invent
an id.** ``BUSY``, ``DUPLICATE`` and ``LANE_NOT_ALLOWED`` always have a validated command
and therefore always publish. ``INVALID_SIGNATURE``, ``INVALID_PAYLOAD`` and
``UNDECODABLE_PAYLOAD`` publish when the raw record still yields both identifiers and
are recorded as unattributable when it does not — the quarantine record, which already
exists on those paths, stays the durable evidence either way.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from deploy_agent.events import (
    EnumRejectionReason,
    ModelRejectionNotice,
    Scope,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. The notice model: what the consumer resolved, with nothing invented.
# ---------------------------------------------------------------------------


def test_the_notice_carries_the_reason_and_whatever_identifiers_resolved() -> None:
    cid = uuid4()
    notice = ModelRejectionNotice(
        reason=EnumRejectionReason.BUSY,
        correlation_id=cid,
        scope=Scope.FULL,
    )
    assert notice.reason is EnumRejectionReason.BUSY
    assert notice.correlation_id == cid
    assert notice.scope is Scope.FULL


def test_the_notice_can_say_it_resolved_nothing() -> None:
    """An undecodable record yields no identifiers, and that must be expressible.

    ``None`` here is a measured absence. The alternative — omitting the field, or
    defaulting it to a placeholder id — is what turns "we could not attribute this" into
    a false claim about a command.
    """
    notice = ModelRejectionNotice(
        reason=EnumRejectionReason.UNDECODABLE_PAYLOAD,
        correlation_id=None,
        scope=None,
    )
    assert notice.correlation_id is None
    assert notice.scope is None


def test_the_notice_defaults_nothing() -> None:
    """Every field is supplied by the site that did the resolving."""
    for name, field in ModelRejectionNotice.model_fields.items():
        assert field.is_required(), (
            f"{name} carries a default; the consumer site that resolved the rejection "
            "is the only thing that knows this value, and a default here invents one"
        )


# ---------------------------------------------------------------------------
# 2. The publish decision, on the agent's single helper.
# ---------------------------------------------------------------------------


class _CapturingAgent:
    """The agent's publish helper, replaced so the decision is observable."""

    def __init__(self) -> None:
        self.published: list[Any] = []

    def capture(self, event: Any) -> bool:
        self.published.append(event)
        return True


def _notice(
    reason: EnumRejectionReason,
    *,
    correlation_id: UUID | None,
    scope: Scope | None,
) -> ModelRejectionNotice:
    return ModelRejectionNotice(
        reason=reason, correlation_id=correlation_id, scope=scope
    )


@pytest.mark.parametrize(
    "reason",
    [
        EnumRejectionReason.BUSY,
        EnumRejectionReason.DUPLICATE,
        EnumRejectionReason.LANE_NOT_ALLOWED,
        EnumRejectionReason.INVALID_SIGNATURE,
        EnumRejectionReason.INVALID_PAYLOAD,
        EnumRejectionReason.UNDECODABLE_PAYLOAD,
    ],
)
def test_a_fully_identified_rejection_publishes_for_every_reason(
    reason: EnumRejectionReason,
) -> None:
    """All six silent reasons publish once their identifiers are known.

    Parametrized over the exact set that published nothing before this change, so a
    reason quietly dropped from the routing fails here by name.
    """
    from deploy_agent.agent import DeployAgent

    cap = _CapturingAgent()
    cid = uuid4()

    DeployAgent.publish_rejection_notice(  # type: ignore[call-arg]
        _notice(reason, correlation_id=cid, scope=Scope.RUNTIME),
        publish=cap.capture,
    )

    assert len(cap.published) == 1, (
        f"{reason.value} produced {len(cap.published)} events; every identified "
        "refusal must reach the topic exactly once"
    )
    event = cap.published[0]
    assert event.reason is reason
    assert event.correlation_id == cid
    assert event.scope is Scope.RUNTIME


@pytest.mark.parametrize(
    ("correlation_id", "scope"),
    [
        (None, Scope.FULL),
        (uuid4(), None),
        (None, None),
    ],
)
def test_an_unattributable_rejection_publishes_nothing_rather_than_inventing_one(
    correlation_id: UUID | None, scope: Scope | None
) -> None:
    """A missing identifier is a non-publish, never a placeholder.

    A rejection carrying a fabricated correlation id is worse than no rejection: it is a
    durable record pointing at a command that never existed, and a reader cannot tell it
    from a real one.
    """
    from deploy_agent.agent import DeployAgent

    cap = _CapturingAgent()

    DeployAgent.publish_rejection_notice(  # type: ignore[call-arg]
        _notice(
            EnumRejectionReason.UNDECODABLE_PAYLOAD,
            correlation_id=correlation_id,
            scope=scope,
        ),
        publish=cap.capture,
    )

    assert cap.published == [], (
        "an unattributable rejection was published; the quarantine record is the "
        "durable evidence on that path, and the event must be withheld"
    )


def test_the_supersession_reason_is_not_routed_through_this_path() -> None:
    """SUPERSEDED carries two extra required fields and keeps its own builder.

    Routing it here would drop ``superseded_by_sha`` and
    ``superseded_by_correlation_id``, which ``ModelRebuildRejected`` refuses for that
    reason — so this must fail loudly rather than publish a malformed supersession.
    """
    from deploy_agent.agent import DeployAgent

    cap = _CapturingAgent()

    with pytest.raises(ValueError, match="superseded"):
        DeployAgent.publish_rejection_notice(  # type: ignore[call-arg]
            _notice(
                EnumRejectionReason.SUPERSEDED,
                correlation_id=uuid4(),
                scope=Scope.FULL,
            ),
            publish=cap.capture,
        )
    assert cap.published == []


# ---------------------------------------------------------------------------
# 3. The consumer really fires the hook, at every site that refuses a command.
# ---------------------------------------------------------------------------


def test_the_consumer_exposes_an_on_rejected_hook() -> None:
    """Mirrors the existing ``on_superseded`` injection rather than inventing a path."""
    import inspect

    from deploy_agent.consumer import DeployConsumer

    params = inspect.signature(DeployConsumer.__init__).parameters
    assert "on_rejected" in params, (
        "the consumer takes no on_rejected hook, so the six reasons it resolves have no "
        "route to the agent's publish helper"
    )


def test_every_reason_the_consumer_resolves_reaches_the_hook() -> None:
    """Source-level: each refusal site routes through ``_reject``, naming its reason.

    Parsed rather than exercised because each site sits behind its own broker, HMAC or
    job-store precondition; what this pins is WHICH reasons are routed, so a refusal
    added later as a bare ``return None, "reason"`` fails here by name. The behavioural
    half is covered by the publish tests above.

    Its own positive control is the parse itself: zero routed reasons means the AST walk
    is not seeing the method, and every verdict below it would be vacuous.
    """
    import ast
    import inspect
    import textwrap

    from deploy_agent.consumer import DeployConsumer

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(DeployConsumer._process_message))
    )

    routed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_reject"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "EnumRejectionReason"
        ):
            routed.add(node.args[0].attr)

    assert routed, (
        "positive control failed: no _reject call naming an EnumRejectionReason member "
        "was parsed out of _process_message, so this test cannot see the refusal sites "
        "and any verdict it gives would be vacuous"
    )

    expected = {
        "UNDECODABLE_PAYLOAD",
        "INVALID_SIGNATURE",
        "INVALID_PAYLOAD",
        "LANE_NOT_ALLOWED",
        "BUSY",
        "DUPLICATE",
    }
    assert routed == expected, (
        f"_process_message routes {sorted(routed)} through the rejection hook; "
        f"expected exactly {sorted(expected)}. A reason that only returns its token "
        "publishes nothing, which is the LD-11 defect this closes."
    )


def test_no_refusal_site_still_returns_a_bare_reason_string() -> None:
    """The shape guard: a tuple-return of a bare string is the pre-fix defect.

    ``_process_message`` returns ``(None, <token>)`` on every refusal, and after this
    change the token always comes from ``_reject``. A literal string in that slot is a
    site that committed the offset and published nothing.
    """
    import ast
    import inspect
    import textwrap

    from deploy_agent.consumer import DeployConsumer

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(DeployConsumer._process_message))
    )

    bare: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    bare.append(elt.value)

    assert not bare, (
        f"these refusal sites still return a bare reason string: {sorted(bare)}. "
        "Route them through self._reject so they reach the rejection topic."
    )
