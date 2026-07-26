# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14830 — HandlerRunnerFleetHealthEvaluate is REACHABLE through real dispatch.

RED-against-EXISTS-but-WRONG proof for the canonical def-B flip (hand-flip, OMN-14781).

Before this ticket ``HandlerRunnerFleetHealthEvaluate.handle`` had the multi-positional
signature ``handle(self, correlation_id, snapshot)``. The canonical-shape ratchet
(``scripts/ci/canonical_handler_shape.py``) classifies that ``nonadaptable`` and the
shared runtime adapter's ``_make_dispatch_callback`` binds ``handle`` and invokes it with
the SINGLE validated ``ModelRunnerFleetHealthEvaluateCommand`` payload — so the
two-argument entrypoint raises ``TypeError: handle() missing 1 required positional
argument: 'snapshot'`` on the FIRST dispatch. The node passed ingress and then died on
every real command.

These tests drive the REAL production dispatch callback over the REAL handler class (no
fake handler, no patched entrypoint): they fail against the multi-positional handler and
pass only once the def-B ``handle(self, request)`` entrypoint exists. The flip is a pure
signature adaptation that preserves the classification body verbatim; the business-logic
helpers (``_classify_runner`` / ``_annotate_indeterminate`` / ``_recommend_for_assessment``)
are byte-identical base_ref<->HEAD, which the canonical-shape ratchet re-derives from git
(the ``.handflip.json`` proof). Equivalence to the direct ``handle`` call is asserted per
case here (dispatch produces the same verdict the pre-flip code would have).

The SELECTED input corpus below (by ``input_hash`` of the command) is the exact set bound
into both the adequacy receipt
(``omnibase_infra.nodes.node_runner_fleet_health_compute.json``) and the hand-flip proof
(``...handflip.json``) under ``scripts/ci/adequacy_receipts/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_runner_fleet_health_compute.handlers.handler_runner_fleet_health_evaluate import (
    HandlerRunnerFleetHealthEvaluate,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_health_state import (
    EnumRunnerFleetHealthState,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_evaluate_command import (
    ModelRunnerFleetHealthEvaluateCommand,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_verdict import (
    ModelRunnerFleetHealthVerdict,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef

pytestmark = [pytest.mark.unit]

# The contract-declared payload_type_match event model the runtime binds.
_EVENT_MODEL = ModelHandlerRef(
    name="ModelRunnerFleetHealthEvaluateCommand",
    module=(
        "omnibase_infra.nodes.node_runner_fleet_health_compute.models."
        "model_runner_fleet_health_evaluate_command"
    ),
)

# Deterministic fixed timestamp + correlation so the SELECTED corpus hashes reproducibly.
_FIXED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_FIXED_CID = "00000000-0000-4000-8000-000000000000"


def _fact(name: str, **overrides: object) -> ModelRunnerFleetRunnerFact:
    defaults: dict[str, object] = {
        "name": name,
        "github_status": "online",
        "github_busy": False,
    }
    defaults.update(overrides)
    return ModelRunnerFleetRunnerFact(**defaults)  # type: ignore[arg-type]


def _command(**overrides: object) -> ModelRunnerFleetHealthEvaluateCommand:
    snap_defaults: dict[str, object] = {
        "correlation_id": _FIXED_CID,
        "collected_at": _FIXED_AT,
        "host": "192.168.86.201",
        "expected_count": 3,
        "runners": (),
    }
    snap_defaults.update(overrides)
    snapshot = ModelRunnerFleetSnapshot(**snap_defaults)  # type: ignore[arg-type]
    return ModelRunnerFleetHealthEvaluateCommand(
        correlation_id=snapshot.correlation_id, snapshot=snapshot
    )


# The SELECTED corpus: id -> command exercising a distinct classification branch.
_CASES: list[tuple[str, ModelRunnerFleetHealthEvaluateCommand]] = [
    ("healthy", _command(runners=(_fact("r1"), _fact("r2")))),
    (
        "saturated",
        _command(
            runners=(
                _fact("r1", github_busy=True),
                _fact("r2", github_busy=True),
            )
        ),
    ),
    (
        "crashloop_and_zombie",
        _command(
            runners=(
                _fact("r1", docker_restart_count=9),
                _fact("r2", diag_heartbeat_age_seconds=1200.0),
                _fact("r3"),
            )
        ),
    ),
    (
        "wedged",
        _command(
            runners=(_fact("r1"), _fact("r2")),
            oldest_queued_job_age_seconds=900.0,
        ),
    ),
]
_CASE_IDS = [c[0] for c in _CASES]


def test_handler_handle_is_single_request_adaptable() -> None:
    """The def-B invariant: ``handle`` exposes exactly one non-self positional param.

    RED against the pre-OMN-14830 ``handle(self, correlation_id, snapshot)``: the
    multi-positional signature is what ``_handle_is_adaptable`` rejects and what makes
    the real dispatch callback pass one arg to a two-arg method.
    """
    import inspect

    sig = inspect.signature(HandlerRunnerFleetHealthEvaluate.handle)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["request"], (
        f"def-B requires a single ``request`` param; got {params!r}. A multi-positional "
        "handle() is nonadaptable and breaks the shared runtime dispatch adapter."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "command"), _CASES, ids=_CASE_IDS)
async def test_real_dispatch_callback_returns_verdict(
    case_id: str, command: ModelRunnerFleetHealthEvaluateCommand
) -> None:
    """LOAD-BEARING: a real command dispatched through the REAL auto-wiring callback
    reaches ``handle`` and yields the SAME verdict a direct call produces.

    Against the multi-positional handler this raises ``TypeError`` (one arg passed to a
    two-arg method) instead of returning a result — that raise IS the bug this flip
    closes.
    """
    handler = HandlerRunnerFleetHealthEvaluate()
    callback = _make_dispatch_callback(handler, _EVENT_MODEL)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=command,
        correlation_id=uuid4(),
        event_type="ModelRunnerFleetHealthEvaluateCommand",
    )

    result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — the handler never ran."
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"Expected SUCCESS dispatch status, got {result.status!r}."
    )
    assert len(result.output_events) == 1, (
        f"Expected exactly one ModelRunnerFleetHealthVerdict, got {result.output_events!r}"
    )
    dispatched = result.output_events[0]
    assert isinstance(dispatched, ModelRunnerFleetHealthVerdict)

    # Behavior equivalence: the dispatch-produced verdict matches a direct handle() call.
    direct = await handler.handle(command)
    assert [a.state for a in dispatched.assessments] == [
        a.state for a in direct.assessments
    ]
    assert dispatched.online_count == direct.online_count
    assert dispatched.saturation_ratio == direct.saturation_ratio
    assert [a.action_type for a in dispatched.recommended_actions] == [
        a.action_type for a in direct.recommended_actions
    ]


@pytest.mark.asyncio
async def test_multipositional_handle_breaks_dispatch_is_the_red() -> None:
    """Documents the exact RED the flip closes.

    A handler whose ``handle`` keeps the legacy two-argument signature is invoked by the
    REAL callback with a single validated payload and raises ``TypeError`` — reproduced
    here directly so the missing-adaptability defect can never silently regress.
    """

    class _LegacyTwoArgHandler:
        async def handle(
            self,
            correlation_id: object,
            snapshot: object,
        ) -> ModelRunnerFleetHealthVerdict:
            raise AssertionError("unreachable: dispatch passes only one positional arg")

    callback = _make_dispatch_callback(_LegacyTwoArgHandler(), _EVENT_MODEL)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=_command(runners=(_fact("r1"),)),
        correlation_id=uuid4(),
        event_type="ModelRunnerFleetHealthEvaluateCommand",
    )
    with pytest.raises(TypeError, match="positional argument"):
        await callback(envelope)


@pytest.mark.asyncio
async def test_dispatched_verdict_classifies_saturated_incident() -> None:
    """End-to-end sanity: the 2026-07-04 zero-idle incident classifies SATURATED
    through the real dispatch path (not just a direct unit call)."""
    handler = HandlerRunnerFleetHealthEvaluate()
    callback = _make_dispatch_callback(handler, _EVENT_MODEL)
    command = _command(
        runners=(
            _fact("r1", github_busy=True),
            _fact("r2", github_busy=True),
            _fact("r3", github_busy=True),
        )
    )
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=command,
        correlation_id=uuid4(),
        event_type="ModelRunnerFleetHealthEvaluateCommand",
    )
    result = await callback(envelope)
    assert result is not None and result.status is EnumDispatchStatus.SUCCESS
    verdict = result.output_events[0]
    assert isinstance(verdict, ModelRunnerFleetHealthVerdict)
    assert verdict.saturation_ratio == 1.0
    assert all(
        a.state == EnumRunnerFleetHealthState.SATURATED for a in verdict.assessments
    )
