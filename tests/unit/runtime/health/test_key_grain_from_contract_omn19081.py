# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The drop dimension reads the declared grain instead of a literal list.

OMN-18910 shipped the exemption as a four-entry literal naming the two
content-addressed exposures, and said in its own comment that the literal was
interim. OMN-18908 has since landed ``key_grain`` on every projection
exposure's contract, so the fact is declared and the copy in this repository
is now a second, unsynchronised source of it.

The failure that copy produces is quiet in both directions. A genuinely
content-addressed exposure added tomorrow gets no exemption unless somebody
remembers to edit a tuple in a different repository, so its intended
idempotence reads as loss. An exposure wrongly added to the tuple gets a
permanent exemption with no evidence behind it, which is the rubber stamp the
declaration exists to prevent.

Reading the contract is not a layering violation and does not import
omnimarket: the wiring already reads ``event_bus.dlq_topics`` out of the same
raw contract YAML through ``_read_dlq_topics``, for the same reason, and this
follows that established seam.

Related Tickets:
    - OMN-19081: this change
    - OMN-18908: the declaration it reads
    - OMN-18910: the literal it replaces
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.health import projection_apply_flow
from omnibase_infra.runtime.health.projection_apply_flow import (
    evaluate_projection_apply_flow,
    projection_delta_dropped_status,
)
from omnibase_infra.runtime.observability.projection_apply_counters import (
    ProjectionApplyCounters,
)

pytestmark = pytest.mark.unit

TOPIC = "onex.evt.omniclaude.task-delegated.v1"  # onex-topic-allow: fixture


def _contract(tmp_path: Path, *grains: str | None) -> Path:
    """One contract declaring an exposure per grain. None omits the field."""
    lines = [
        "name: node_projection_probe",
        "projection_api:",
        "  expose: true",
        "  exposures:",
    ]
    for index, grain in enumerate(grains):
        lines.append(f"    - topic: onex.snapshot.projection.probe-{index}.v1")
        if grain is not None:
            lines.append(f'      key_grain: "{grain}"')
        lines.append("      table: probe_rows")
    path = tmp_path / "contract.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# AC1 -- the literal is gone and the declaration is what is read
# --------------------------------------------------------------------------


def test_ac1_the_unconditional_exemption_list_is_gone() -> None:
    """The literal that applied unconditionally no longer exists.

    REVISED after the pre-PR proof failed and the operator ruled, 2026-09-21.
    The first version of this test asserted no literal at all, and that shape
    of the change was measured on dogfood-101 to turn both content-addressed
    exposures DEGRADED on a runtime carrying an omnimarket predating the
    declaration. What is required is not the absence of a literal but the
    absence of an UNCONDITIONAL one: the surviving list is reached only where
    the contract resolves nothing, and its name says so.
    """
    assert not hasattr(projection_apply_flow, "IMMUTABLE_GRAIN_PROJECTIONS")
    assert projection_apply_flow.FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS


def test_ac1_an_immutable_declaration_is_read_from_the_contract(
    tmp_path: Path,
) -> None:
    """A grain the contract declares reaches the wiring, with no edit here."""
    path = _contract(tmp_path, "immutable")
    assert handler_wiring._read_declared_key_grains(path) == ("immutable",)


def test_ac1_a_new_immutable_exposure_is_exempt_without_editing_this_repo(
    tmp_path: Path,
) -> None:
    """The whole point: a name this repository has never heard of."""
    counters = ProjectionApplyCounters()
    counters.register("HandlerNeverSeenBefore", TOPIC, key_grain="immutable")
    assert counters.immutable_grain_projections() == ("HandlerNeverSeenBefore",)


# --------------------------------------------------------------------------
# AC2 -- the two live exemptions are behaviourally unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "projection",
    ["HandlerProjectionSessionReplay", "HandlerProjectionWorkEvents"],
)
def test_ac2_the_live_exemptions_stay_exempt_when_declared(projection: str) -> None:
    counters = ProjectionApplyCounters()
    counters.register(projection, TOPIC, key_grain="immutable")
    assert counters.immutable_grain_projections() == (projection,)


def test_ac2_negative_control_a_mutable_declaration_is_not_exempt() -> None:
    """Without this, an exemption set that returned everything would pass."""
    counters = ProjectionApplyCounters()
    counters.register("HandlerProjectionRunnerFleet", TOPIC, key_grain="mutable")
    assert counters.immutable_grain_projections() == ()


def test_ac2_a_declared_exemption_is_still_excluded_from_the_drop_grade() -> None:
    """End to end through the verdict, not just the accessor."""
    counters = ProjectionApplyCounters()
    counters.register("HandlerProjectionWorkEvents", TOPIC, key_grain="immutable")
    for total in (5, 40, 400):
        counters.record_drop_total("HandlerProjectionWorkEvents", TOPIC, total)
        counters.close_window()
    verdict = evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
        immutable_grain_projections=counters.immutable_grain_projections(),
    )
    assert verdict.drop_accumulating_projections == ()
    assert projection_delta_dropped_status(verdict) == "HEALTHY"


def test_ac2_the_same_accumulation_on_a_mutable_grain_does_grade() -> None:
    """The control that proves the exclusion above is doing the work."""
    counters = ProjectionApplyCounters()
    counters.register("HandlerProjectionRunnerFleet", TOPIC, key_grain="mutable")
    for total in (5, 40, 400):
        counters.record_drop_total("HandlerProjectionRunnerFleet", TOPIC, total)
        counters.close_window()
    verdict = evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
        immutable_grain_projections=counters.immutable_grain_projections(),
    )
    assert verdict.drop_accumulating_projections == ("HandlerProjectionRunnerFleet",)
    assert projection_delta_dropped_status(verdict) == "DEGRADED"


# --------------------------------------------------------------------------
# AC3 -- an unresolvable grain is a declared outcome, never a default
# --------------------------------------------------------------------------


def test_ac3_an_undeclared_grain_is_not_silently_exempt() -> None:
    """Silently exempt would HIDE a real accumulation behind a missing field."""
    counters = ProjectionApplyCounters()
    counters.register("HandlerProjectionUnknown", TOPIC, key_grain=None)
    assert counters.immutable_grain_projections() == ()


def test_ac3_an_undeclared_grain_is_named_rather_than_passed_over() -> None:
    """Graded is the right default; graded SILENTLY is not.

    A reader has to be able to tell "this projection declares a mutable grain"
    from "this dimension could not resolve what grain it has", because the
    remedies differ and only one of them is a contract edit.
    """
    counters = ProjectionApplyCounters()
    counters.register("HandlerProjectionUnknown", TOPIC, key_grain=None)
    assert counters.grain_unresolved_projections() == ("HandlerProjectionUnknown",)


def test_ac3_the_unresolved_outcome_reaches_the_rendered_detail() -> None:
    counters = ProjectionApplyCounters()
    counters.register("HandlerProjectionUnknown", TOPIC, key_grain=None)
    counters.close_window()
    verdict = evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
        immutable_grain_projections=counters.immutable_grain_projections(),
        grain_unresolved_projections=counters.grain_unresolved_projections(),
    )
    detail = projection_apply_flow.describe_projection_delta_dropped(verdict)
    assert projection_apply_flow.OUTCOME_KEY_GRAIN_UNRESOLVED in detail
    assert "HandlerProjectionUnknown" in detail


def test_ac3_negative_control_a_resolved_grain_raises_no_unresolved_token() -> None:
    counters = ProjectionApplyCounters()
    counters.register("HandlerProjectionRunnerFleet", TOPIC, key_grain="mutable")
    counters.close_window()
    verdict = evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
        immutable_grain_projections=counters.immutable_grain_projections(),
        grain_unresolved_projections=counters.grain_unresolved_projections(),
    )
    detail = projection_apply_flow.describe_projection_delta_dropped(verdict)
    assert projection_apply_flow.OUTCOME_KEY_GRAIN_UNRESOLVED not in detail


# --------------------------------------------------------------------------
# Reading the contract: the shapes, and the failures
# --------------------------------------------------------------------------


def test_both_contract_shapes_are_read(tmp_path: Path) -> None:
    """The tree carries an inline exposure and an exposures list."""
    legacy = tmp_path / "contract.yaml"
    legacy.write_text(
        "projection_api:\n"
        "  expose: true\n"
        "  topic: onex.snapshot.projection.legacy.v1\n"
        '  key_grain: "immutable"\n'
    )
    assert handler_wiring._read_declared_key_grains(legacy) == ("immutable",)


def test_a_mixed_contract_is_not_exempt(tmp_path: Path) -> None:
    """One mutable exposure is enough to grade the projection.

    A handler whose contract declares both cannot be exempted on the strength
    of the immutable half: the drop it discards might belong to either.
    """
    path = _contract(tmp_path, "immutable", "mutable")
    grains = handler_wiring._read_declared_key_grains(path)
    assert set(grains) == {"immutable", "mutable"}
    counters = ProjectionApplyCounters()
    counters.register(
        "HandlerMixed", TOPIC, key_grain=handler_wiring.resolve_key_grain(grains)
    )
    assert counters.immutable_grain_projections() == ()


def test_an_absent_contract_file_resolves_to_unknown(tmp_path: Path) -> None:
    """Never a raise on the wiring path, and never a silent 'immutable'."""
    assert handler_wiring._read_declared_key_grains(tmp_path / "nope.yaml") == ()
    assert handler_wiring.resolve_key_grain(()) is None


def test_an_unexposed_contract_declares_no_grain(tmp_path: Path) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text('projection_api:\n  expose: false\n  key_grain: "immutable"\n')
    assert handler_wiring._read_declared_key_grains(path) == ()


# --------------------------------------------------------------------------
# The seam itself -- without this the accessors could be right and the
# wiring still never pass a grain, which is the shape that ships a silent
# regression past every test above
# --------------------------------------------------------------------------


def test_the_wiring_factory_registers_the_grain_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the shipped factory, not a stand-in for it."""
    from omnibase_infra.runtime.observability.projection_apply_counters import (
        get_projection_apply_counters,
        reset_projection_apply_counters_for_test,
    )
    from tests.helpers.application_db_topology import (
        configure_projection_dsns,
        projection_database_target,
    )

    configure_projection_dsns(monkeypatch)

    class _Probe:
        topics = [TOPIC]

        def handle(self, input_data: dict[str, object]) -> dict[str, int]:
            return {"rows_upserted": 1}

    reset_projection_apply_counters_for_test()
    try:
        handler_wiring._make_projection_dispatch_callback(
            _Probe(),
            projection_database_target("delegation_events"),
            (TOPIC,),
            sinks=handler_wiring.ProjectionDispatchSinks(key_grain="immutable"),
        )
        assert get_projection_apply_counters().immutable_grain_projections() == (
            "_Probe",
        )
    finally:
        reset_projection_apply_counters_for_test()


def test_negative_control_the_factory_given_no_grain_exempts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: a grain absent at the seam must not become an exemption."""
    from omnibase_infra.runtime.observability.projection_apply_counters import (
        get_projection_apply_counters,
        reset_projection_apply_counters_for_test,
    )
    from tests.helpers.application_db_topology import (
        configure_projection_dsns,
        projection_database_target,
    )

    configure_projection_dsns(monkeypatch)

    class _Probe:
        topics = [TOPIC]

        def handle(self, input_data: dict[str, object]) -> dict[str, int]:
            return {"rows_upserted": 1}

    reset_projection_apply_counters_for_test()
    try:
        handler_wiring._make_projection_dispatch_callback(
            _Probe(),
            projection_database_target("delegation_events"),
            (TOPIC,),
        )
        counters = get_projection_apply_counters()
        assert counters.immutable_grain_projections() == ()
        assert counters.grain_unresolved_projections() == ("_Probe",)
    finally:
        reset_projection_apply_counters_for_test()


# --------------------------------------------------------------------------
# The fallback, and its resolution order. Operator ruling 2026-09-21, option
# two, after the pre-PR proof on dogfood-101 measured the literal-free shape
# turning healthy behaviour DEGRADED on a pre-declaration runtime.
# --------------------------------------------------------------------------


def _rising_gauge(projection: str, grain: str | None) -> ProjectionApplyCounters:
    """A rising discarded-delta gauge, which is idempotence on an immutable grain."""
    counters = ProjectionApplyCounters()
    counters.register(projection, TOPIC, key_grain=grain)
    for total in (5, 40, 400):
        counters.record_drop_total(projection, TOPIC, total)
        counters.close_window()
    return counters


def _verdict(counters: ProjectionApplyCounters) -> object:
    return evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
        immutable_grain_projections=counters.immutable_grain_projections(),
        grain_unresolved_projections=counters.grain_unresolved_projections(),
        fallback_immutable_projections=(
            projection_apply_flow.FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS
        ),
    )


def test_the_0_4_178_replay_reads_healthy() -> None:
    """The exact case the proof failed on, now the regression test.

    A runtime whose omnimarket predates key_grain resolves no grain, so the
    fallback carries the exemption and the intended idempotence of a
    content-addressed exposure does not alarm.
    """
    verdict = _verdict(_rising_gauge("HandlerProjectionWorkEvents", None))
    assert verdict.drop_accumulating_projections == ()
    assert projection_delta_dropped_status(verdict) == "HEALTHY"
    assert verdict.fallback_exempted_projections == ("HandlerProjectionWorkEvents",)


def test_the_contract_wins_over_the_fallback() -> None:
    """POSITIVE CONTROL for the resolution order, and the point of the change.

    A projection named in the fallback whose contract declares it MUTABLE is
    GRADED. The literal can never override a live declaration, which is what
    keeps it a window-cover rather than a second opinion.
    """
    verdict = _verdict(_rising_gauge("HandlerProjectionWorkEvents", "mutable"))
    assert verdict.drop_accumulating_projections == ("HandlerProjectionWorkEvents",)
    assert verdict.fallback_exempted_projections == ()
    assert projection_delta_dropped_status(verdict) == "DEGRADED"


def test_a_contract_declared_immutable_is_not_credited_to_the_fallback() -> None:
    """A name the fallback has never heard of, exempted by its declaration."""
    verdict = _verdict(_rising_gauge("HandlerNeverSeenBefore", "immutable"))
    assert verdict.drop_accumulating_projections == ()
    assert verdict.fallback_exempted_projections == ()
    assert verdict.excluded_immutable_grain == ("HandlerNeverSeenBefore",)


def test_negative_control_absent_from_both_is_graded() -> None:
    """Unresolved and not in the fallback: graded, never exempted.

    Without this the fallback could be widened to everything and every test
    above would still pass.
    """
    verdict = _verdict(_rising_gauge("HandlerProjectionRunnerFleet", None))
    assert verdict.drop_accumulating_projections == ("HandlerProjectionRunnerFleet",)
    assert verdict.fallback_exempted_projections == ()
    assert projection_delta_dropped_status(verdict) == "DEGRADED"


def test_an_unresolved_grain_is_still_reported_even_when_the_fallback_exempts() -> None:
    """The token is a REPORT, not an exemption, per the ruling.

    A reader deciding whether the literal can be deleted has to be able to see
    which exposures still depend on it.
    """
    verdict = _verdict(_rising_gauge("HandlerProjectionWorkEvents", None))
    assert verdict.grain_unresolved_projections == ("HandlerProjectionWorkEvents",)
    detail = projection_apply_flow.describe_projection_delta_dropped(verdict)
    assert projection_apply_flow.OUTCOME_KEY_GRAIN_UNRESOLVED in detail
    assert "INTERIM fallback" in detail
