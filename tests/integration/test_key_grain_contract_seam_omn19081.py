# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The declared grain travels contract-to-verdict through the shipped seam.

The unit tests for this change hand the grain to the counters directly. That
proves the resolution order and it does not prove the thing the pre-PR proof
on dogfood-101 had to measure by hand: that a grain written in a
``contract.yaml`` on disk reaches the dimension's verdict through the real
wiring factory, with no step in between supplying it.

Those are different failures. Every accessor can be correct while the wiring
never passes a grain, and that shape ships a silent regression past every
unit test in the sibling module: the exemption would quietly come from the
fallback on every runtime, including ones whose contracts declare it, and
nothing would say so.

So this walks the chain the runtime walks:

    contract.yaml on disk
        -> _read_declared_key_grains / resolve_key_grain
        -> _make_projection_dispatch_callback (the shipped factory)
        -> the process apply counters
        -> evaluate_projection_apply_flow
        -> projection_delta_dropped

Related Tickets:
    - OMN-19081: this change
    - OMN-18908: the contract field it reads
    - OMN-18910: the dimension it feeds
    - OMN-19093: deleting the fallback once every runtime declares
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
    _read_declared_key_grains,
    resolve_key_grain,
)
from omnibase_infra.runtime.health.projection_apply_flow import (
    FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS,
    evaluate_projection_apply_flow,
    projection_delta_dropped_status,
)
from omnibase_infra.runtime.observability.projection_apply_counters import (
    get_projection_apply_counters,
    reset_projection_apply_counters_for_test,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

TOPIC = "onex.evt.omniclaude.task-delegated.v1"  # onex-topic-allow: seam fixture


@pytest.fixture(autouse=True)
def _seam(monkeypatch: pytest.MonkeyPatch) -> object:
    configure_projection_dsns(monkeypatch)
    reset_projection_apply_counters_for_test()
    yield
    reset_projection_apply_counters_for_test()


def _contract(tmp_path: Path, grain: str | None) -> Path:
    """A projection contract on disk, with or without a declared grain."""
    grain_line = f'      key_grain: "{grain}"\n' if grain is not None else ""
    path = tmp_path / "contract.yaml"
    path.write_text(
        "name: node_projection_seam\n"
        "projection_api:\n"
        "  expose: true\n"
        "  exposures:\n"
        "    - topic: onex.snapshot.projection.seam.v1\n"
        f"{grain_line}"
        "      table: seam_rows\n"
    )
    return path


def _drive(handler: object, contract_path: Path) -> None:
    """Wire through the shipped factory, resolving the grain as the runtime does."""
    _make_projection_dispatch_callback(
        handler,
        projection_database_target("delegation_events"),
        (TOPIC,),
        sinks=ProjectionDispatchSinks(
            key_grain=resolve_key_grain(_read_declared_key_grains(contract_path))
        ),
    )


def _rising_verdict(projection: str = "_Writer") -> object:
    """Grade a rising discarded-delta gauge off the PROCESS counters.

    The counters read here are the ones the shipped factory populated, not a
    fresh instance, which is what makes this a seam test rather than another
    unit test of the grader.
    """
    counters = get_projection_apply_counters()
    for total in (5, 40, 400):
        counters.record_drop_total(projection, TOPIC, total)
        counters.close_window()
    return evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
        immutable_grain_projections=counters.immutable_grain_projections(),
        grain_unresolved_projections=counters.grain_unresolved_projections(),
        fallback_immutable_projections=FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS,
    )


class _Writer:
    """A projection whose class name is NOT in the fallback list.

    Deliberately unknown to this repository, so an exemption it receives can
    only have come from the contract.
    """

    topics = [TOPIC]

    def handle(self, input_data: dict[str, object]) -> dict[str, int]:
        return {"rows_upserted": 1}


@pytest.mark.integration
def test_a_declared_immutable_grain_reaches_the_verdict_through_the_seam(
    tmp_path: Path,
) -> None:
    """Contract to verdict, with nothing in between supplying the grain."""
    handler = _Writer()
    _drive(handler, _contract(tmp_path, "immutable"))

    verdict = _rising_verdict()

    assert verdict.excluded_immutable_grain == ("_Writer",)
    assert verdict.fallback_exempted_projections == ()
    assert projection_delta_dropped_status(verdict) == "HEALTHY"


@pytest.mark.integration
def test_negative_control_a_declared_mutable_grain_still_grades(
    tmp_path: Path,
) -> None:
    """Same seam, same series, opposite declaration, opposite verdict.

    Without this the test above cannot be told apart from a seam that exempts
    whatever it is handed.
    """
    handler = _Writer()
    _drive(handler, _contract(tmp_path, "mutable"))

    verdict = _rising_verdict()

    assert verdict.drop_accumulating_projections == ("_Writer",)
    assert verdict.excluded_immutable_grain == ()
    assert projection_delta_dropped_status(verdict) == "DEGRADED"


@pytest.mark.integration
def test_an_undeclared_grain_is_graded_and_named_rather_than_exempted(
    tmp_path: Path,
) -> None:
    """The version window, for a projection the fallback does not cover.

    This is the case the pre-PR proof measured on a runtime predating the
    declaration: no grain resolves, the fallback does not name this handler,
    so it is graded and reported rather than silently exempted.
    """
    handler = _Writer()
    _drive(handler, _contract(tmp_path, None))

    verdict = _rising_verdict()

    assert verdict.grain_unresolved_projections == ("_Writer",)
    assert verdict.fallback_exempted_projections == ()
    assert projection_delta_dropped_status(verdict) == "DEGRADED"


@pytest.mark.integration
def test_an_absent_contract_never_stops_a_projection_being_wired(
    tmp_path: Path,
) -> None:
    """An unreadable contract must not be able to break the dispatch path.

    The reader this change adds deliberately never raises, unlike the DLQ
    reader it copies: that one guards a destination, this one feeds an
    observability exemption.
    """
    handler = _Writer()
    _drive(handler, tmp_path / "does-not-exist.yaml")

    counters = get_projection_apply_counters()
    assert counters.registered_projections() == ("_Writer",)
    assert counters.grain_unresolved_projections() == ("_Writer",)
