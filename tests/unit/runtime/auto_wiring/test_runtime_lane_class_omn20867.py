# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20867: a lane instance takes only the class contracts its deployment names.

A runtime whose declared lane is ``<class>-<instance>`` may borrow named
contracts scoped to ``<class>`` when its deployment sets
``ONEX_RUNTIME_LANE_CLASS`` and ``ONEX_RUNTIME_LANE_CLASS_CONTRACTS``. Every
selection that cannot be honoured is a discovery error and borrows nothing.

No lane is spelled here: the class, its instance and the other lanes are picked
from core's registered lanes, so the names stay where the deployment keeps them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.constants.constants_runtime_lanes import (
    LAB_RUNTIME_LANES,
    REGISTERED_RUNTIME_LANES,
)
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    ModelRuntimeProfileOwnershipResult,
    runtime_profile_owns_contract,
)
from omnibase_infra.runtime.health.runtime_lane_identity import (
    ENV_RUNTIME_LANE,
    ENV_RUNTIME_LANE_CLASS,
    ENV_RUNTIME_LANE_CLASS_CONTRACTS,
    resolve_runtime_lane,
)

pytestmark = pytest.mark.unit


def _class_and_instance() -> tuple[str, str]:
    """A lab lane class and a registered instance lane of it."""
    for lane_class in sorted(LAB_RUNTIME_LANES):
        instances = sorted(
            lane
            for lane in REGISTERED_RUNTIME_LANES
            if lane.startswith(f"{lane_class}-")
        )
        if instances:
            return lane_class, instances[0]
    raise AssertionError("core registers no instance of any lab lane")


CLASS, INSTANCE = _class_and_instance()
# Another lab lane, not of the class, and a registered lane that is no instance.
OTHER = sorted(LAB_RUNTIME_LANES - {CLASS} - {INSTANCE})[0]
NOT_AN_INSTANCE = sorted(
    lane
    for lane in REGISTERED_RUNTIME_LANES - LAB_RUNTIME_LANES
    if not lane.startswith(f"{CLASS}-")
)[0]

SELECTED = "node_selected"
UNSELECTED = "node_unselected"
OTHER_SCOPE = "node_other_scope"
UNSCOPED = "node_unscoped"


def _contract(name: str, lanes: tuple[str, ...] | None) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        runtime_lanes=ModelRuntimeLaneScope(lanes=lanes) if lanes else None,
    )


MANIFEST = ModelAutoWiringManifest(
    contracts=(
        _contract(SELECTED, (CLASS,)),
        _contract(UNSELECTED, (CLASS, OTHER)),
        _contract(OTHER_SCOPE, (OTHER,)),
        _contract(UNSCOPED, None),
    )
)


def _filter(env: dict[str, str]) -> ModelRuntimeProfileOwnershipResult:
    return filter_manifest_for_runtime_profile(MANIFEST, "main", environ=env)


def _owned(result: ModelRuntimeProfileOwnershipResult) -> set[str]:
    return {contract.name for contract in result.manifest.contracts}


def _class_errors(result: ModelRuntimeProfileOwnershipResult) -> list[str]:
    return [e.error for e in result.manifest.errors if "OMN-20867" in e.error]


def _env(
    lane: str = INSTANCE, lane_class: str = CLASS, contracts: str = SELECTED
) -> dict[str, str]:
    return {
        ENV_RUNTIME_LANE: lane,
        ENV_RUNTIME_LANE_CLASS: lane_class,
        ENV_RUNTIME_LANE_CLASS_CONTRACTS: contracts,
    }


def test_no_selection_leaves_an_instance_excluded_as_before() -> None:
    result = _filter({ENV_RUNTIME_LANE: INSTANCE})
    assert _owned(result) == {UNSCOPED}
    assert result.runtime_lane_class is None
    assert result.lane_class_admitted_contracts == ()
    assert set(result.lane_excluded_contracts) == {SELECTED, UNSELECTED, OTHER_SCOPE}
    assert not result.manifest.errors


def test_the_instance_takes_only_the_named_class_contract() -> None:
    result = _filter(_env())
    assert _owned(result) == {SELECTED, UNSCOPED}
    assert result.runtime_lane == INSTANCE
    assert result.runtime_lane_class == CLASS
    assert result.lane_class_admitted_contracts == (SELECTED,)
    # A class contract the deployment did not name stays off the instance.
    assert set(result.lane_excluded_contracts) == {UNSELECTED, OTHER_SCOPE}
    assert not result.manifest.errors


def test_the_class_lane_itself_keeps_its_own_scope() -> None:
    result = _filter(_env(lane=CLASS))
    # The class is not an instance of itself: the selection is refused, and
    # its own lane still attaches every contract scoped to it.
    assert _owned(result) == {SELECTED, UNSELECTED, UNSCOPED}
    assert result.lane_class_admitted_contracts == ()
    assert len(_class_errors(result)) == 1


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        (
            {ENV_RUNTIME_LANE: INSTANCE, ENV_RUNTIME_LANE_CLASS: CLASS},
            "names no contracts",
        ),
        (
            {ENV_RUNTIME_LANE: INSTANCE, ENV_RUNTIME_LANE_CLASS_CONTRACTS: SELECTED},
            "is not",
        ),
        (_env(lane_class=f"{CLASS}x"), "not a registered lane"),
        (_env(lane=NOT_AN_INSTANCE), "not an instance of lane class"),
        (_env(lane=f"{CLASS}-unregistered"), "declares no registered"),
    ],
    ids=[
        "no-contracts",
        "no-class",
        "unregistered-class",
        "not-an-instance",
        "unregistered-lane",
    ],
)
def test_a_selection_that_cannot_be_honoured_is_an_error_and_borrows_nothing(
    env: dict[str, str], fragment: str
) -> None:
    result = _filter(env)
    assert SELECTED not in _owned(result)
    assert result.runtime_lane_class is None
    assert result.lane_class_admitted_contracts == ()
    errors = _class_errors(result)
    assert len(errors) == 1
    assert fragment in errors[0]


def test_a_named_contract_nobody_discovered_is_an_error() -> None:
    result = _filter(_env(contracts=f"{SELECTED},node_typo"))
    assert result.lane_class_admitted_contracts == (SELECTED,)
    errors = _class_errors(result)
    assert len(errors) == 1
    assert "'node_typo'" in errors[0]


def test_a_named_contract_whose_scope_excludes_the_class_is_an_error() -> None:
    result = _filter(_env(contracts=f"{SELECTED},{OTHER_SCOPE}"))
    assert OTHER_SCOPE not in _owned(result)
    assert OTHER_SCOPE not in result.lane_excluded_contracts
    errors = _class_errors(result)
    assert len(errors) == 1
    assert OTHER_SCOPE in errors[0]


def test_the_raw_contract_path_admits_the_same_selection() -> None:
    selected = {"name": SELECTED, "runtime_lanes": [CLASS]}
    unselected = {"name": UNSELECTED, "runtime_lanes": [CLASS]}
    assert runtime_profile_owns_contract(selected, "main", environ=_env())
    assert not runtime_profile_owns_contract(unselected, "main", environ=_env())
    assert not runtime_profile_owns_contract(
        selected, "main", environ={ENV_RUNTIME_LANE: INSTANCE}
    )
    assert not runtime_profile_owns_contract(
        selected, "main", environ=_env(lane=NOT_AN_INSTANCE)
    )


def test_a_class_never_keys_the_instance_health() -> None:
    # Placement only: an instance does not speak for its class's lane health.
    assert resolve_runtime_lane(_env()) is None
