# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19513: a deployment can hand a node to a dedicated container.

``ONEX_RUNTIME_NODE_EXCLUSION`` names contracts this process must not attach.
The dev lane sets it on the main runtime for ``projection_claude_hook_events``
and sets ``ONEX_RUNTIME_NODE_SELECTION`` to that one name on the dedicated
writer, so the node has exactly one consumer-group owner and a restart of the
main runtime (which joins its consumer groups one at a time) no longer leaves
the projection unsubscribed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_EXCLUSION,
    ENV_RUNTIME_NODE_SELECTION,
    ModelRuntimeNodeSelection,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)

pytestmark = pytest.mark.unit

HANDED_OFF = "node_handed_off_omn19513"
KEPT = "node_kept_omn19513"


def _contract(name: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
    )


MANIFEST = ModelAutoWiringManifest(contracts=(_contract(HANDED_OFF), _contract(KEPT)))


def _names(result_contracts: tuple[ModelDiscoveredContract, ...]) -> set[str]:
    return {contract.name for contract in result_contracts}


def test_unset_exclusion_attaches_every_owned_contract() -> None:
    result = filter_manifest_for_runtime_profile(MANIFEST, "main", environ={})

    assert _names(result.manifest.contracts) == {HANDED_OFF, KEPT}
    assert result.node_selection_excluded_contracts == ()


def test_exclusion_leaves_the_named_contract_unattached() -> None:
    environ = {ENV_RUNTIME_NODE_EXCLUSION: HANDED_OFF}

    result = filter_manifest_for_runtime_profile(MANIFEST, "main", environ=environ)

    assert _names(result.manifest.contracts) == {KEPT}
    assert result.node_selection_excluded_contracts == (HANDED_OFF,)
    assert result.manifest.errors == ()


def test_the_writer_and_the_main_runtime_never_both_own_the_node() -> None:
    """Selection on the writer and exclusion on main partition the node."""
    main = filter_manifest_for_runtime_profile(
        MANIFEST, "main", environ={ENV_RUNTIME_NODE_EXCLUSION: HANDED_OFF}
    )
    writer = filter_manifest_for_runtime_profile(
        MANIFEST, "main", environ={ENV_RUNTIME_NODE_SELECTION: HANDED_OFF}
    )

    assert HANDED_OFF not in _names(main.manifest.contracts)
    assert _names(writer.manifest.contracts) == {HANDED_OFF}
    assert main.manifest.errors == ()
    assert writer.manifest.errors == ()


def test_exclusion_wins_over_selection() -> None:
    environ = {
        ENV_RUNTIME_NODE_SELECTION: f"{HANDED_OFF},{KEPT}",
        ENV_RUNTIME_NODE_EXCLUSION: HANDED_OFF,
    }

    result = filter_manifest_for_runtime_profile(MANIFEST, "main", environ=environ)

    assert _names(result.manifest.contracts) == {KEPT}


def test_exclusion_naming_an_undiscovered_contract_is_a_discovery_error() -> None:
    environ = {ENV_RUNTIME_NODE_EXCLUSION: f"{HANDED_OFF},node_missing_omn19513"}

    result = filter_manifest_for_runtime_profile(MANIFEST, "main", environ=environ)

    assert _names(result.manifest.contracts) == {KEPT}
    errors = [error.error for error in result.manifest.errors]
    assert len(errors) == 1
    assert "node_missing_omn19513" in errors[0]
    assert ENV_RUNTIME_NODE_EXCLUSION in errors[0]


def test_raw_ownership_honours_exclusion() -> None:
    environ = {ENV_RUNTIME_NODE_EXCLUSION: HANDED_OFF}

    assert not runtime_profile_owns_contract(
        {"name": HANDED_OFF}, "main", environ=environ
    )
    assert runtime_profile_owns_contract({"name": KEPT}, "main", environ=environ)
    assert runtime_profile_owns_contract({"name": HANDED_OFF}, "main", environ={})


def test_exclusion_model_parses_and_does_not_activate_selection() -> None:
    model = ModelRuntimeNodeSelection.from_env(
        {ENV_RUNTIME_NODE_EXCLUSION: f" {HANDED_OFF} , ,{HANDED_OFF}"}
    )

    assert model.excluded == (HANDED_OFF,)
    # Exclusion is not a selection: topic provisioning keeps every topic.
    assert not model.is_active
    assert not model.admits(HANDED_OFF)
    assert model.admits(KEPT)
