# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render a fold policy and apply it to the runtime's real discovered contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)
from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from scripts.render_runtime_policy_env import render_env

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
# These roles have contracts in this repository. Projection contracts live in
# downstream packages and are not assumed to be installed on a hosted runner.
_ROLES = ("effects", "workers")


@pytest.mark.parametrize("lane", sorted(REGISTERED_RUNTIME_LANES))
@pytest.mark.parametrize("folded", [False, True], ids=["unfolded", "folded"])
def test_rendered_fold_matches_separate_roles_on_real_contracts(
    lane: str, folded: bool
) -> None:
    raw = yaml.safe_load(
        (_ROOT / "contracts/services/runtime_policy.contract.yaml").read_text()
    )
    raw["profiles"]["dev"]["services"]["main"]["folded_profiles"] = (
        list(_ROLES) if folded else []
    )
    rendered = render_env(ModelRuntimePolicyContract.model_validate(raw))
    environment = {"ONEX_RUNTIME_LANE": lane}
    if "DEV_RUNTIME_MAIN_FOLDED_PROFILES" in rendered:
        environment["ONEX_RUNTIME_FOLDED_PROFILES"] = rendered[
            "DEV_RUNTIME_MAIN_FOLDED_PROFILES"
        ]

    paths = sorted((_ROOT / "src/omnibase_infra/nodes").glob("*/contract.yaml"))
    assert paths
    discovered = discover_contracts_from_paths(paths)
    assert not discovered.errors, discovered.errors
    assert all(
        any(role in contract.runtime_profiles for contract in discovered.contracts)
        for role in _ROLES
    ), "Each folded role must have real contracts in this checkout"

    # Independent reference: each role's existing, unfolded ownership path.
    expected: set[str] = set()
    main_names: set[str] = set()
    for role in ("main", *(_ROLES if folded else ())):
        separate = filter_manifest_for_runtime_profile(
            discovered, role, environ={"ONEX_RUNTIME_LANE": lane}
        )
        assert not separate.manifest.errors, separate.manifest.errors
        names = {contract.name for contract in separate.manifest.contracts}
        expected.update(names)
        if role == "main":
            main_names = names

    ownership = filter_manifest_for_runtime_profile(
        discovered, "main", environ=environment
    )
    attached = {contract.name for contract in ownership.manifest.contracts}
    assert not ownership.manifest.errors, ownership.manifest.errors
    assert attached == expected
    assert ownership.folded_profiles == (_ROLES if folded else ())
    assert set(ownership.folded_contracts) == attached - main_names
    if folded:
        assert attached - main_names, "The fold must attach additional nodes"

    # Raw-contract readers and the manifest path must agree on the same scope.
    for contract in discovered.contracts:
        contract_raw = yaml.safe_load(contract.contract_path.read_text())
        assert runtime_profile_owns_contract(
            contract_raw, "main", environ=environment
        ) == (contract.name in attached), contract.name

    print(
        json.dumps(
            {
                "lane": lane,
                "folded": folded,
                "discovered": len(discovered.contracts),
                "attached": len(attached),
                "folded_contracts": len(ownership.folded_contracts),
                "missing": len(expected - attached),
            },
            sort_keys=True,
        )
    )
