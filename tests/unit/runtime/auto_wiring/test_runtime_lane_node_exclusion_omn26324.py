# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26324: dev's main profile wires only the contracts dev runs, declared as data.

``contracts/services/runtime_lane_node_exclusion.contract.yaml`` names the
contracts the dev profile does not attach and the contracts it needs; a runtime
whose declared lane belongs to that profile reads them. The ownership filter reads it through
``ModelRuntimeNodeSelection.from_env``. Every name asserted here is read from
that file or from ``runtime_policy.contract.yaml``, never spelled in the test.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.runtime.model_runtime_lane_node_exclusion import (
    LANE_PROFILE_PREFIX,
    RUNTIME_LANE_NODE_EXCLUSION_CONTRACT,
    ModelRuntimeLaneNodeExclusion,
)
from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_EXCLUSION,
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

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACTS_DIR = REPO_ROOT / "contracts"
DEV_PROFILE = "dev"
DEV_LANE = f"{LANE_PROFILE_PREFIX}{DEV_PROFILE}"
assert DEV_LANE in REGISTERED_RUNTIME_LANES
UNRELATED = "node_unrelated_to_the_dev_profile_omn26324"


def _dev_environ(contracts_dir: Path = CONTRACTS_DIR) -> dict[str, str]:
    return {"ONEX_RUNTIME_LANE": DEV_LANE, "ONEX_CONTRACTS_DIR": str(contracts_dir)}


def _declaration(contracts_dir: Path = CONTRACTS_DIR) -> ModelRuntimeLaneNodeExclusion:
    return ModelRuntimeLaneNodeExclusion.load(
        ModelRuntimeLaneNodeExclusion.declaration_path(contracts_dir)
    )


def _excluded(contracts_dir: Path = CONTRACTS_DIR) -> tuple[str, ...]:
    return _declaration(contracts_dir).entry_for(DEV_LANE).excluded_contracts


def _required(contracts_dir: Path = CONTRACTS_DIR) -> tuple[str, ...]:
    """What dev needs: the declaration's list and dev main's boot priority list."""
    policy = yaml.safe_load(
        (CONTRACTS_DIR / "services" / "runtime_policy.contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    priority = policy["profiles"]["dev"]["services"]["main"]["boot"][
        "priority_contracts"
    ]
    declared = _declaration(contracts_dir).entry_for(DEV_LANE).required_contracts
    return tuple(dict.fromkeys((*declared, *priority)))


def _contract(name: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
    )


def _manifest(*name_groups: tuple[str, ...]) -> ModelAutoWiringManifest:
    names = dict.fromkeys(name for group in name_groups for name in group)
    return ModelAutoWiringManifest(contracts=tuple(_contract(n) for n in names))


def _wired(manifest: ModelAutoWiringManifest, environ: dict[str, str]) -> set[str]:
    result = filter_manifest_for_runtime_profile(manifest, "main", environ=environ)
    return {contract.name for contract in result.manifest.contracts}


def _dropped_required(contracts_dir: Path) -> set[str]:
    """Required contracts the dev lane would not wire from this declaration."""
    required = _required(contracts_dir)
    manifest = _manifest(required, _excluded(contracts_dir), (UNRELATED,))
    return set(required) - _wired(manifest, _dev_environ(contracts_dir))


def _mutated_contracts_dir(tmp_path: Path, extra_exclusion: str) -> Path:
    """A copy of the declaration whose dev exclusion also names one more contract."""
    declared = REPO_ROOT / "contracts" / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT
    target = tmp_path / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT
    target.parent.mkdir(parents=True)
    shutil.copy(declared, target)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    data["profiles"][DEV_PROFILE]["excluded_contracts"].append(extra_exclusion)
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def test_dev_profile_contract_set_declares_a_trim() -> None:
    assert _excluded(), "the dev lane declares no excluded contract"


def test_dev_profile_contract_set_leaves_declared_contracts_unattached() -> None:
    excluded = _excluded()
    manifest = _manifest(excluded, (UNRELATED,))

    result = filter_manifest_for_runtime_profile(
        manifest, "main", environ=_dev_environ()
    )

    assert {c.name for c in result.manifest.contracts} == {UNRELATED}
    assert result.node_selection_excluded_contracts == excluded
    assert result.manifest.errors == ()


def test_dev_profile_contract_set_reaches_the_raw_ownership_path() -> None:
    """The legacy subscription loops read the same list as the manifest filter."""
    name = _excluded()[0]

    assert not runtime_profile_owns_contract(
        {"name": name}, "main", environ=_dev_environ()
    )
    assert runtime_profile_owns_contract(
        {"name": UNRELATED}, "main", environ=_dev_environ()
    )


@pytest.mark.parametrize("lane", sorted(REGISTERED_RUNTIME_LANES - {DEV_LANE}) + [""])
def test_dev_profile_contract_set_leaves_every_other_lane_unchanged(lane: str) -> None:
    excluded = _excluded()
    environ = {"ONEX_RUNTIME_LANE": lane, "ONEX_CONTRACTS_DIR": str(CONTRACTS_DIR)}

    assert _wired(_manifest(excluded, (UNRELATED,)), environ) == {*excluded, UNRELATED}
    assert ModelRuntimeNodeSelection.from_env(environ).excluded == ()


def test_dev_profile_contract_set_absent_declaration_keeps_todays_behaviour(
    tmp_path: Path,
) -> None:
    excluded = _excluded()
    manifest = _manifest(excluded, (UNRELATED,))

    for environ in (
        {"ONEX_RUNTIME_LANE": DEV_LANE},
        {"ONEX_RUNTIME_LANE": DEV_LANE, "ONEX_CONTRACTS_DIR": ""},
        {"ONEX_RUNTIME_LANE": DEV_LANE, "ONEX_CONTRACTS_DIR": str(tmp_path)},
    ):
        assert _wired(manifest, environ) == {*excluded, UNRELATED}


def test_dev_profile_contract_set_malformed_declaration_refuses(
    tmp_path: Path,
) -> None:
    target = tmp_path / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT
    target.parent.mkdir(parents=True)
    target.write_text("name: runtime_lane_node_exclusion\nversion: 0\n", "utf-8")

    with pytest.raises(ValueError, match="version"):
        ModelRuntimeNodeSelection.from_env(_dev_environ(tmp_path))


def test_dev_profile_contract_set_composes_with_the_environment_exclusion() -> None:
    declared = _excluded()
    environ = {
        **_dev_environ(),
        ENV_RUNTIME_NODE_EXCLUSION: f"{declared[0]},{UNRELATED}",
    }

    selection = ModelRuntimeNodeSelection.from_env(environ)

    assert selection.excluded == (declared[0], UNRELATED)
    assert selection.declared_excluded == declared[1:]
    assert not selection.admits(UNRELATED)
    assert not selection.admits(declared[-1])


def test_dev_profile_contract_set_stale_declared_name_is_a_discovery_error() -> None:
    """A declared name no discovered contract has is reported, and never stops startup."""
    declared = _excluded()
    manifest = _manifest(declared[1:], (UNRELATED,))

    result = filter_manifest_for_runtime_profile(
        manifest, "main", environ=_dev_environ()
    )

    assert {c.name for c in result.manifest.contracts} == {UNRELATED}
    assert [e.entry_point_name for e in result.manifest.errors] == [declared[0]]


def test_dev_profile_contract_set_unknown_environment_name_still_refuses() -> None:
    """The environment's own exclusion keeps its refusal (OMN-19513)."""
    environ = {**_dev_environ(), ENV_RUNTIME_NODE_EXCLUSION: UNRELATED}

    with pytest.raises(RuntimeHostError, match=UNRELATED):
        filter_manifest_for_runtime_profile(
            _manifest(_excluded()), "main", environ=environ
        )


def test_dev_profile_contract_set_adds_no_contract_name_list_under_src() -> None:
    """The declared names live in the YAML; no .py file under src spells a list of them.

    A lone name can be an unrelated literal (an enum value that shares a contract's
    name); two or more declared names in one file is a name list.
    """
    declared = _excluded()
    offenders: dict[str, list[str]] = {}
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        spelled = [
            name
            for name in declared
            if re.search(rf"""["']{re.escape(name)}["']""", text)
        ]
        if len(spelled) >= 2:
            offenders[str(path.relative_to(REPO_ROOT))] = spelled

    assert offenders == {}


def test_dev_required_contracts_wired() -> None:
    required = _required()
    assert required, "no required contract is declared for the dev lane"
    assert not set(required) & set(_excluded())

    assert _dropped_required(CONTRACTS_DIR) == set()


def test_dev_required_contracts_wired_fails_when_one_is_excluded(
    tmp_path: Path,
) -> None:
    """Control: adding one required contract to the exclusion is caught here, not on dev."""
    victim = _required()[0]

    dropped = _dropped_required(_mutated_contracts_dir(tmp_path, victim))

    assert dropped == {victim}
