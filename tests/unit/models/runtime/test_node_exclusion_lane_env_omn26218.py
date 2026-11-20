# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane's node exclusion is typed and checked against the known nodes (OMN-26232).

``ONEX_RUNTIME_NODE_EXCLUSION`` is the runtime's switch for leaving a node out.
The runtime refuses an unknown name only at container boot; this model refuses it
where the deploy agent reads the lane env. Every package and node name below is
synthetic; one test reads this repository's own checkout as the real positive
control.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.handler_package_announcer import read_package_inventory
from omnibase_infra.models.runtime.model_node_exclusion_lane_env import (
    ENV_RUNTIME_NODE_EXCLUSION,
    ModelNodeExclusionLaneEnv,
    contract_roots,
    known_contract_names,
    validate_node_exclusion,
)
from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ModelRuntimeNodeSelection,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
KNOWN = frozenset({"alpha_projection", "beta_effect"})


def _package(root: Path, entries: dict[str, tuple[str, str | None]]) -> Path:
    """A synthetic checkout: entry point name -> (module, contract text or None)."""
    root.mkdir(parents=True)
    lines = ['[project.entry-points."onex.nodes"]']
    for entry, (module, contract) in entries.items():
        lines.append(f'{entry} = "{module}"')
        if contract is not None:
            directory = root / "src" / Path(*module.split("."))
            directory.mkdir(parents=True)
            (directory / "contract.yaml").write_text(contract, encoding="utf-8")
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.mark.unit
def test_parse_splits_a_comma_list_in_order_without_blanks_or_duplicates() -> None:
    parsed = ModelNodeExclusionLaneEnv.parse(
        " beta_effect, ,alpha_projection,beta_effect,"
    )
    assert parsed.names == ("beta_effect", "alpha_projection")


@pytest.mark.unit
@pytest.mark.parametrize("raw", [None, "", "   ", " , ,"], ids=repr)
def test_unset_and_blank_name_nothing_and_are_accepted(raw: str | None) -> None:
    parsed = ModelNodeExclusionLaneEnv.parse(raw)
    assert parsed.names == ()
    assert parsed.require_known(KNOWN) is parsed
    assert parsed.require_known(frozenset()) is parsed


@pytest.mark.unit
def test_known_names_are_accepted() -> None:
    parsed = ModelNodeExclusionLaneEnv.parse("alpha_projection,beta_effect")
    assert parsed.require_known(KNOWN) is parsed


@pytest.mark.unit
def test_a_mistyped_name_is_refused_with_the_name_and_a_close_match() -> None:
    parsed = ModelNodeExclusionLaneEnv.parse("alpha_projection,beta_efect")
    with pytest.raises(ValueError) as excinfo:
        parsed.require_known(KNOWN)
    message = str(excinfo.value)
    assert "'beta_efect'" in message
    assert "did you mean 'beta_effect'" in message
    assert "'alpha_projection'" not in message
    assert ENV_RUNTIME_NODE_EXCLUSION in message


@pytest.mark.unit
def test_every_unknown_name_is_named() -> None:
    parsed = ModelNodeExclusionLaneEnv.parse("zzz_one,alpha_projection,zzz_two")
    with pytest.raises(ValueError) as excinfo:
        parsed.require_known(KNOWN)
    assert "'zzz_one'" in str(excinfo.value)
    assert "'zzz_two'" in str(excinfo.value)


@pytest.mark.unit
def test_the_model_is_strict_frozen_and_forbids_extras() -> None:
    config = ModelNodeExclusionLaneEnv.model_config
    assert config["frozen"] is True
    assert config["extra"] == "forbid"
    assert config["strict"] is True
    with pytest.raises(ValidationError):
        ModelNodeExclusionLaneEnv.model_validate({"names": "alpha_projection"})
    with pytest.raises(ValidationError):
        ModelNodeExclusionLaneEnv.model_validate({"names": (), "other": 1})


@pytest.mark.unit
def test_known_names_come_from_entry_points_and_contract_names(tmp_path: Path) -> None:
    root = _package(
        tmp_path / "pkg",
        {
            "node_named": ("pkg.nodes.node_named", "name: declared_name\nversion: 1\n"),
            "node_quoted": ("pkg.nodes.node_quoted", 'name: "quoted_name"  # c\n'),
            "node_unnamed": ("pkg.nodes.node_unnamed", "version: 1\n"),
            "node_nofile": ("pkg.nodes.node_nofile", None),
        },
    )
    assert known_contract_names([root]) == {
        "declared_name",
        "quoted_name",
        "node_unnamed",
    }


@pytest.mark.unit
def test_a_root_without_a_readable_pyproject_contributes_nothing(
    tmp_path: Path,
) -> None:
    assert known_contract_names([tmp_path / "absent", tmp_path]) == frozenset()


@pytest.mark.unit
def test_roots_are_the_deploy_source_and_the_omnimarket_sibling(tmp_path: Path) -> None:
    infra = _package(tmp_path / "infra", {})
    market = _package(tmp_path / "home" / "omnimarket", {})
    roots = contract_roots(
        {"DEPLOY_AGENT_REPO_DIR": str(infra), "OMNI_HOME": str(tmp_path / "home")},
        checkout_root=tmp_path / "checkout",
    )
    assert roots == (infra, market)
    assert contract_roots({}, checkout_root=tmp_path / "checkout") == ()


@pytest.mark.unit
def test_validation_reads_the_roots_the_environment_names(tmp_path: Path) -> None:
    infra = _package(
        tmp_path / "infra", {"n": ("p.nodes.n", "name: alpha_projection\n")}
    )
    market = _package(
        tmp_path / "home" / "omnimarket", {"m": ("q.nodes.m", "name: beta_effect\n")}
    )
    environ = {"DEPLOY_AGENT_REPO_DIR": str(infra), "OMNI_HOME": str(tmp_path / "home")}
    validate_node_exclusion("alpha_projection,beta_effect", environ, market)
    validate_node_exclusion("", environ, market)
    with pytest.raises(ValueError, match="alpha_projectoin"):
        validate_node_exclusion("alpha_projectoin", environ, market)


@pytest.mark.unit
def test_a_name_cannot_be_checked_against_no_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be checked"):
        validate_node_exclusion("alpha_projection", {}, tmp_path)
    validate_node_exclusion("", {}, tmp_path)  # nothing named, nothing to check


@pytest.mark.unit
def test_static_known_names_are_the_names_the_runtime_discovers() -> None:
    # The runtime's own inventory pass over this repository's package is the
    # control: the static read must find the same contract names, or the deploy
    # agent would refuse a name the runtime accepts, or accept one it refuses.
    inventory = read_package_inventory(frozenset({"omnibase_infra"}))
    discovered = {r.contract_name for r in inventory.records if r.contract_name}
    assert discovered, "positive control: the runtime found no contract"
    assert known_contract_names([REPO_ROOT]) == discovered


@pytest.mark.unit
def test_the_runtime_selection_reads_the_same_list_the_model_parses() -> None:
    raw = " beta_effect,alpha_projection,beta_effect"
    selection = ModelRuntimeNodeSelection.from_env({ENV_RUNTIME_NODE_EXCLUSION: raw})
    assert selection.excluded == ModelNodeExclusionLaneEnv.parse(raw).names
    assert ENV_RUNTIME_NODE_EXCLUSION == "ONEX_RUNTIME_NODE_EXCLUSION"
