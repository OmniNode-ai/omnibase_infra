# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane's overlay can turn a node off, and a typo is refused before any recreate (OMN-26232).

``ONEX_RUNTIME_NODE_EXCLUSION`` is in the ``runtime_lane_env.v1`` allowlist. A
fragment (and the overlay home's ``runtime_lane_env`` overlay, which shares the
fragment parser) that names a node no contract declares is refused with
:class:`LaneEnvFragmentError`, naming the typo, at the moment the agent reads the
lane env, so no container has been recreated yet.

Every lane, package and node name here is synthetic.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.instance_lanes import (
    LANE_ENV_FRAGMENT_DIR,
    LANE_ENV_FRAGMENT_REF,
    LANE_ENV_FRAGMENT_SCHEMA_KEYS,
    LANE_ENV_FRAGMENT_SUFFIX,
    LaneEnvFragmentError,
    parse_lane_env_fragment,
    read_lane_env_fragment,
)

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

NODE_EXCLUSION_KEY = "ONEX_RUNTIME_NODE_EXCLUSION"
SCHEMA = "runtime_lane_env.v1"
LANE = "compose-lane-a"
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_SOURCE = (
    REPO_ROOT / "src/omnibase_infra/models/runtime/model_node_exclusion_lane_env.py"
)


def _fragment(env: Mapping[str, object]) -> str:
    return json.dumps(
        {"schema_version": SCHEMA, "lane_id": LANE, "description": "t", "env": env}
    )


def _checkout(root: Path, entry_points: Mapping[str, str]) -> Path:
    """A synthetic package checkout: entry point name -> contract name."""
    lines = ['[project.entry-points."onex.nodes"]']
    for entry, name in entry_points.items():
        lines.append(f'{entry} = "pkg.nodes.{entry}"')
        directory = root / "src/pkg/nodes" / entry
        directory.mkdir(parents=True)
        (directory / "contract.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def known_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deploy source and the omnimarket sibling the agent reads node names from."""
    infra = _checkout(tmp_path / "infra", {"node_one": "node_alpha"})
    home = tmp_path / "home"
    _checkout(home / "omnimarket", {"node_two": "node_beta_projection"})
    monkeypatch.setenv("DEPLOY_AGENT_REPO_DIR", str(infra))
    monkeypatch.setenv("OMNI_HOME", str(home))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )


def _overlay_clone(tmp_path: Path, env: Mapping[str, object]) -> Path:
    repo = tmp_path / "overlay-clone"
    relpath = f"{LANE_ENV_FRAGMENT_DIR}/{LANE}{LANE_ENV_FRAGMENT_SUFFIX}"
    (repo / Path(relpath).parent).mkdir(parents=True)
    (repo / relpath).write_text(_fragment(env), encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", f"refs/remotes/{LANE_ENV_FRAGMENT_REF}", "HEAD")
    return repo


def _literal(path: Path, name: str) -> str:
    """The string a module assigns to ``name`` at top level, read without importing."""
    for stmt in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == name
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return stmt.value.value
    raise AssertionError(f"{name} not found in {path}")


@pytest.mark.unit
def test_the_allowlist_literal_is_the_name_the_runtime_reads() -> None:
    # A literal, because the agent's environment need not carry omnibase_infra:
    # pinned to the model's constant and to the runtime selection's re-export.
    assert NODE_EXCLUSION_KEY in LANE_ENV_FRAGMENT_SCHEMA_KEYS[SCHEMA]
    assert _literal(MODEL_SOURCE, "ENV_RUNTIME_NODE_EXCLUSION") == NODE_EXCLUSION_KEY
    selection = (
        REPO_ROOT / "src/omnibase_infra/models/runtime/model_runtime_node_selection.py"
    ).read_text(encoding="utf-8")
    assert "ENV_RUNTIME_NODE_EXCLUSION,\n" in selection  # imported, not redefined
    assert 'ENV_RUNTIME_NODE_EXCLUSION = "' not in selection


@pytest.mark.unit
@pytest.mark.usefixtures("known_nodes")
@pytest.mark.parametrize(
    "value",
    ["node_alpha", "node_beta_projection", "node_alpha, node_beta_projection"],
    ids=["deploy-source", "omnimarket-sibling", "both"],
)
def test_a_known_name_is_accepted_and_reaches_the_env(value: str) -> None:
    env = parse_lane_env_fragment(
        _fragment({NODE_EXCLUSION_KEY: value}), lane_id=LANE, source="t"
    )
    assert env == {NODE_EXCLUSION_KEY: value}


@pytest.mark.unit
@pytest.mark.usefixtures("known_nodes")
def test_a_mistyped_name_is_refused_with_the_typo_in_the_message() -> None:
    with pytest.raises(LaneEnvFragmentError) as excinfo:
        parse_lane_env_fragment(
            _fragment({NODE_EXCLUSION_KEY: "node_alpha,node_beta_projction"}),
            lane_id=LANE,
            source="overlay:file",
        )
    message = str(excinfo.value)
    assert "node_beta_projction" in message
    assert "did you mean 'node_beta_projection'" in message
    assert message.startswith("overlay:file: ")
    assert "'node_alpha'" not in message


@pytest.mark.unit
@pytest.mark.usefixtures("known_nodes")
def test_empty_and_unset_are_accepted() -> None:
    assert parse_lane_env_fragment(
        _fragment({NODE_EXCLUSION_KEY: ""}), lane_id=LANE, source="t"
    ) == {NODE_EXCLUSION_KEY: ""}
    assert parse_lane_env_fragment(_fragment({}), lane_id=LANE, source="t") == {}


@pytest.mark.unit
def test_an_empty_value_is_still_refused_for_every_other_key() -> None:
    with pytest.raises(LaneEnvFragmentError, match="ONEX_RUNTIME_LANE_CLASS is empty"):
        parse_lane_env_fragment(
            _fragment({"ONEX_RUNTIME_LANE_CLASS": ""}), lane_id=LANE, source="t"
        )


@pytest.mark.unit
def test_a_name_with_no_checkout_to_check_it_against_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_REPO_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "absent-home"))
    with pytest.raises(LaneEnvFragmentError, match="cannot be checked"):
        parse_lane_env_fragment(
            _fragment({NODE_EXCLUSION_KEY: "node_alpha"}), lane_id=LANE, source="t"
        )


@pytest.mark.unit
@pytest.mark.usefixtures("known_nodes")
def test_the_clone_reader_applies_the_check_before_returning_a_fragment(
    tmp_path: Path,
) -> None:
    good = _overlay_clone(tmp_path / "good", {NODE_EXCLUSION_KEY: "node_alpha"})
    assert read_lane_env_fragment(good, LANE) == {NODE_EXCLUSION_KEY: "node_alpha"}
    bad = _overlay_clone(tmp_path / "bad", {NODE_EXCLUSION_KEY: "node_alpah"})
    with pytest.raises(LaneEnvFragmentError, match="node_alpah"):
        read_lane_env_fragment(bad, LANE)


@pytest.mark.unit
@pytest.mark.usefixtures("known_nodes")
def test_the_compose_env_carries_the_name_and_refuses_a_typo_before_any_recreate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_compose_env`` is every compose command's env, so a refusal here precedes a recreate."""
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(REPO_ROOT))
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(tmp_path))
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.delenv(NODE_EXCLUSION_KEY, raising=False)
    monkeypatch.setattr(
        executor_mod,
        "DEV_INSTANCE_RUNTIME_LANES",
        {executor_mod.active_dev_instance(): LANE},
    )
    good = _overlay_clone(
        tmp_path / "good", {NODE_EXCLUSION_KEY: "node_beta_projection"}
    )
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", good)
    assert executor_mod._compose_env()[NODE_EXCLUSION_KEY] == "node_beta_projection"

    bad = _overlay_clone(tmp_path / "bad", {NODE_EXCLUSION_KEY: "node_beta_projction"})
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", bad)
    with pytest.raises(LaneEnvFragmentError, match="node_beta_projction"):
        executor_mod._compose_env()
