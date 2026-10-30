# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract-routed resolver, kernel and pre-push guard agree on real trees."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast

import pytest
import yaml
from pydantic import BaseModel

from omnibase_infra.models.workspace import (
    ModelWorkspaceConfigRootRequest,
    ModelWorkspaceConfigRootResolution,
)
from omnibase_infra.runtime.service_kernel import (
    WorkspaceRuntimeConfigMissingError,
    resolve_embedded_runtime_config,
)
from scripts.hooks.pytest_full_suite_host_guard import _workspace_config_root

pytestmark = pytest.mark.integration

_OWNER_CONFIG = """
description: "the selected owner's runtime config"
event_bus:
  type: "kafka"
  profile: "local"
  lane: "dev"
"""

_NEIGHBOUR_CONFIG = """
description: "a neighbouring directory's runtime config (must never be read)"
event_bus:
  type: "kafka"
  profile: "local"
  lane: "dev"
"""

CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_workspace_config_root_resolve_effect/contract.yaml"
)


class _Resolver(Protocol):
    def handle(
        self, request: ModelWorkspaceConfigRootRequest
    ) -> ModelWorkspaceConfigRootResolution: ...


def _load(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("winner", ["environment", "developer", "unset"])
def test_contract_kernel_and_guard_resolve_real_directory_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winner: str
) -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    (route,) = contract["handler_routing"]["handlers"]
    assert route["operation"] == "workspace_config_root.resolve"
    handler_class = _load(route["handler"]["module"], route["handler"]["name"])
    request_class = _load(
        contract["input_model"]["module"], contract["input_model"]["name"]
    )
    output_class = _load(
        contract["output_model"]["module"], contract["output_model"]["name"]
    )
    assert isinstance(handler_class, type)
    assert isinstance(request_class, type) and issubclass(request_class, BaseModel)
    assert request_class is ModelWorkspaceConfigRootRequest
    assert isinstance(output_class, type)
    assert output_class is ModelWorkspaceConfigRootResolution

    workspace = tmp_path / "registry"
    workspace.mkdir()
    # Beside the workspace sit directories that hold a real, loadable runtime
    # config: a resolver that guessed one would answer from it instead of
    # refusing or taking the developer directory.
    for neighbour in ("config", "neighbor", "workspace-config"):
        planted = tmp_path / neighbour / "config/onex/runtime/runtime_config.yaml"
        planted.parent.mkdir(parents=True)
        planted.write_text(_NEIGHBOUR_CONFIG, encoding="utf-8")
    developer = workspace / ".onex/workspace-config"
    explicit = tmp_path / "explicit"
    if winner != "unset":
        developer.mkdir(parents=True)
    env: dict[str, str] = {}
    if winner == "environment":
        explicit.mkdir()
        env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(explicit)
        monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(explicit))
    else:
        monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # A developer directory under HOME, where a resolver reading it from the home
    # directory instead of the workspace would find a loadable config.
    home_decoy = tmp_path / "home/.onex/workspace-config/config/onex/runtime"
    home_decoy.mkdir(parents=True)
    (home_decoy / "runtime_config.yaml").write_text(_NEIGHBOUR_CONFIG, encoding="utf-8")
    monkeypatch.setenv("OMNI_HOME", str(workspace))
    handler = cast("_Resolver", handler_class())
    request = request_class.model_validate(
        {"workspace_root": workspace, "environment": env}
    )
    assert isinstance(request, ModelWorkspaceConfigRootRequest)
    result = handler.handle(request)
    assert isinstance(result, output_class)
    expected = {"environment": explicit, "developer": developer, "unset": None}[winner]
    assert result.source == winner
    assert result.config_root == (None if expected is None else expected.resolve())

    # The real kernel answers from exactly the selected owner, never a decoy: it
    # refuses naming the selected owner's file while that file is absent, loads it
    # once written, and says that no config root is declared when none is.
    if expected is None:
        with pytest.raises(WorkspaceRuntimeConfigMissingError) as exc:
            resolve_embedded_runtime_config(workspace_root=workspace)
        assert "ONEX_WORKSPACE_CONFIG_ROOT is unset" in str(exc.value)
        assert "a config root (none is declared)" in str(exc.value)
        # Nothing was read: the decoys beside the workspace and under HOME all
        # hold loadable configs, so a kernel that read one would have answered.
        assert exc.value.model.context["config_path"] == ""
    else:
        expected_file = expected.resolve() / "config/onex/runtime/runtime_config.yaml"
        with pytest.raises(WorkspaceRuntimeConfigMissingError) as exc:
            resolve_embedded_runtime_config(workspace_root=workspace)
        assert exc.value.model.context["config_path"] == str(expected_file)
        expected_file.parent.mkdir(parents=True, exist_ok=True)
        expected_file.write_text(_OWNER_CONFIG, encoding="utf-8")
        loaded, provenance = resolve_embedded_runtime_config(workspace_root=workspace)
        assert loaded.description == "the selected owner's runtime config"
        assert provenance == f"workspace tier-1 runtime config at {expected_file}"
    # The pre-push guard honours the explicit variable only, never the
    # developer directory, and agrees with the handler under that policy.
    guard_root = _workspace_config_root(env)
    assert guard_root == (explicit.resolve() if winner == "environment" else None)
    guard_policy = handler.handle(
        ModelWorkspaceConfigRootRequest(
            workspace_root=workspace, environment=env, allow_developer_config=False
        )
    )
    assert guard_policy.config_root == guard_root
    # The flag, not the tree, is what rejects the developer directory: the same
    # request with the default policy took it above.
    assert guard_policy.source == (
        "environment" if winner == "environment" else "unset"
    )
