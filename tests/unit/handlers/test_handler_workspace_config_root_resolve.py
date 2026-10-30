# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Workspace config root policy and import boundaries (OMN-17232)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.handler_workspace_config_root_resolve import (
    DEVELOPER_WORKSPACE_CONFIG_RELATIVE_PATH,
    ENV_WORKSPACE_CONFIG_ROOT,
    HandlerWorkspaceConfigRootResolve,
)
from omnibase_infra.models.workspace import (
    ModelWorkspaceConfigRootRequest,
    ModelWorkspaceConfigRootResolution,
)

pytestmark = pytest.mark.unit


def _plant_neighbours(workspace: Path) -> None:
    """Directories beside the registry root that a resolver must never adopt."""
    for name in ("config", "neighbor", "workspace-config", ".onex"):
        (workspace.parent / name).mkdir(exist_ok=True)


@pytest.mark.parametrize("winner", ["environment", "developer", "unset"])
def test_rules_win_in_order_with_resolved_paths(tmp_path: Path, winner: str) -> None:
    workspace = tmp_path / "registry"
    workspace.mkdir()
    _plant_neighbours(workspace)
    developer = workspace / DEVELOPER_WORKSPACE_CONFIG_RELATIVE_PATH
    explicit = tmp_path / "explicit"
    env = {}
    if winner in {"environment", "developer"}:
        developer.mkdir(parents=True)
    if winner == "environment":
        env[ENV_WORKSPACE_CONFIG_ROOT] = str(explicit / ".." / "explicit")
    result = HandlerWorkspaceConfigRootResolve().handle(
        ModelWorkspaceConfigRootRequest(workspace_root=workspace / ".", environment=env)
    )
    expected = {"environment": explicit, "developer": developer, "unset": None}[winner]
    if expected is None:
        assert result.config_root is None
    else:
        assert result.config_root == expected.resolve()
        assert result.config_root.is_absolute()
    assert result.source == winner


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_blank_override_is_unset_and_no_neighbour_is_adopted(
    tmp_path: Path, blank: str
) -> None:
    workspace = tmp_path / "registry"
    workspace.mkdir()
    _plant_neighbours(workspace)
    result = HandlerWorkspaceConfigRootResolve().handle(
        ModelWorkspaceConfigRootRequest(
            workspace_root=workspace,
            environment={ENV_WORKSPACE_CONFIG_ROOT: blank},
        )
    )
    assert result.config_root is None
    assert result.source == "unset"


def test_developer_config_can_be_skipped(tmp_path: Path) -> None:
    workspace = tmp_path / "registry"
    (workspace / DEVELOPER_WORKSPACE_CONFIG_RELATIVE_PATH).mkdir(parents=True)
    result = HandlerWorkspaceConfigRootResolve().handle(
        ModelWorkspaceConfigRootRequest(
            workspace_root=workspace, environment={}, allow_developer_config=False
        )
    )
    assert result.config_root is None
    assert result.source == "unset"


def test_captured_mapping_replaces_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "ambient"
    captured = tmp_path / "captured"
    monkeypatch.setenv(ENV_WORKSPACE_CONFIG_ROOT, str(ambient))
    handler = HandlerWorkspaceConfigRootResolve()
    request = ModelWorkspaceConfigRootRequest(workspace_root=tmp_path / "registry")
    assert handler.handle(request).config_root == ambient.resolve()
    result = handler.handle(
        request.model_copy(
            update={"environment": {ENV_WORKSPACE_CONFIG_ROOT: str(captured)}}
        )
    )
    assert result.config_root == captured.resolve()
    assert result.source == "environment"
    empty = handler.handle(request.model_copy(update={"environment": {}}))
    assert empty.source == "unset"
    assert empty.config_root is None


def test_models_are_frozen_and_forbid_extra_fields(tmp_path: Path) -> None:
    request = ModelWorkspaceConfigRootRequest(workspace_root=tmp_path)
    resolution = HandlerWorkspaceConfigRootResolve().handle(request)
    for model in (request, resolution):
        with pytest.raises(ValidationError, match="frozen"):
            model.__setattr__(next(iter(type(model).model_fields)), tmp_path)
        with pytest.raises(ValidationError, match="extra_forbidden"):
            type(model).model_validate({**model.model_dump(), "unexpected": True})
    assert isinstance(resolution, ModelWorkspaceConfigRootResolution)
    assert (
        HandlerWorkspaceConfigRootResolve().handler_type
        is EnumHandlerType.INFRA_HANDLER
    )
    assert (
        HandlerWorkspaceConfigRootResolve().handler_category
        is EnumHandlerTypeCategory.EFFECT
    )


def test_handler_import_does_not_load_service_kernel() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import omnibase_infra.handlers.handler_workspace_config_root_resolve; assert 'omnibase_infra.runtime.service_kernel' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kernel_calls_the_contract_declared_resolver_and_owns_none_of_it() -> None:
    root = Path(__file__).resolve().parents[3]
    kernel = root / "src/omnibase_infra/runtime/service_kernel.py"
    handler = (
        root / "src/omnibase_infra/handlers/handler_workspace_config_root_resolve.py"
    )
    tree = ast.parse(kernel.read_text())
    assert not any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "workspace_runtime_config_root"
        for n in ast.walk(tree)
    )
    module = "omnibase_infra.handlers.handler_workspace_config_root_resolve"
    nodes = list(ast.walk(tree))
    assert any(
        isinstance(n, ast.ImportFrom)
        and n.module == module
        and any(a.name == "HandlerWorkspaceConfigRootResolve" for a in n.names)
        for n in nodes
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "HandlerWorkspaceConfigRootResolve"
        for n in nodes
    )
    # The resolver probes no directory beside the workspace root: it reads the
    # variable, then the developer directory under the root, and nothing else.
    handler_nodes = list(ast.walk(ast.parse(handler.read_text())))
    assert not any(
        isinstance(n, ast.Constant) and n.value == ".." for n in handler_nodes
    )
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "parent" for n in handler_nodes
    )


def test_guard_reads_the_explicit_root_exactly_as_the_handler_does(
    tmp_path: Path,
) -> None:
    from scripts.hooks.pytest_full_suite_host_guard import _workspace_config_root

    handler = HandlerWorkspaceConfigRootResolve()
    for value in ("", "  ", str(tmp_path), f"{tmp_path}/./x/.."):
        env = {ENV_WORKSPACE_CONFIG_ROOT: value}
        expected = handler.handle(
            ModelWorkspaceConfigRootRequest(
                workspace_root=tmp_path, environment=env, allow_developer_config=False
            )
        ).config_root
        assert _workspace_config_root(env) == expected
    assert _workspace_config_root({}) is None
