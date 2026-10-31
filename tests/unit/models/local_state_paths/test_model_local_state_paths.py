# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verify declared paths and their runtime defaults and overrides."""

from pathlib import Path
from types import ModuleType

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.clients.runtime_skill_client import default_runtime_socket_path
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.local_state_paths import (
    ModelLocalStatePaths,
    load_local_state_paths,
    model_local_state_paths,
)
from omnibase_infra.runtime import gateway_canary_probe, gateway_forwarder
from omnibase_infra.runtime.models.model_local_runtime_ingress_config import (
    ModelLocalRuntimeIngressConfig,
)


def test_loaded_paths() -> None:
    declaration = Path(model_local_state_paths.__file__).with_name(
        "local_state_paths.yaml"
    )
    settings = yaml.safe_load(declaration.read_text(encoding="utf-8"))["settings"]
    assert load_local_state_paths().model_dump() == settings


@pytest.mark.parametrize("key", ModelLocalStatePaths.model_fields)
def test_missing_setting_raises(tmp_path: Path, key: str) -> None:
    settings = load_local_state_paths().model_dump()
    del settings[key]
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"settings": settings}), encoding="utf-8")
    with pytest.raises(ValidationError, match=key):
        ModelLocalStatePaths.from_file(config)


@pytest.mark.parametrize("content", ["", "[]", "{}", "settings: null", "settings: []"])
def test_missing_settings_mapping_raises(tmp_path: Path, content: str) -> None:
    config = tmp_path / "paths.yaml"
    config.write_text(content, encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError, match="settings mapping"):
        ModelLocalStatePaths.from_file(config)


def test_runtime_socket_default_and_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ONEX_LOCAL_RUNTIME_SOCKET_PATH", raising=False)
    assert default_runtime_socket_path() == load_local_state_paths().runtime_socket_path
    override = str(tmp_path / "runtime.sock")
    monkeypatch.setenv("ONEX_LOCAL_RUNTIME_SOCKET_PATH", override)
    assert default_runtime_socket_path() == override
    monkeypatch.setenv("ONEX_LOCAL_RUNTIME_SOCKET_PATH", "")
    assert default_runtime_socket_path() == ""


def test_ingress_socket_default_and_override(tmp_path: Path) -> None:
    assert (
        ModelLocalRuntimeIngressConfig().socket_path
        == load_local_state_paths().runtime_socket_path
    )
    override = str(tmp_path / "ingress.sock")
    assert ModelLocalRuntimeIngressConfig(socket_path=override).socket_path == override


@pytest.mark.parametrize("module", [gateway_canary_probe, gateway_forwarder])
def test_gateway_argparse_defaults_and_overrides(
    module: ModuleType, tmp_path: Path
) -> None:
    # Both entrypoints share health paths; only the probe has a cadence cache.
    parser = module._build_parser()
    paths = load_local_state_paths()
    expected = {
        "egress_health_file": paths.gateway_egress_health_path,
        "lane_mirror_health_file": paths.gateway_lane_mirror_health_path,
    }
    arguments = ["--config", str(tmp_path / "gateway.yaml")]
    if module is gateway_canary_probe:
        expected["state_file"] = paths.gateway_canary_probe_state_path
        arguments.extend(["--broker-ref-map", str(tmp_path / "brokers.yaml")])
    defaults = parser.parse_args(arguments)
    for name, value in expected.items():
        assert getattr(defaults, name) == Path(value)
        arguments.extend(["--" + name.replace("_", "-"), str(tmp_path / name)])
    overrides = parser.parse_args(arguments)
    for name in expected:
        assert getattr(overrides, name) == tmp_path / name


def test_scope_models_default_to_declared_manifest_path() -> None:
    from uuid import uuid4

    from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extract_input import (
        ModelScopeExtractInput,
    )
    from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extracted import (
        ModelScopeExtracted,
    )
    from omnibase_infra.nodes.node_scope_file_read_effect.models.model_scope_file_read_request import (
        ModelScopeFileReadRequest,
    )
    from omnibase_infra.nodes.node_scope_file_read_effect.models.model_scope_file_read_result import (
        ModelScopeFileReadResult,
    )
    from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.model_scope_check_command import (
        ModelScopeCheckCommand,
    )

    declared = load_local_state_paths().scope_manifest_path
    correlation_id = uuid4()
    built = (
        ModelScopeExtractInput(
            correlation_id=correlation_id, plan_file_path="plan.md", content="x"
        ),
        ModelScopeExtracted(correlation_id=correlation_id, plan_file_path="plan.md"),
        ModelScopeFileReadRequest(correlation_id=correlation_id, file_path="plan.md"),
        ModelScopeFileReadResult(
            correlation_id=correlation_id, file_path="plan.md", content="x"
        ),
        ModelScopeCheckCommand(correlation_id=correlation_id, plan_file_path="plan.md"),
    )
    assert {model.output_path for model in built} == {declared}
