# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime composition of the signed execution-graph gateway capability."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess


def _key_file(path: Path, runtime_id: str = "gateway") -> Path:
    path.write_text(
        json.dumps({"keys": {runtime_id: base64.urlsafe_b64encode(b"a" * 32).decode()}})
    )
    path.chmod(0o600)
    return path


def _config(path: Path) -> dict[str, object]:
    return {
        "service_name": "omnibase-infra",
        "node_name": "graph-read",
        "execution_graph_read_gateway": {
            "command_topic": "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1",
            "runtime_id": "gateway",
            "realm": "dev",
            "bus_id": "bus",
            "public_key_path": str(path),
        },
    }


@pytest.mark.unit  # type: ignore[untyped-decorator]
def test_graph_gateway_runtime_config_builds_exact_provider(tmp_path: Path) -> None:
    host = RuntimeHostProcess(config=_config(_key_file(tmp_path / "keys.json")))
    ingress, provider = host._execution_graph_read_ingress_dependencies()
    assert ingress is not None and provider is not None
    assert next(iter(ingress.gateway_policy.scopes)).runtime_id == "gateway"


@pytest.mark.unit  # type: ignore[untyped-decorator]
def test_graph_gateway_runtime_config_refuses_missing_or_unregistered_key(
    tmp_path: Path,
) -> None:
    host = RuntimeHostProcess(config=_config(tmp_path / "missing.json"))
    with pytest.raises(ProtocolConfigurationError):
        host._execution_graph_read_ingress_dependencies()
    host = RuntimeHostProcess(
        config=_config(_key_file(tmp_path / "other.json", "other"))
    )
    with pytest.raises(ProtocolConfigurationError):
        host._execution_graph_read_ingress_dependencies()
