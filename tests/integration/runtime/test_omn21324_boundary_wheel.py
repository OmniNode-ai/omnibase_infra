# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run boundary and replay checks from a built wheel outside the checkout."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.mark.integration
def test_deployable_wheel_classifies_undecodable_body_and_refuses_replay(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    wheels = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (wheel,) = wheels.glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    probe = r"""
import json
import sys
from pathlib import Path
from uuid import UUID
sys.path.insert(0, sys.argv[1])
import omnibase_infra
from omnibase_infra.runtime.boundary_failure_terminal import classify_boundary_failure
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import ModelDlqReplayEngineConfig, should_replay
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import ModelDlqMessage
assert Path(omnibase_infra.__file__).resolve().is_relative_to(Path(sys.argv[1]))
correlation = UUID("95c783ae-79b6-47bb-a1d9-2edc46f66a45")
config = ModelDlqReplayEngineConfig(bootstrap_servers="localhost:9092", dlq_topic="onex.dlq.omnibase-infra.commands.v1")
for exc in [json.JSONDecodeError("bad body", "", 0), UnicodeDecodeError("utf8", b"\xff", 0, 1, "invalid")]:
    terminal = classify_boundary_failure(exc, topic="onex.cmd.omnimarket.delegate-skill.v1", correlation_id=correlation, failure_reason=str(exc))
    assert terminal.failure_code == "ONEX_CORE_007_INVALID_INPUT"
    assert terminal.retryable is False
for name, expected in [("UnicodeDecodeError", False), ("ProtocolConfigurationError", False), ("TimeoutError", True)]:
    message = ModelDlqMessage(original_topic="onex.cmd.omnimarket.delegate-skill.v1", original_value='{"task":"x"}', failure_reason=f"HandlerDispatchFailureError: {name}: failed", correlation_id=correlation, error_type="HandlerDispatchFailureError", dlq_offset=1, dlq_partition=0)
    assert should_replay(message, config)[0] is expected
print("wheel boundary and replay probe passed")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(installed)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "wheel boundary and replay probe passed" in result.stdout
