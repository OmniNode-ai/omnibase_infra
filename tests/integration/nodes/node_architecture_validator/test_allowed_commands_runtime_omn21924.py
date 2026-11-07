# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Round-trip command policy through the installed node CLI and its event bus."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("field", "command", "allowed"),
    [
        pytest.param("command", ["onex-runtime"], True, id="runtime-allowed"),
        pytest.param(
            "command", "onex-runtime --help", True, id="runtime-arguments-allowed"
        ),
        pytest.param(
            "command", ["python", "-m", "planted.sidecar"], False, id="argv-refused"
        ),
        pytest.param(
            "command", "python -m planted.sidecar", False, id="string-refused"
        ),
        pytest.param(
            "entrypoint",
            ["python", "-m", "planted.sidecar"],
            False,
            id="entrypoint-argv-refused",
        ),
        pytest.param(
            "entrypoint",
            "python -m planted.sidecar",
            False,
            id="entrypoint-string-refused",
        ),
        pytest.param(
            "command",
            ["sh", "-c", "onex-runtime; python -m planted.sidecar"],
            False,
            id="shell-wrapper-refused",
        ),
    ],
)
def test_allowed_commands_runtime_round_trip(
    tmp_path: Path, field: str, command: str | list[str], allowed: bool
) -> None:
    """Real discovery, wiring and dispatch preserve admission and refusal details."""
    policy = {
        "allowed_commands": [
            {
                "argv": ["onex-runtime"],
                "match": "executable",
                "field": "command",
                "services": ["omninode-runtime"],
                "reason": "Runtime command positive control",
            }
        ],
        "command_baseline": [],
        "baseline": [],
        "console_scripts": {},
    }
    correlation_id = str(uuid4())
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "correlation_id": correlation_id,
                "stack_shape": {
                    "policy_yaml": json.dumps(policy),
                    "compose_files": {
                        "probe.yml": json.dumps(
                            {"services": {"omninode-runtime": {field: command}}}
                        )
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "runtime-state"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "onex",
            "node",
            "node_architecture_validator",
            "--input",
            str(request),
            "--backend",
            "event_bus=inmemory",
            "--state-root",
            str(state),
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((state / "workflow_result.json").read_text(encoding="utf-8"))
    terminal = result["terminal_payload"]
    assert terminal["correlation_id"] == correlation_id
    assert "stack_shape_commands" in terminal["rules_checked"]
    if allowed:
        assert terminal["violations"] == [], terminal
    else:
        (violation,) = terminal["violations"]
        assert violation["rule_id"] == "command_not_allowed"
        assert violation["target_name"] == "omninode-runtime"
        assert violation["location"] == "probe.yml"
        assert field in violation["message"]
        assert "omninode-runtime" in violation["message"]
        assert "python -m planted.sidecar" in violation["message"]
