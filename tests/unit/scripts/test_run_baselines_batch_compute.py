# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/run_baselines_batch_compute.py [OMN-11177]."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
import yaml

from omnibase_core.validators.contract_config_compliance import validate_file

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_baselines_batch_compute.py"
CONTRACT_PATH = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_baselines_batch_compute"
    / "contract.yaml"
)
COMMAND_TOPIC = "onex.cmd.omnibase-infra.baselines-batch-compute.v1"
HANDLER_MODULE = (
    "omnibase_infra.nodes.node_baselines_batch_compute.handlers."
    "handler_baselines_batch_compute"
)

pytestmark = pytest.mark.unit


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_test_run_baselines_batch_compute",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_prints_typed_command_envelope() -> None:
    correlation_id = "00000000-0000-4000-8000-000000011177"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--correlation-id",
            correlation_id,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload_text = result.stdout.split("\n(dry-run:", maxsplit=1)[0]
    envelope = json.loads(payload_text)

    assert envelope["event_type"] == COMMAND_TOPIC
    assert envelope["source_tool"] == "run_baselines_batch_compute"
    assert envelope["target_tool"] == "node_baselines_batch_compute"
    assert envelope["payload_type"] == "ModelBaselinesBatchComputeCommand"
    assert envelope["correlation_id"] == correlation_id
    assert envelope["payload"] == {
        "operation": "baselines.batch_compute",
        "correlation_id": correlation_id,
    }


def test_build_envelope_does_not_import_handler_module() -> None:
    sys.modules.pop(HANDLER_MODULE, None)
    script = _load_script()

    envelope = script.build_envelope(UUID("00000000-0000-4000-8000-000000011177"))

    assert envelope.event_type == COMMAND_TOPIC
    assert HANDLER_MODULE not in sys.modules


def test_contract_config_validator_has_no_bus_bypass_import_finding() -> None:
    findings = validate_file(SCRIPT_PATH, frozenset({"bus_bypass_import"}))

    assert findings == []


def test_node_contract_declares_command_subscription() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert COMMAND_TOPIC in contract["event_bus"]["subscribe_topics"]
