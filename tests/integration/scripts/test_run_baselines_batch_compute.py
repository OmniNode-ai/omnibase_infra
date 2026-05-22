# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for scripts/run_baselines_batch_compute.py [OMN-11177].

Validates the bus-publisher integration contract:
  - The command topic is read from the canonical contract.yaml at runtime (not hardcoded)
  - The command model class loads cleanly via importlib isolation
  - dry-run mode produces a valid, schema-correct ModelEventEnvelope
  - The published topic matches the node contract subscribe_topic declaration

No Kafka connection is required; tests use --dry-run mode throughout.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml

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

pytestmark = pytest.mark.integration


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location(
        "_test_run_baselines_batch_compute_integration",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_command_topic_matches_node_contract_subscribe_topic() -> None:
    """The script must read its publish target from the node contract at runtime.

    This integration test verifies the full round-trip: the script loads the
    contract.yaml, extracts the subscribe_topic, and uses it as the event_type
    in the published envelope — ensuring topic drift between the script and the
    node contract is caught automatically.
    """
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    subscribe_topics = contract["event_bus"]["subscribe_topics"]
    cmd_topics = [t for t in subscribe_topics if t.startswith("onex.cmd.")]
    assert len(cmd_topics) == 1, (
        f"Expected exactly one onex.cmd.* subscribe topic in node contract, "
        f"got {cmd_topics}"
    )
    expected_topic = cmd_topics[0]

    correlation_id = str(uuid4())
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
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    payload_text = result.stdout.split("\n(dry-run:", maxsplit=1)[0]
    envelope = json.loads(payload_text)
    assert envelope["event_type"] == expected_topic, (
        f"Script published to {envelope['event_type']!r} but contract declares "
        f"{expected_topic!r} as the subscribe_topic. They must match."
    )


@pytest.mark.integration
def test_envelope_schema_is_valid_for_bus_consumption() -> None:
    """The published envelope must carry all required ModelEventEnvelope fields."""
    correlation_id = "00000000-0000-4000-a000-000011177001"
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
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    payload_text = result.stdout.split("\n(dry-run:", maxsplit=1)[0]
    envelope = json.loads(payload_text)

    # All fields required by ModelEventEnvelope for bus consumption
    required_fields = {
        "event_type",
        "source_tool",
        "target_tool",
        "payload_type",
        "correlation_id",
        "payload",
    }
    missing = required_fields - envelope.keys()
    assert not missing, f"Envelope is missing required fields: {missing}"

    # payload must be a dict (not null) with correlation_id propagated
    assert isinstance(envelope["payload"], dict), "payload must be a dict"
    assert envelope["payload"]["correlation_id"] == correlation_id

    # UUID round-trip: correlation_id must be a valid UUID
    assert UUID(envelope["correlation_id"]) == UUID(correlation_id)


@pytest.mark.integration
def test_command_model_loads_via_importlib_isolation() -> None:
    """The script must load ModelBaselinesBatchComputeCommand without importing the full node."""
    module = _load_script()
    command_cls = (
        module.build_command.__func__
        if hasattr(module.build_command, "__func__")
        else None
    )  # type: ignore[attr-defined]

    # Use the module's public build_command function with a known correlation_id
    from uuid import UUID

    correlation_id = UUID("00000000-0000-4000-b000-000011177002")
    command = module.build_command(correlation_id)  # type: ignore[attr-defined]

    assert command.correlation_id == correlation_id
    dumped = command.model_dump(mode="json")
    assert dumped["correlation_id"] == str(correlation_id)
