# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: build loop handler runs end-to-end in-memory.

Tests that HandlerLoopOrchestrator completes a full FSM cycle with
placeholder data, verifying JSON serialization, disk state writing,
and workflow YAML validity — the same execution path RuntimeLocal uses.

Related:
    - OMN-7472: RuntimeLocal integration test
    - OMN-7475: Proof of life
    - OMN-7468: Local ONEX Workflow Runner epic
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.handlers.handler_loop_orchestrator import (
    HandlerLoopOrchestrator,
)
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.models.model_loop_start_command import (
    ModelLoopStartCommand,
)

WORKFLOW_YAML_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "workflows"
    / "build_loop_workflow.yaml"
)


def _runtime_local_complete() -> bool:
    """Check if RuntimeLocal has the completed _build_initial_payload method."""
    try:
        from omnibase_core.runtime.runtime_local import RuntimeLocal

        return hasattr(RuntimeLocal, "_build_initial_payload")
    except ImportError:
        return False


@pytest.mark.unit
class TestBuildLoopRuntimeLocalIntegration:
    """Tests that mirror what RuntimeLocal does: run handler, serialize, write to disk."""

    async def test_handler_result_json_serializable(self) -> None:
        """Handler result can be serialized to JSON (for disk state)."""
        handler = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=uuid4(),
            max_cycles=1,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await handler.handle(command)

        data = result.model_dump(mode="json")
        json_str = json.dumps(data, indent=2, default=str)
        assert "cycles_completed" in json_str
        assert "cycle_summaries" in json_str

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["cycles_completed"] == 1
        assert parsed["cycles_failed"] == 0

    async def test_handler_result_written_to_disk(self, tmp_path: Path) -> None:
        """Verify we can write handler result to disk like RuntimeLocal would."""
        handler = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=uuid4(),
            max_cycles=1,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await handler.handle(command)

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "workflow_result.json"
        data = result.model_dump(mode="json")
        state_file.write_text(json.dumps(data, indent=2, default=str))

        assert state_file.exists()
        loaded = json.loads(state_file.read_text())
        assert loaded["cycles_completed"] == 1
        assert loaded["cycles_failed"] == 0
        assert loaded["cycle_summaries"][0]["final_phase"] == "complete"

    async def test_skip_closeout_written_to_disk(self, tmp_path: Path) -> None:
        """skip_closeout=True result also serializes correctly."""
        handler = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=uuid4(),
            max_cycles=1,
            skip_closeout=True,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await handler.handle(command)
        assert result.cycles_completed == 1

        state_file = tmp_path / "workflow_result.json"
        state_file.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, default=str)
        )
        loaded = json.loads(state_file.read_text())
        assert loaded["cycle_summaries"][0]["final_phase"] == "complete"


@pytest.mark.unit
class TestWorkflowYamlValidity:
    """Validate the build_loop_workflow.yaml contract."""

    def test_workflow_yaml_exists(self) -> None:
        """build_loop_workflow.yaml must exist at the expected path."""
        assert WORKFLOW_YAML_PATH.exists(), (
            f"Workflow YAML not found at {WORKFLOW_YAML_PATH}"
        )

    def test_workflow_yaml_parseable(self) -> None:
        """Workflow YAML must parse as valid YAML."""
        content = yaml.safe_load(WORKFLOW_YAML_PATH.read_text())
        assert isinstance(content, dict)

    def test_workflow_yaml_has_required_fields(self) -> None:
        """Workflow YAML must have all required contract fields."""
        content = yaml.safe_load(WORKFLOW_YAML_PATH.read_text())
        assert content["workflow_id"] == "build_loop"
        assert "handler" in content
        assert content["handler"]["class"] == "HandlerLoopOrchestrator"
        assert content["handler"]["input_model"]["class"] == "ModelLoopStartCommand"

    def test_workflow_yaml_handler_importable(self) -> None:
        """Handler class referenced in YAML must be importable."""
        content = yaml.safe_load(WORKFLOW_YAML_PATH.read_text())
        handler_module = content["handler"]["module"]
        handler_class = content["handler"]["class"]

        import importlib

        mod = importlib.import_module(handler_module)
        cls = getattr(mod, handler_class)
        assert cls is HandlerLoopOrchestrator

    def test_workflow_yaml_input_model_importable(self) -> None:
        """Input model referenced in YAML must be importable."""
        content = yaml.safe_load(WORKFLOW_YAML_PATH.read_text())
        model_module = content["handler"]["input_model"]["module"]
        model_class = content["handler"]["input_model"]["class"]

        import importlib

        mod = importlib.import_module(model_module)
        cls = getattr(mod, model_class)
        assert cls is ModelLoopStartCommand

    def test_workflow_yaml_nodes_list(self) -> None:
        """Workflow must declare all 7 nodes."""
        content = yaml.safe_load(WORKFLOW_YAML_PATH.read_text())
        nodes = content["nodes"]
        assert len(nodes) == 7
        assert "node_autonomous_loop_orchestrator" in nodes
        assert "node_loop_state_reducer" in nodes


@pytest.mark.unit
class TestRuntimeLocalProofOfLife:
    """Proof-of-life: RuntimeLocal with workflow YAML (conditional on PR merge)."""

    def test_runtime_local_import(self) -> None:
        """RuntimeLocal must be importable from omnibase_core."""
        from omnibase_core.runtime.runtime_local import RuntimeLocal

        assert RuntimeLocal is not None

    def test_runtime_local_has_run_method(self) -> None:
        """RuntimeLocal must expose a run() method."""
        from omnibase_core.runtime.runtime_local import RuntimeLocal

        assert hasattr(RuntimeLocal, "run")

    @pytest.mark.skipif(
        not _runtime_local_complete(),
        reason="RuntimeLocal._build_initial_payload not yet available (PR not merged)",
    )
    async def test_runtime_local_with_workflow_yaml(self, tmp_path: Path) -> None:
        """RuntimeLocal loads the workflow YAML and runs the build loop."""
        from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
        from omnibase_core.runtime.runtime_local import RuntimeLocal

        runtime = RuntimeLocal(
            workflow_path=WORKFLOW_YAML_PATH,
            state_root=tmp_path / "state",
            timeout=30,
        )
        result = runtime.run()
        assert result == EnumWorkflowResult.COMPLETED
        assert runtime.exit_code == 0

        state_file = tmp_path / "state" / "workflow_result.json"
        assert state_file.exists()
