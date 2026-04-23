# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Schema tests for .github/workflows/reusable-runtime-provenance.yml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "reusable-runtime-provenance.yml"
)


def _on_block(workflow: dict[str, Any]) -> dict[str, Any]:
    typed: dict[Any, Any] = workflow
    block = typed.get("on", typed.get(True))
    assert isinstance(block, dict)
    return block


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_workflow_is_reusable(workflow: dict[str, Any]) -> None:
    assert "workflow_call" in _on_block(workflow)


def test_build_source_input_is_required(workflow: dict[str, Any]) -> None:
    inputs = _on_block(workflow)["workflow_call"]["inputs"]
    build_source = inputs["build_source"]
    assert build_source["required"] is True
    assert build_source["type"] == "string"
    description = str(build_source.get("description", "")).lower()
    assert "workspace" in description and "release" in description


def test_workspace_root_input_defaults_to_dot(workflow: dict[str, Any]) -> None:
    inputs = _on_block(workflow)["workflow_call"]["inputs"]
    workspace_root = inputs["workspace_root"]
    assert workspace_root["type"] == "string"
    assert workspace_root["default"] == "."


def test_job_contains_release_and_workspace_validation_steps(
    workflow: dict[str, Any],
) -> None:
    steps = workflow["jobs"]["provenance"]["steps"]
    text = "\n".join(str(step.get("run", "")) for step in steps)
    assert "runtime_build release-requirement" in text
    assert "runtime_build write-workspace-manifest" in text
    assert "git+https://github.com/OmniNode-ai" in text


def test_job_uploads_artifact(workflow: dict[str, Any]) -> None:
    steps = workflow["jobs"]["provenance"]["steps"]
    assert any("actions/upload-artifact" in str(step.get("uses", "")) for step in steps)
