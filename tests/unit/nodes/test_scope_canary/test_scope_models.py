# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scope canary Pydantic models."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extracted import (
    ModelScopeExtracted,
)
from omnibase_infra.nodes.node_scope_file_read_effect.models.model_scope_file_read_result import (
    ModelScopeFileReadResult,
)
from omnibase_infra.nodes.node_scope_manifest_write_effect.models.model_scope_manifest_written import (
    ModelScopeManifestWritten,
)
from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.model_scope_check_command import (
    ModelScopeCheckCommand,
)
from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.model_scope_check_result import (
    ModelScopeCheckResult,
)


@pytest.mark.unit
class TestScopeModels:
    """Tests for all scope canary models."""

    def test_scope_check_command(self):
        """ModelScopeCheckCommand validates correctly."""
        cid = uuid4()
        cmd = ModelScopeCheckCommand(
            correlation_id=cid,
            plan_file_path="/var/data/plan.md",
        )
        assert cmd.correlation_id == cid
        assert cmd.output_path == "~/.claude/scope-manifest.json"
        assert cmd.auto_confirm is False

    def test_scope_check_command_frozen(self):
        """ModelScopeCheckCommand is immutable."""
        cmd = ModelScopeCheckCommand(
            correlation_id=uuid4(), plan_file_path="/var/data/plan.md"
        )
        with pytest.raises(ValidationError):
            cmd.plan_file_path = "/other"  # type: ignore[misc]

    def test_scope_check_command_forbids_extra(self):
        """ModelScopeCheckCommand rejects extra fields."""
        with pytest.raises(ValidationError):
            ModelScopeCheckCommand(
                correlation_id=uuid4(),
                plan_file_path="/var/data/plan.md",
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_scope_file_read_result(self):
        """ModelScopeFileReadResult constructs correctly."""
        cid = uuid4()
        result = ModelScopeFileReadResult(
            correlation_id=cid,
            file_path="/var/data/plan.md",
            content="hello",
        )
        assert result.success is True
        assert result.error_message == ""

    def test_scope_extracted(self):
        """ModelScopeExtracted uses tuples for immutable collections."""
        cid = uuid4()
        extracted = ModelScopeExtracted(
            correlation_id=cid,
            plan_file_path="plan.md",
            files=("a.py", "b.py"),
            repos=("omniclaude",),
        )
        assert isinstance(extracted.files, tuple)
        assert len(extracted.files) == 2

    def test_scope_manifest_written(self):
        """ModelScopeManifestWritten constructs correctly."""
        cid = uuid4()
        written = ModelScopeManifestWritten(
            correlation_id=cid,
            manifest_path="/var/data/manifest.json",
        )
        assert written.success is True

    def test_scope_check_result(self):
        """ModelScopeCheckResult constructs correctly."""
        cid = uuid4()
        result = ModelScopeCheckResult(
            correlation_id=cid,
            manifest_path="/var/data/manifest.json",
            files_count=3,
            directories_count=1,
            repos_count=2,
            systems_count=1,
        )
        assert result.status.value == "complete"
        assert result.files_count == 3
