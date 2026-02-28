# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelBaselinesBatchComputeCommand and ModelBaselinesBatchComputeOutput.

Ticket: OMN-3043
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit

from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_command import (
    ModelBaselinesBatchComputeCommand,
)
from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_output import (
    ModelBaselinesBatchComputeOutput,
)
from omnibase_infra.services.observability.baselines.models.model_batch_compute_baselines_result import (
    ModelBatchComputeBaselinesResult,
)


class TestModelBaselinesBatchComputeCommand:
    """Tests for ModelBaselinesBatchComputeCommand (D1: required correlation_id)."""

    def test_command_requires_correlation_id(self) -> None:
        """D1: missing correlation_id must raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelBaselinesBatchComputeCommand()  # type: ignore[call-arg]

    def test_command_accepts_correlation_id(self) -> None:
        """D1: valid UUID4 correlation_id is accepted."""
        cid = uuid4()
        cmd = ModelBaselinesBatchComputeCommand(correlation_id=cid)
        assert cmd.operation == "baselines.batch_compute"
        assert cmd.correlation_id == cid

    def test_command_operation_default(self) -> None:
        """operation defaults to 'baselines.batch_compute'."""
        cmd = ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        assert cmd.operation == "baselines.batch_compute"

    def test_command_is_frozen(self) -> None:
        """model_config frozen=True: assignment after creation raises."""
        cmd = ModelBaselinesBatchComputeCommand(correlation_id=uuid4())
        with pytest.raises(Exception):
            cmd.operation = "other"  # type: ignore[misc]

    def test_command_rejects_extra_fields(self) -> None:
        """model_config extra='forbid': unknown fields raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelBaselinesBatchComputeCommand(
                correlation_id=uuid4(),
                unexpected_field="x",  # type: ignore[call-arg]
            )


class TestModelBaselinesBatchComputeOutput:
    """Tests for ModelBaselinesBatchComputeOutput."""

    def _make_result(self, **kwargs: object) -> ModelBatchComputeBaselinesResult:
        return ModelBatchComputeBaselinesResult(
            completed_at=datetime.now(UTC), **kwargs
        )

    def test_output_snapshot_emitted_true(self) -> None:
        result = self._make_result(comparisons_rows=5, trend_rows=3, breakdown_rows=2)
        out = ModelBaselinesBatchComputeOutput(result=result, snapshot_emitted=True)
        assert out.snapshot_emitted is True
        assert out.result.total_rows == 10

    def test_output_snapshot_emitted_false(self) -> None:
        result = self._make_result()
        out = ModelBaselinesBatchComputeOutput(result=result, snapshot_emitted=False)
        assert out.snapshot_emitted is False

    def test_output_is_frozen(self) -> None:
        result = self._make_result()
        out = ModelBaselinesBatchComputeOutput(result=result, snapshot_emitted=False)
        with pytest.raises(Exception):
            out.snapshot_emitted = True  # type: ignore[misc]
