# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Stage handlers of the stand-in sweep (OMN-21071).

Each answers its stage command with a value naming the stage node, so the
sweep's result shows which hosted node answered. The silent stage answers
nothing, which is how a stage terminal that never arrives is produced.
"""

from __future__ import annotations

from tests.helpers.execution_graph_stand_in.model_stand_in_sweep import (
    ModelStandInStageRequest,
    ModelStandInStageResult,
)


class HandlerStandInStageAlpha:
    """Answers as ``node_stand_in_stage_alpha``."""

    def handle(self, request: ModelStandInStageRequest) -> ModelStandInStageResult:
        return ModelStandInStageResult(
            correlation_id=request.correlation_id,
            stage_key=request.stage_key,
            value=f"node_stand_in_stage_alpha:{request.operand}",
        )


class HandlerStandInStageBeta:
    """Answers as ``node_stand_in_stage_beta``."""

    def handle(self, request: ModelStandInStageRequest) -> ModelStandInStageResult:
        return ModelStandInStageResult(
            correlation_id=request.correlation_id,
            stage_key=request.stage_key,
            value=f"node_stand_in_stage_beta:{request.operand}",
        )


class HandlerStandInStageSilent:
    """Receives its stage command and publishes no terminal."""

    def handle(self, request: ModelStandInStageRequest) -> None:
        del request
