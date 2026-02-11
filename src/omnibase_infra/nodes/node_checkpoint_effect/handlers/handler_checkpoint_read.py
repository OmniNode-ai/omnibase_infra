# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that reads a specific checkpoint from disk.

When multiple attempts exist for the same phase, returns the checkpoint with
the highest ``attempt_number`` (latest attempt).

Ticket: OMN-2143
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import yaml
from pydantic import ValidationError

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint
from omnibase_infra.nodes.node_checkpoint_effect.models.model_checkpoint_effect_output import (
    ModelCheckpointEffectOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".claude" / "checkpoints"


class HandlerCheckpointRead:
    """Reads a checkpoint for a given ticket, run, and phase.

    If multiple attempts exist, the one with the highest ``attempt_number``
    is returned.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler."""
        self._initialized = True
        logger.info("HandlerCheckpointRead initialized")

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerCheckpointRead shutdown")

    async def execute(self, envelope: dict[str, object]) -> ModelHandlerOutput:
        """Read the latest checkpoint for the requested phase.

        Envelope keys:
            ticket_id: str — which ticket.
            run_id: UUID — which pipeline run.
            phase: EnumCheckpointPhase — which phase to read.
            correlation_id: UUID for tracing.
            base_dir: Optional override for the checkpoint root.
        """
        correlation_id_raw = envelope.get("correlation_id")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else None,
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="read_checkpoint",
            target_name="checkpoint_yaml",
        )
        corr_id = context.correlation_id  # with_correlation() guarantees a UUID

        ticket_id = envelope.get("ticket_id")
        run_id = envelope.get("run_id")
        phase_raw = envelope.get("phase")

        if not ticket_id or not run_id or phase_raw is None:
            raise RuntimeHostError(
                "read_checkpoint requires ticket_id, run_id, and phase",
                context=context,
            )

        # Coerce phase
        if isinstance(phase_raw, EnumCheckpointPhase):
            phase = phase_raw
        elif isinstance(phase_raw, str):
            phase = EnumCheckpointPhase(phase_raw)
        else:
            raise RuntimeHostError(
                f"Unexpected phase type: {type(phase_raw).__name__}",
                context=context,
            )

        run_id_val = UUID(str(run_id)) if not isinstance(run_id, UUID) else run_id

        # Resolve base directory
        base_dir_raw = envelope.get("base_dir")
        base_dir = Path(str(base_dir_raw)) if base_dir_raw else _DEFAULT_BASE_DIR

        # Scan for matching files (with path traversal guard)
        target_dir = base_dir / str(ticket_id) / str(run_id_val)
        if not target_dir.resolve().is_relative_to(base_dir.resolve()):
            raise RuntimeHostError(
                "Path traversal detected: ticket_id escapes checkpoint root",
                context=context,
            )
        phase_prefix = f"phase_{phase.phase_number}_{phase.value}_a"

        if not target_dir.is_dir():
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-checkpoint-read",
                result=ModelCheckpointEffectOutput(
                    success=False,
                    correlation_id=corr_id,
                    error=f"Checkpoint directory not found: {target_dir.name}",
                ),
            )

        # Find all attempt files for this phase, sorted by attempt number
        def _attempt_number(path: Path) -> int:
            """Extract numeric attempt from filename like phase_1_implement_a3.yaml."""
            stem = path.stem  # e.g. "phase_1_implement_a3"
            after_a = stem.rsplit("_a", maxsplit=1)
            if len(after_a) == 2 and after_a[1].isdigit():
                return int(after_a[1])
            return 0

        matching_files = sorted(
            target_dir.glob(f"{phase_prefix}*.yaml"),
            key=_attempt_number,
        )

        if not matching_files:
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-checkpoint-read",
                result=ModelCheckpointEffectOutput(
                    success=False,
                    correlation_id=corr_id,
                    error=f"No checkpoint found for phase {phase.value}",
                ),
            )

        # Read the latest attempt (last file alphabetically = highest attempt)
        latest_file = matching_files[-1]
        try:
            raw_data = yaml.safe_load(latest_file.read_text(encoding="utf-8"))
            if not isinstance(raw_data, dict):
                raise RuntimeHostError(
                    f"Corrupt checkpoint file {latest_file.name}: expected mapping, "
                    f"got {type(raw_data).__name__}",
                    context=context,
                )
            checkpoint = ModelCheckpoint.model_validate(raw_data)
        except yaml.YAMLError as exc:
            raise RuntimeHostError(
                f"Corrupt checkpoint file {latest_file.name}: invalid YAML",
                context=context,
            ) from exc
        except ValidationError as exc:
            raise RuntimeHostError(
                f"Corrupt checkpoint file {latest_file.name}: {exc.error_count()} errors",
                context=context,
            ) from exc

        logger.info(
            "Checkpoint read: %s (phase=%s, attempt=%d)",
            latest_file.name,
            checkpoint.phase.value,
            checkpoint.attempt_number,
        )

        result = ModelCheckpointEffectOutput(
            success=True,
            correlation_id=corr_id,
            checkpoint=checkpoint,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-checkpoint-read",
            result=result,
        )


__all__: list[str] = ["HandlerCheckpointRead"]
