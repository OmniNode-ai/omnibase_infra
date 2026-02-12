# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that lists available checkpoints for a ticket/run.

Returns all checkpoints found, across all phases and attempts.

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
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint
from omnibase_infra.nodes.node_checkpoint_effect.handlers.util_checkpoint_sorting import (
    attempt_number,
)
from omnibase_infra.nodes.node_checkpoint_effect.models.model_checkpoint_effect_output import (
    ModelCheckpointEffectOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".claude" / "checkpoints"


class HandlerCheckpointList:
    """Lists all checkpoint files for a ticket and optionally a specific run."""

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
        logger.info("HandlerCheckpointList initialized")

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerCheckpointList shutdown")

    async def execute(self, envelope: dict[str, object]) -> ModelHandlerOutput:
        """List checkpoints for a ticket and optional run.

        Envelope keys:
            ticket_id: str — which ticket.
            run_id: UUID | None — optional run scope.
            correlation_id: UUID for tracing.
            base_dir: Optional override for the checkpoint root.
        """
        correlation_id_raw = envelope.get("correlation_id")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else None,
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="list_checkpoints",
            target_name="checkpoint_yaml",
        )
        corr_id = context.correlation_id
        if corr_id is None:
            raise RuntimeError("correlation_id must not be None")

        ticket_id = envelope.get("ticket_id")
        if not ticket_id:
            raise RuntimeHostError(
                "list_checkpoints requires ticket_id",
                context=context,
            )

        # Resolve base directory
        base_dir_raw = envelope.get("base_dir")
        if base_dir_raw is not None and not isinstance(base_dir_raw, (str, Path)):
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-checkpoint-list",
                result=ModelCheckpointEffectOutput(
                    success=False,
                    correlation_id=corr_id,
                    error=(
                        f"Invalid base_dir type: expected str or Path, "
                        f"got {type(base_dir_raw).__name__}"
                    ),
                ),
            )
        base_dir = Path(str(base_dir_raw)) if base_dir_raw else _DEFAULT_BASE_DIR

        if not base_dir.is_absolute():
            raise RuntimeHostError(
                f"base_dir must be an absolute path, got: {base_dir}",
                context=context,
            )
        if ".." in base_dir.parts:
            raise RuntimeHostError(
                f"base_dir must not contain '..' components, got: {base_dir}",
                context=context,
            )

        ticket_dir = base_dir / str(ticket_id)
        if not ticket_dir.resolve().is_relative_to(base_dir.resolve()):
            raise RuntimeHostError(
                "Path traversal detected: ticket_id escapes checkpoint root",
                context=context,
            )
        if not ticket_dir.is_dir():
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-checkpoint-list",
                result=ModelCheckpointEffectOutput(
                    success=True,
                    correlation_id=corr_id,
                    checkpoints=(),
                ),
            )

        # Optionally scope to a specific run
        run_id_raw = envelope.get("run_id")
        if run_id_raw is not None:
            if isinstance(run_id_raw, UUID):
                run_id = run_id_raw
            else:
                try:
                    run_id = UUID(str(run_id_raw))
                except ValueError:
                    return ModelHandlerOutput.for_compute(
                        input_envelope_id=uuid4(),
                        correlation_id=corr_id,
                        handler_id="handler-checkpoint-list",
                        result=ModelCheckpointEffectOutput(
                            success=False,
                            correlation_id=corr_id,
                            error=f"Invalid run_id: not a valid UUID: {run_id_raw!r}",
                        ),
                    )
            scan_dirs = [ticket_dir / str(run_id)]
        else:
            # Cap the number of directories scanned to prevent unbounded I/O.
            _MAX_SCAN_DIRS = 1000
            all_dirs = [d for d in ticket_dir.iterdir() if d.is_dir()]
            if len(all_dirs) > _MAX_SCAN_DIRS:
                logger.warning(
                    "Ticket %s has %d run directories; capping scan at %d",
                    ticket_id,
                    len(all_dirs),
                    _MAX_SCAN_DIRS,
                )
                all_dirs = all_dirs[:_MAX_SCAN_DIRS]
            scan_dirs = all_dirs

        checkpoints: list[ModelCheckpoint] = []
        for scan_dir in scan_dirs:
            if not scan_dir.is_dir():
                continue
            # Guard against symlinks escaping the checkpoint root
            if not scan_dir.resolve().is_relative_to(base_dir.resolve()):
                logger.warning("Skipping escaped path: %s", scan_dir.name)
                continue
            for yaml_file in sorted(scan_dir.glob("phase_*.yaml"), key=attempt_number):
                try:
                    raw_data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                    if not isinstance(raw_data, dict):
                        logger.warning(
                            "Skipping non-mapping checkpoint %s", yaml_file.name
                        )
                        continue
                    checkpoint = ModelCheckpoint.model_validate(raw_data)
                    checkpoints.append(checkpoint)
                except (ValidationError, yaml.YAMLError) as exc:
                    logger.warning(
                        "Skipping corrupt checkpoint %s: %s",
                        yaml_file.name,
                        str(exc)[:200],
                    )

        logger.info(
            "Listed %d checkpoints for ticket=%s",
            len(checkpoints),
            ticket_id,
        )

        result = ModelCheckpointEffectOutput(
            success=True,
            correlation_id=corr_id,
            checkpoints=tuple(checkpoints),
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-checkpoint-list",
            result=result,
        )


__all__: list[str] = ["HandlerCheckpointList"]
