# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that writes a pipeline checkpoint to disk.

Writes are append-only: each attempt creates a new file with a monotonically
increasing ``attempt_number``.  Existing checkpoint files are never modified.

File layout::

    {base_dir}/{ticket_id}/{run_id}/phase_{N}_{name}_a{attempt}.yaml

Ticket: OMN-2143
"""

from __future__ import annotations

import logging
import os
import tempfile
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
from omnibase_infra.nodes.node_checkpoint_effect.models.model_checkpoint_effect_output import (
    ModelCheckpointEffectOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".claude" / "checkpoints"


def _checkpoint_dir(base_dir: Path, ticket_id: str, run_id: UUID) -> Path:
    """Return the directory for a given ticket + run."""
    return base_dir / ticket_id / str(run_id)


def _checkpoint_filename(phase_number: int, phase_value: str, attempt: int) -> str:
    """Return the canonical filename for a checkpoint attempt."""
    return f"phase_{phase_number}_{phase_value}_a{attempt}.yaml"


def _serialize_checkpoint(checkpoint: ModelCheckpoint) -> dict[str, object]:
    """Serialize a checkpoint to a YAML-safe dict."""
    data: dict[str, object] = checkpoint.model_dump(mode="json")
    return data


class HandlerCheckpointWrite:
    """Writes a phase checkpoint YAML file to disk (append-only).

    Each call writes a new file; existing checkpoints are never overwritten.
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
        logger.info("HandlerCheckpointWrite initialized")

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerCheckpointWrite shutdown")

    async def execute(self, envelope: dict[str, object]) -> ModelHandlerOutput:
        """Write a checkpoint to disk.

        Envelope keys:
            checkpoint: ModelCheckpoint or dict to persist.
            correlation_id: UUID for tracing.
            base_dir: Optional override for the checkpoint root.
        """
        correlation_id_raw = envelope.get("correlation_id")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else None,
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="write_checkpoint",
            target_name="checkpoint_yaml",
        )
        corr_id = context.correlation_id
        if corr_id is None:
            raise RuntimeError("correlation_id must not be None")

        raw_checkpoint = envelope.get("checkpoint")
        if raw_checkpoint is None:
            raise RuntimeHostError(
                "write_checkpoint requires a 'checkpoint' in the envelope",
                context=context,
            )

        # Coerce dict → ModelCheckpoint for validation
        if isinstance(raw_checkpoint, dict):
            try:
                checkpoint = ModelCheckpoint.model_validate(raw_checkpoint)
            except ValidationError as exc:
                raise RuntimeHostError(
                    f"Invalid checkpoint data: {exc.error_count()} validation errors",
                    context=context,
                ) from exc
        elif isinstance(raw_checkpoint, ModelCheckpoint):
            checkpoint = raw_checkpoint
        else:
            raise RuntimeHostError(
                f"Unexpected checkpoint type: {type(raw_checkpoint).__name__}",
                context=context,
            )

        # Resolve base directory
        base_dir_raw = envelope.get("base_dir")
        base_dir = Path(str(base_dir_raw)) if base_dir_raw else _DEFAULT_BASE_DIR

        # Build path and guard against path traversal via ticket_id
        target_dir = _checkpoint_dir(base_dir, checkpoint.ticket_id, checkpoint.run_id)
        if not target_dir.resolve().is_relative_to(base_dir.resolve()):
            raise RuntimeHostError(
                "Path traversal detected: ticket_id escapes checkpoint root",
                context=context,
            )
        filename = _checkpoint_filename(
            checkpoint.phase.phase_number,
            checkpoint.phase.value,
            checkpoint.attempt_number,
        )
        target_path = target_dir / filename

        # Validate path normalization: artifact_paths must be relative
        for artifact in checkpoint.artifact_paths:
            if Path(artifact).is_absolute():
                raise RuntimeHostError(
                    f"Absolute artifact path forbidden: {artifact}",
                    context=context,
                )

        # Write (append-only: refuse to overwrite existing checkpoints)
        #
        # Atomic write: content goes to a temp file first, then os.link()
        # atomically creates the target or raises FileExistsError (TOCTOU-safe).
        # This ensures checkpoints are always complete — a crash mid-write
        # leaves only the temp file, which is cleaned up on the next attempt.
        target_dir.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.dump(
            _serialize_checkpoint(checkpoint),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

        tmp_fd = -1
        tmp_path_str = ""
        try:
            tmp_fd, tmp_path_str = tempfile.mkstemp(
                dir=str(target_dir), suffix=".tmp", prefix=".ckpt_"
            )
            os.write(tmp_fd, yaml_content.encode("utf-8"))
            os.fsync(tmp_fd)
            os.close(tmp_fd)
            tmp_fd = -1

            # Atomic link: fails with FileExistsError if target already exists.
            # Hard link requires same filesystem; mkstemp(dir=target_dir) ensures this.
            os.link(tmp_path_str, str(target_path))
            Path(tmp_path_str).unlink()
            tmp_path_str = ""
        except FileExistsError as exc:
            raise RuntimeHostError(
                f"Checkpoint already exists: {filename} — "
                f"increment attempt_number to create a new checkpoint",
                context=context,
            ) from exc
        finally:
            if tmp_fd >= 0:
                os.close(tmp_fd)
            if tmp_path_str and Path(tmp_path_str).exists():
                Path(tmp_path_str).unlink()

        # Return relative path from base_dir
        relative_path = str(target_path.relative_to(base_dir))

        logger.info(
            "Checkpoint written: %s (phase=%s, attempt=%d)",
            relative_path,
            checkpoint.phase.value,
            checkpoint.attempt_number,
        )

        result = ModelCheckpointEffectOutput(
            success=True,
            correlation_id=corr_id,
            checkpoint_path=relative_path,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-checkpoint-write",
            result=result,
        )


__all__: list[str] = ["HandlerCheckpointWrite"]
