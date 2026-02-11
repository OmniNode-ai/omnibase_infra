# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pure COMPUTE handler for checkpoint structural validation.

Validates data consistency, path normalization, commit SHA format, and
phase-payload agreement — with zero filesystem I/O.

Ticket: OMN-2143
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import ValidationError

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.checkpoint.model_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    ModelCheckpoint,
)
from omnibase_infra.nodes.node_checkpoint_validate_compute.models.model_checkpoint_validate_output import (
    ModelCheckpointValidateOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
"""Pattern matching a valid (possibly abbreviated) git commit SHA."""


class HandlerCheckpointValidate:
    """Pure validation of checkpoint data — no I/O, fully deterministic.

    Classification: ``COMPUTE_HANDLER`` + ``COMPUTE`` — this is a pure compute
    handler.  Effect handlers use ``NODE_HANDLER`` + ``EFFECT`` instead.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler."""
        self._initialized = True
        logger.info("HandlerCheckpointValidate initialized")

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerCheckpointValidate shutdown")

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelCheckpointValidateOutput]:
        """Validate checkpoint data structurally.

        Envelope keys:
            checkpoint: ModelCheckpoint or dict to validate.
            correlation_id: UUID for tracing.
        """
        correlation_id_raw = envelope.get("correlation_id")
        corr_id = (
            correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else UUID("00000000-0000-0000-0000-000000000000")
        )
        input_envelope_id = uuid4()

        raw_checkpoint = envelope.get("checkpoint")

        # ── Step 0: Deserialize if needed ────────────────────────────
        if isinstance(raw_checkpoint, dict):
            try:
                checkpoint = ModelCheckpoint.model_validate(raw_checkpoint)
            except ValidationError as exc:
                errors = tuple(
                    f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                    for e in exc.errors()
                )
                return ModelHandlerOutput.for_compute(
                    input_envelope_id=input_envelope_id,
                    correlation_id=corr_id,
                    handler_id="handler-checkpoint-validate",
                    result=ModelCheckpointValidateOutput(
                        is_valid=False,
                        correlation_id=corr_id,
                        errors=errors,
                    ),
                )
        elif isinstance(raw_checkpoint, ModelCheckpoint):
            checkpoint = raw_checkpoint
        else:
            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=corr_id,
                handler_id="handler-checkpoint-validate",
                result=ModelCheckpointValidateOutput(
                    is_valid=False,
                    correlation_id=corr_id,
                    errors=("checkpoint field is missing or has unexpected type",),
                ),
            )

        # ── Validate ─────────────────────────────────────────────────
        result = self.validate(checkpoint, corr_id)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=corr_id,
            handler_id="handler-checkpoint-validate",
            result=result,
        )

    def validate(
        self,
        checkpoint: ModelCheckpoint,
        correlation_id: UUID,
    ) -> ModelCheckpointValidateOutput:
        """Pure validation function — deterministic, no I/O.

        Checks:
            1. Schema version compatibility
            2. Attempt number is positive
            3. No absolute artifact paths
            4. Commit SHAs in repo_commit_map are valid hex
            5. Phase payload ``phase`` matches checkpoint ``phase``
            6. Timestamp is not in the future
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Schema version
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
            warnings.append(
                f"Schema version mismatch: expected {CHECKPOINT_SCHEMA_VERSION}, "
                f"got {checkpoint.schema_version}"
            )

        # 2. Attempt number
        if checkpoint.attempt_number < 1:
            errors.append(
                f"attempt_number must be >= 1, got {checkpoint.attempt_number}"
            )

        # 3. Path normalization: no absolute paths in artifact_paths
        for path_str in checkpoint.artifact_paths:
            if PurePosixPath(path_str).is_absolute():
                errors.append(f"Absolute artifact path forbidden: {path_str}")

        # 4. Commit SHA format in repo_commit_map
        for repo_path, sha in checkpoint.repo_commit_map.items():
            if not _SHA_RE.match(sha):
                errors.append(f"Invalid commit SHA for repo '{repo_path}': '{sha}'")

        # 5. Phase payload consistency
        payload_phase = checkpoint.phase_payload.phase
        if payload_phase != checkpoint.phase.value:
            errors.append(
                f"Phase mismatch: header says '{checkpoint.phase.value}', "
                f"payload says '{payload_phase}'"
            )

        # 6. Timestamp not in the future (with 60s tolerance)
        # Note: ModelCheckpoint._ensure_utc guarantees tzinfo is never None.
        now = datetime.now(UTC)
        if checkpoint.timestamp_utc > now + timedelta(seconds=60):
            warnings.append("timestamp_utc is in the future")

        is_valid = len(errors) == 0

        return ModelCheckpointValidateOutput(
            is_valid=is_valid,
            correlation_id=correlation_id,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )


__all__: list[str] = ["HandlerCheckpointValidate"]
