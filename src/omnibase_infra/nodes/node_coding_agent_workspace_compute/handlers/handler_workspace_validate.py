# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent workspace COMPUTE handler (OMN-13247, plan §5.5).

Pure pre-flight workspace safety. Deterministic, no I/O beyond pure path
resolution math (``Path.resolve`` strict=False does not touch the filesystem for
the components it normalizes here — the EFFECT owns git/clean-tree reads). The
handler folds a ``ModelWorkspaceValidateCommand`` into a
``ModelWorkspaceValidateResult`` and returns it via ``for_compute``.

Checks (all must pass to be valid):
  1. allowed-root resolution — the resolved path lives under an allowed root.
  2. symlink-escape rejection — a resolved path that escapes every allowed root
     (e.g. via ``..`` or a symlink target outside the root) is rejected.
  3. sandbox/write-mode coherence — a READ_ONLY sandbox is always coherent; the
     command shape itself carries the write posture, so an empty allowed_roots
     set is incoherent (nothing is permitted) and is rejected.
  4. command-shape validation — non-empty prompt, non-empty workspace_path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.coding_agent.model_workspace_validate_command import (
    ModelWorkspaceValidateCommand,
)
from omnibase_infra.models.coding_agent.model_workspace_validate_result import (
    ModelWorkspaceValidateResult,
)

HANDLER_ID = "coding-agent-workspace-compute"

# Dispatch payloads are coerced at runtime; the protocol entry is generic over the
# envelope payload type (ProtocolMessageHandler.handle(ModelEventEnvelope[T])).
T = TypeVar("T")


def validate_workspace(
    command: ModelWorkspaceValidateCommand,
) -> ModelWorkspaceValidateResult:
    """Pure workspace pre-flight check. Returns a deterministic verdict."""
    if not command.prompt.strip():
        return ModelWorkspaceValidateResult(
            correlation_id=command.correlation_id,
            valid=False,
            rejection_reason="empty prompt",
        )

    if not command.workspace_path.strip():
        return ModelWorkspaceValidateResult(
            correlation_id=command.correlation_id,
            valid=False,
            rejection_reason="empty workspace_path",
        )

    if not command.allowed_roots:
        return ModelWorkspaceValidateResult(
            correlation_id=command.correlation_id,
            valid=False,
            rejection_reason="no allowed roots configured; nothing is permitted",
        )

    # Resolve symlinks/.. without requiring existence; the resolved real path is
    # what we test against the allowed roots so a symlink/.. escape is caught.
    resolved = Path(command.workspace_path).resolve()
    resolved_roots = [Path(r).resolve() for r in command.allowed_roots]

    under_allowed_root = any(
        resolved == root or root in resolved.parents for root in resolved_roots
    )
    if not under_allowed_root:
        return ModelWorkspaceValidateResult(
            correlation_id=command.correlation_id,
            valid=False,
            resolved_path=str(resolved),
            rejection_reason=(
                "workspace path resolves outside every allowed root "
                "(symlink-escape or traversal rejected)"
            ),
        )

    return ModelWorkspaceValidateResult(
        correlation_id=command.correlation_id,
        valid=True,
        resolved_path=str(resolved),
    )


class HandlerWorkspaceValidate:
    """Pure compute: fold a workspace-validate command into a verdict."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, envelope: ModelEventEnvelope[T]
    ) -> ModelHandlerOutput[ModelWorkspaceValidateResult]:
        """Validate the workspace and return the verdict as a compute result."""
        command = _coerce_command(envelope.payload)
        result = validate_workspace(command)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=envelope.envelope_id,
            correlation_id=(
                envelope.correlation_id or command.correlation_id or uuid4()
            ),
            handler_id=HANDLER_ID,
            result=result,
        )


def _coerce_command(payload: object) -> ModelWorkspaceValidateCommand:
    """Coerce the dispatched payload into a ``ModelWorkspaceValidateCommand``."""
    if isinstance(payload, ModelWorkspaceValidateCommand):
        return payload
    if isinstance(payload, Mapping):
        return ModelWorkspaceValidateCommand.model_validate(dict(payload))
    if hasattr(payload, "model_dump"):
        return ModelWorkspaceValidateCommand.model_validate(
            payload.model_dump(mode="json")
        )
    raise TypeError(
        "workspace-validate payload must be ModelWorkspaceValidateCommand or a "
        f"mapping; got {type(payload).__name__}"
    )


__all__: list[str] = [
    "HANDLER_ID",
    "HandlerWorkspaceValidate",
    "validate_workspace",
]
