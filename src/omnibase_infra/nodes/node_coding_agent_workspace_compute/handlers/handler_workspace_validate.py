# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent workspace COMPUTE handler (OMN-13247, plan §5.5).

Pure pre-flight workspace safety. Deterministic, no I/O beyond pure path
resolution math (``Path.resolve`` strict=False does not touch the filesystem for
the components it normalizes here — the EFFECT owns git/clean-tree reads). The
handler folds a ``ModelWorkspaceValidateCommand`` into a
``ModelWorkspaceValidateResult`` and returns it directly (def-B, OMN-14355) —
the shared runtime adapter supplies the typed command and folds the bare
result into ``output_events``.

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

from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.coding_agent.model_workspace_validate_command import (
    ModelWorkspaceValidateCommand,
)
from omnibase_infra.models.coding_agent.model_workspace_validate_result import (
    ModelWorkspaceValidateResult,
)

HANDLER_ID = "coding-agent-workspace-compute"


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
        self, command: ModelWorkspaceValidateCommand
    ) -> ModelWorkspaceValidateResult:
        """Validate the workspace and return the verdict directly (def-B).

        The shared runtime adapter (``omnibase_infra.runtime.auto_wiring
        .handler_wiring``) supplies the already-validated typed command and
        folds the bare ``ModelWorkspaceValidateResult`` return into
        ``output_events`` itself — no envelope wrapping/unwrapping happens in
        the handler (OMN-14355 canonical shape, definition B).
        """
        return validate_workspace(command)


__all__: list[str] = [
    "HANDLER_ID",
    "HandlerWorkspaceValidate",
    "validate_workspace",
]
