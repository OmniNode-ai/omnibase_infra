# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler that writes RRH result artifacts and manages symlinks.

Artifact Layout:
    {output_dir}/
    ├── artifacts/
    │   └── rrh_{correlation_id}_{timestamp}.json
    ├── latest_by_ticket/
    │   └── {ticket_id} -> ../artifacts/rrh_...json
    └── latest_by_repo/
        └── {repo_name} -> ../artifacts/rrh_...json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from uuid import uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.nodes.node_rrh_storage_effect.models.model_rrh_storage_request import (
    ModelRRHStorageRequest,
)
from omnibase_infra.nodes.node_rrh_storage_effect.models.model_rrh_storage_result import (
    ModelRRHStorageResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)


class HandlerRRHStorageWrite:
    """Write RRH result JSON and update convenience symlinks.

    Attributes:
        handler_type: ``INFRA_HANDLER``
        handler_category: ``EFFECT``
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelRRHStorageRequest) -> ModelRRHStorageResult:
        """Write the RRH result and manage symlinks.

        Args:
            request: Storage request with result and output directory.

        Returns:
            ``ModelRRHStorageResult`` with paths and success status.
        """
        try:
            base = Path(request.output_dir)
            error_context = ModelInfraErrorContext.with_correlation(
                correlation_id=request.correlation_id,
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="validate_output_dir",
            )
            if not base.is_absolute():
                raise ProtocolConfigurationError(
                    "output_dir must be absolute",
                    context=error_context,
                )
            if ".." in base.parts:
                raise ProtocolConfigurationError(
                    "output_dir must not contain '..' components",
                    context=error_context,
                )
            artifacts_dir = base / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Build timestamped artifact filename.  Microseconds (%f)
            # prevent collisions from rapid evaluations within the same second.
            ts = request.result.evaluated_at.strftime("%Y%m%dT%H%M%S_%f")
            cid = str(request.correlation_id)[:8]
            filename = f"rrh_{cid}_{ts}.json"
            artifact_path = artifacts_dir / filename

            # Serialize result to JSON.
            payload = request.result.model_dump(mode="json")
            artifact_path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )

            # Manage symlinks.
            ticket_symlink = ""
            repo_symlink = ""

            if request.result.ticket_id:
                ticket_symlink = self._update_symlink(
                    base / "latest_by_ticket",
                    request.result.ticket_id,
                    artifact_path,
                )

            if request.result.repo_name:
                repo_symlink = self._update_symlink(
                    base / "latest_by_repo",
                    request.result.repo_name,
                    artifact_path,
                )

            return ModelRRHStorageResult(
                artifact_path=str(artifact_path),
                ticket_symlink=ticket_symlink,
                repo_symlink=repo_symlink,
                success=True,
                correlation_id=request.correlation_id,
            )

        except Exception as exc:
            logger.warning("RRH storage write failed: %s", sanitize_error_message(exc))
            return ModelRRHStorageResult(
                artifact_path="",
                success=False,
                error=sanitize_error_message(exc),
                correlation_id=request.correlation_id,
            )

    @staticmethod
    def _update_symlink(
        symlink_dir: Path,
        name: str,
        target: Path,
    ) -> str:
        """Create or update a symlink pointing to the artifact.

        Args:
            symlink_dir: Directory containing symlinks (e.g. latest_by_ticket/).
            name: Symlink name (e.g. ticket ID or repo name).
            target: Absolute path to the artifact file.

        Returns:
            String path to the created symlink (empty on failure).
        """
        try:
            symlink_dir.mkdir(parents=True, exist_ok=True)
            # Sanitize name to prevent path traversal.  Strip all directory
            # components first, then apply a strict allowlist to eliminate any
            # remaining dangerous characters (e.g., NUL bytes, backslashes).
            safe_name = re.sub(r"[^a-zA-Z0-9_.\-]", "_", Path(name).name or "_")
            if safe_name in (".", ".."):
                safe_name = "_"
            link_path = symlink_dir / safe_name
            # Compute relative target for portable symlinks.
            rel_target = Path("..") / "artifacts" / target.name
            # Atomic symlink replacement: create temp link, then Path.replace()
            # over the final path.  Path.replace() delegates to os.replace()
            # which is atomic on POSIX.
            tmp_link = symlink_dir / f".tmp_{safe_name}_{uuid4().hex[:8]}"
            try:
                tmp_link.symlink_to(rel_target)
                tmp_link.replace(link_path)
            except OSError:
                # Fallback: non-atomic if replace fails (e.g., cross-device).
                tmp_link.unlink(missing_ok=True)
                if link_path.is_symlink() or link_path.exists():
                    link_path.unlink()
                link_path.symlink_to(rel_target)
            return str(link_path)
        except OSError as exc:
            logger.debug(
                "Failed to update symlink %s: %s", name, sanitize_error_message(exc)
            )
            return ""


__all__: list[str] = ["HandlerRRHStorageWrite"]
