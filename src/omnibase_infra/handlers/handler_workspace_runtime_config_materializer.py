# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Materialize workspace runtime configuration from git objects (OMN-19212).

Only the object database is read: the repository's index and shared working
tree are never consulted or changed. The config and its source sidecar live
under workspace state, with the sidecar written last to record attribution.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.models.workspace import (
    ModelMaterializedWorkspaceRuntimeConfig,
    ModelWorkspaceRuntimeConfigMaterialization,
    ModelWorkspaceRuntimeConfigSidecar,
)
from omnibase_infra.utils.util_atomic_file import write_atomic_bytes

MATERIALIZED_CONTRACTS_RELATIVE_PATH = Path(".onex_state") / "workspace-runtime"
MATERIALIZED_SIDECAR_NAME = "runtime_config.source.json"
SOURCE_REF = "origin/main"
SOURCE_PATH_IN_REPO = "config/onex/runtime/runtime_config.yaml"
STALE_AFTER = timedelta(hours=24)

logger = logging.getLogger(__name__)


class HandlerWorkspaceRuntimeConfigMaterializer:
    """Refresh and read the workspace's attributable tier-1 runtime config."""

    def materialize(
        self, workspace_root: Path
    ) -> ModelWorkspaceRuntimeConfigMaterialization:
        """Read origin/main's config without touching the index or working tree.

        Git failures return an unsuccessful outcome before any copy is written,
        preserving an earlier successful materialization. Each file is replaced
        atomically, YAML first and provenance last, even if the SHA is unchanged.
        """
        action = "locate workspace repository"
        try:
            root = workspace_root.resolve()
            toplevel = self._git(root, "rev-parse", "--show-toplevel")
            if Path(os.fsdecode(toplevel).strip()).resolve() != root:
                return ModelWorkspaceRuntimeConfigMaterialization(
                    ok=False, detail=f"Workspace root {root} is not the repository root"
                )
            action = f"resolve {SOURCE_REF}"
            sha = (
                self._git(root, "rev-parse", "--verify", f"{SOURCE_REF}^{{commit}}")
                .decode("ascii")
                .strip()
            )
            action = f"read {SOURCE_REF}:{SOURCE_PATH_IN_REPO}"
            config_bytes = self._git(
                root, "show", f"{SOURCE_REF}:{SOURCE_PATH_IN_REPO}"
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ModelWorkspaceRuntimeConfigMaterialization(
                ok=False, detail=" ".join(f"Could not {action}: {exc}".split())
            )

        sidecar = ModelWorkspaceRuntimeConfigSidecar(
            source_ref=SOURCE_REF,
            source_path=SOURCE_PATH_IN_REPO,
            sha=sha,
            materialized_at=datetime.now(UTC),
        )
        contracts_dir = root / MATERIALIZED_CONTRACTS_RELATIVE_PATH
        runtime_dir = contracts_dir / "runtime"
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            write_atomic_bytes(runtime_dir / "runtime_config.yaml", config_bytes)
            write_atomic_bytes(
                runtime_dir / MATERIALIZED_SIDECAR_NAME,
                sidecar.model_dump_json().encode("utf-8"),
            )
        except (OSError, InfraConnectionError) as exc:
            return ModelWorkspaceRuntimeConfigMaterialization(
                ok=False,
                detail=" ".join(
                    f"Could not write workspace runtime config: {exc}".split()
                ),
            )
        return ModelWorkspaceRuntimeConfigMaterialization(ok=True, sha=sha)

    def read(
        self, workspace_root: Path, *, now: datetime | None = None
    ) -> ModelMaterializedWorkspaceRuntimeConfig | None:
        """Read a copy only when its config and valid source sidecar both exist.

        An unreadable sidecar is warned about and treated as an absent copy.
        Stale copies remain available, with freshness recorded in the outcome.
        """
        contracts_dir = workspace_root / MATERIALIZED_CONTRACTS_RELATIVE_PATH
        config_path = contracts_dir / "runtime" / "runtime_config.yaml"
        sidecar_path = config_path.parent / MATERIALIZED_SIDECAR_NAME
        if not config_path.is_file() or not sidecar_path.is_file():
            return None
        try:
            sidecar = ModelWorkspaceRuntimeConfigSidecar.model_validate_json(
                sidecar_path.read_bytes()
            )
        except (OSError, ValidationError) as exc:
            logger.warning(
                "Unreadable workspace runtime config sidecar %s: %s", sidecar_path, exc
            )
            return None
        return ModelMaterializedWorkspaceRuntimeConfig(
            contracts_dir=contracts_dir,
            config_path=config_path,
            sha=sidecar.sha,
            materialized_at=sidecar.materialized_at,
            stale=(now or datetime.now(UTC)) - sidecar.materialized_at > STALE_AFTER,
        )

    @staticmethod
    def _git(workspace_root: Path, *args: str) -> bytes:
        """Run a bounded object-database command with optional locks disabled."""
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(workspace_root), *args],
            timeout=30,
            capture_output=True,
            check=True,
        )
        return result.stdout
