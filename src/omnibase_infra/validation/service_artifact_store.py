# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Artifact storage for the validation pipeline.

Manages the on-disk directory structure for validation artifacts:

    ~/.claude/validation/
    |-- {candidate_id}/
    |   |-- plan.yaml
    |   |-- {validation_run_id}/
    |   |   |-- result.yaml
    |   |   |-- verdict.yaml
    |   |   |-- attribution.yaml
    |   |   |-- artifacts/
    |   |       |-- junit.xml
    |   |       |-- coverage.json
    |   |       |-- logs/
    |   |-- ...
    |-- latest_by_pattern/
        |-- {pattern_id} -> ../{candidate_id}/{validation_run_id}/

The ``latest_by_pattern`` directory contains symlinks pointing to the
most recent validation run for each pattern, enabling quick lookup
without scanning all candidate directories.

Ticket: OMN-2151
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def _default_artifact_root() -> Path:
    """Lazily resolve the default artifact root directory.

    Avoids calling ``Path.home()`` at module import time, which can
    fail in containers or environments where ``HOME`` is unset.

    Returns:
        Default artifact root path (``~/.claude/validation``).
    """
    return Path.home() / ".claude" / "validation"


class ModelArtifactStoreConfig(BaseModel):
    """Configuration for the artifact store.

    Attributes:
        root_dir: Root directory for artifact storage.
        create_dirs: Whether to create directories on write.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    root_dir: str = Field(
        default_factory=lambda: str(_default_artifact_root()),
        description="Root directory for artifact storage. "
        "Defaults to ~/.claude/validation when not specified.",
    )
    create_dirs: bool = Field(
        default=True,
        description="Whether to create directories on write.",
    )


class ServiceArtifactStore:
    """Manages validation artifacts on disk.

    Provides methods to store and retrieve validation plans, results,
    verdicts, and attribution data. Manages the ``latest_by_pattern``
    symlinks for quick pattern-based lookups.

    Attributes:
        root: Root directory for all artifact storage.
    """

    def __init__(self, config: ModelArtifactStoreConfig | None = None) -> None:
        """Initialize the artifact store.

        Args:
            config: Store configuration. Uses defaults if None.
        """
        config = config or ModelArtifactStoreConfig()
        self.root = Path(config.root_dir)
        self._create_dirs = config.create_dirs

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path_within(filename: str, allowed_root: Path) -> Path:
        """Validate that *filename* does not escape *allowed_root*.

        Prevents path-traversal attacks where a caller-supplied filename
        such as ``../../etc/passwd`` could escape the intended directory,
        **including attacks via symlinks** inside the allowed root that
        point to locations outside it.

        Uses :meth:`Path.resolve` with ``strict=False`` (Python 3.6+) to
        resolve symlinks without requiring the target path to exist.  This
        catches symlink escapes even for paths that haven't been created
        yet, unlike :func:`os.path.normpath` which only collapses ``..``
        segments textually.

        Args:
            filename: Caller-supplied filename (may include subdirectory
                components like ``logs/check.log``).
            allowed_root: The directory that the resulting path must
                reside within.  Must already be resolved / absolute.

        Returns:
            The resolved absolute path, guaranteed to be within
            *allowed_root*.

        Raises:
            ValueError: If the resulting path escapes *allowed_root*.
        """
        resolved_root = allowed_root.resolve()
        # Build the candidate path and resolve symlinks + collapse ".."
        # segments.  ``strict=False`` allows the target to not yet exist
        # while still resolving any symlink components that *do* exist.
        candidate = (resolved_root / filename).resolve(strict=False)

        if not candidate.is_relative_to(resolved_root):
            raise ValueError(
                f"Path traversal detected: filename {filename!r} "
                f"resolves to {candidate}, "
                f"which is outside {resolved_root}"
            )

        # Secondary defence: if the candidate (or any component) already
        # exists on disk, re-resolve to catch any symlinks created
        # between the first resolve and now (TOCTOU mitigation).
        if candidate.exists() or candidate.is_symlink():
            real_candidate = candidate.resolve()
            if not real_candidate.is_relative_to(resolved_root):
                raise ValueError(
                    f"Symlink escape detected: filename {filename!r} "
                    f"resolves through symlinks to {real_candidate}, "
                    f"which is outside {resolved_root}"
                )

        return candidate

    # ------------------------------------------------------------------
    # Directory structure helpers
    # ------------------------------------------------------------------

    def candidate_dir(self, candidate_id: UUID) -> Path:
        """Return the directory for a given candidate.

        Args:
            candidate_id: Unique candidate identifier.

        Returns:
            Path to the candidate's artifact directory.
        """
        return self.root / str(candidate_id)

    def run_dir(self, candidate_id: UUID, run_id: UUID) -> Path:
        """Return the directory for a specific validation run.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.

        Returns:
            Path to the run's artifact directory.
        """
        return self.candidate_dir(candidate_id) / str(run_id)

    def artifacts_dir(self, candidate_id: UUID, run_id: UUID) -> Path:
        """Return the artifacts subdirectory for a validation run.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.

        Returns:
            Path to the artifacts subdirectory.
        """
        return self.run_dir(candidate_id, run_id) / "artifacts"

    def logs_dir(self, candidate_id: UUID, run_id: UUID) -> Path:
        """Return the logs subdirectory for a validation run.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.

        Returns:
            Path to the logs subdirectory.
        """
        return self.artifacts_dir(candidate_id, run_id) / "logs"

    def latest_by_pattern_dir(self) -> Path:
        """Return the latest_by_pattern symlink directory.

        Returns:
            Path to the latest_by_pattern directory.
        """
        return self.root / "latest_by_pattern"

    # ------------------------------------------------------------------
    # Ensure directories exist
    # ------------------------------------------------------------------

    def ensure_run_dirs(self, candidate_id: UUID, run_id: UUID) -> Path:
        """Create the full directory tree for a validation run.

        Creates the candidate directory, run directory, artifacts
        subdirectory, logs subdirectory, and latest_by_pattern directory.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.

        Returns:
            Path to the run directory.
        """
        run_path = self.run_dir(candidate_id, run_id)
        if self._create_dirs:
            self.logs_dir(candidate_id, run_id).mkdir(parents=True, exist_ok=True)
            self.latest_by_pattern_dir().mkdir(parents=True, exist_ok=True)
        return run_path

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------

    # ONEX_EXCLUDE: any_type - YAML plan data is heterogeneous dict from yaml.safe_load
    def write_plan(self, candidate_id: UUID, plan_data: dict[str, Any]) -> Path:
        """Write the validation plan to disk.

        The plan is stored at ``{candidate_dir}/plan.yaml`` and is
        shared across all runs for the same candidate.

        Args:
            candidate_id: Unique candidate identifier.
            plan_data: Plan data to serialize as YAML.

        Returns:
            Path to the written plan file.
        """
        cand_dir = self.candidate_dir(candidate_id)
        if self._create_dirs:
            cand_dir.mkdir(parents=True, exist_ok=True)

        plan_path = cand_dir / "plan.yaml"
        plan_path.write_text(
            yaml.dump(plan_data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.debug("Wrote plan to %s", plan_path)
        return plan_path

    # ONEX_EXCLUDE: any_type - YAML result data is heterogeneous dict for yaml.dump
    def write_result(
        self, candidate_id: UUID, run_id: UUID, result_data: dict[str, Any]
    ) -> Path:
        """Write the executor result to disk.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.
            result_data: Result data to serialize as YAML.

        Returns:
            Path to the written result file.
        """
        self.ensure_run_dirs(candidate_id, run_id)
        result_path = self.run_dir(candidate_id, run_id) / "result.yaml"
        result_path.write_text(
            yaml.dump(result_data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.debug("Wrote result to %s", result_path)
        return result_path

    # ONEX_EXCLUDE: any_type - YAML verdict data is heterogeneous dict for yaml.dump
    def write_verdict(
        self, candidate_id: UUID, run_id: UUID, verdict_data: dict[str, Any]
    ) -> Path:
        """Write the verdict to disk.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.
            verdict_data: Verdict data to serialize as YAML.

        Returns:
            Path to the written verdict file.
        """
        self.ensure_run_dirs(candidate_id, run_id)
        verdict_path = self.run_dir(candidate_id, run_id) / "verdict.yaml"
        verdict_path.write_text(
            yaml.dump(verdict_data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.debug("Wrote verdict to %s", verdict_path)
        return verdict_path

    # ONEX_EXCLUDE: any_type - YAML attribution data is heterogeneous dict for yaml.dump
    def write_attribution(
        self, candidate_id: UUID, run_id: UUID, attribution_data: dict[str, Any]
    ) -> Path:
        """Write attribution data to disk.

        Attribution tracks which agent/tool produced the validation
        results and the correlation chain for traceability.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.
            attribution_data: Attribution data to serialize as YAML.

        Returns:
            Path to the written attribution file.
        """
        self.ensure_run_dirs(candidate_id, run_id)
        attr_path = self.run_dir(candidate_id, run_id) / "attribution.yaml"
        attr_path.write_text(
            yaml.dump(attribution_data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.debug("Wrote attribution to %s", attr_path)
        return attr_path

    def write_artifact(
        self, candidate_id: UUID, run_id: UUID, filename: str, content: str | bytes
    ) -> Path:
        """Write an arbitrary artifact file.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.
            filename: Filename within the artifacts directory.  Must not
                contain path components that escape the artifacts
                directory (e.g. ``../../etc/passwd``).
            content: File content (string or bytes).

        Returns:
            Path to the written artifact file.

        Raises:
            ValueError: If *filename* resolves to a path outside the
                artifacts directory (path traversal).
        """
        artifacts = self.artifacts_dir(candidate_id, run_id)
        if self._create_dirs:
            artifacts.mkdir(parents=True, exist_ok=True)

        # Validate BEFORE creating any subdirectories to prevent
        # traversal attacks from creating directories outside the
        # artifacts tree.
        artifact_path = self._validate_path_within(filename, artifacts)

        # Ensure parent dir exists for nested filenames (e.g., "logs/check.log").
        # Respect _create_dirs: if disabled, let the subsequent write fail
        # naturally when the parent directory does not exist.
        if self._create_dirs:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            artifact_path.write_bytes(content)
        else:
            artifact_path.write_text(content, encoding="utf-8")

        logger.debug("Wrote artifact to %s", artifact_path)
        return artifact_path

    # ------------------------------------------------------------------
    # Read artifacts
    # ------------------------------------------------------------------

    # ONEX_EXCLUDE: any_type - YAML plan data is heterogeneous dict from yaml.safe_load
    def read_plan(self, candidate_id: UUID) -> dict[str, Any] | None:
        """Read the validation plan from disk.

        Args:
            candidate_id: Unique candidate identifier.

        Returns:
            Plan data as a dict, or None if not found.
        """
        plan_path = self.candidate_dir(candidate_id) / "plan.yaml"
        if not plan_path.is_file():
            return None
        content = plan_path.read_text(encoding="utf-8")
        # ONEX_EXCLUDE: any_type - yaml.safe_load returns heterogeneous dict
        result: dict[str, Any] | None = yaml.safe_load(content)
        # Return None for empty files (yaml.safe_load returns None for empty
        # content), preserving the distinction: None = file missing or empty,
        # dict = file has content.
        return result

    # ONEX_EXCLUDE: any_type - YAML verdict data is heterogeneous dict from yaml.safe_load
    def read_verdict(self, candidate_id: UUID, run_id: UUID) -> dict[str, Any] | None:
        """Read the verdict from disk.

        Args:
            candidate_id: Unique candidate identifier.
            run_id: Unique validation run identifier.

        Returns:
            Verdict data as a dict, or None if not found.
        """
        verdict_path = self.run_dir(candidate_id, run_id) / "verdict.yaml"
        if not verdict_path.is_file():
            return None
        content = verdict_path.read_text(encoding="utf-8")
        # ONEX_EXCLUDE: any_type - yaml.safe_load returns heterogeneous dict
        result: dict[str, Any] | None = yaml.safe_load(content)
        # Return None for empty files (yaml.safe_load returns None for empty
        # content), preserving the distinction: None = file missing or empty,
        # dict = file has content.
        return result

    # ------------------------------------------------------------------
    # Symlink management (latest_by_pattern)
    # ------------------------------------------------------------------

    def update_latest_symlink(
        self,
        pattern_id: UUID,
        candidate_id: UUID,
        run_id: UUID,
    ) -> Path:
        """Update the latest_by_pattern symlink for a pattern.

        Creates or replaces the symlink at
        ``latest_by_pattern/{pattern_id}`` to point to the specified
        validation run directory.

        Args:
            pattern_id: Pattern identifier.
            candidate_id: Candidate identifier.
            run_id: Validation run identifier.

        Returns:
            Path to the symlink.
        """
        symlink_dir = self.latest_by_pattern_dir()
        if self._create_dirs:
            symlink_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = symlink_dir / str(pattern_id)

        # Use relative target for portability
        relative_target = Path("..") / str(candidate_id) / str(run_id)

        # Atomic symlink update: create at a temporary name, then rename.
        # This avoids the TOCTOU race of check-unlink-create where two
        # concurrent processes could interfere with each other.
        try:
            # Create a temp file name in the same directory so rename is
            # guaranteed to be on the same filesystem (required for
            # atomic os.rename / os.replace on POSIX).
            fd, tmp_path_str = tempfile.mkstemp(
                dir=str(symlink_dir), prefix=".symlink_tmp_"
            )
            os.close(fd)
            tmp_path = Path(tmp_path_str)
            # mkstemp creates a regular file; remove it so we can
            # create a symlink at the same path.
            tmp_path.unlink()

            tmp_path.symlink_to(relative_target)
            # Path.rename is atomic on POSIX when source and target are
            # on the same filesystem.  It replaces any existing entry
            # at symlink_path in a single operation.
            tmp_path.rename(symlink_path)
        except OSError:
            # Graceful degradation: symlink creation may fail on Windows
            # without developer mode / SeCreateSymbolicLinkPrivilege, or
            # on filesystems that do not support symlinks.  Log a warning
            # with exception info and return the target path so callers
            # can still locate the run directory.
            logger.warning(
                "Failed to create symlink %s -> %s (symlinks may not be "
                "supported on this platform); falling back to target path.",
                symlink_path,
                relative_target,
                exc_info=True,
            )
            # Clean up the temp symlink if it was created but rename failed.
            try:
                if tmp_path.is_symlink() or tmp_path.exists():
                    tmp_path.unlink()
            except (OSError, UnboundLocalError):
                pass
            return self.run_dir(candidate_id, run_id)

        logger.debug(
            "Updated latest symlink %s -> %s",
            symlink_path,
            relative_target,
        )
        return symlink_path

    def resolve_latest(self, pattern_id: UUID) -> Path | None:
        """Resolve the latest validation run directory for a pattern.

        Args:
            pattern_id: Pattern identifier.

        Returns:
            Resolved path to the latest run directory, or None if
            no symlink exists.
        """
        symlink_path = self.latest_by_pattern_dir() / str(pattern_id)
        if not symlink_path.is_symlink():
            return None
        return symlink_path.resolve()

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_candidates(self) -> list[str]:
        """List all candidate IDs with stored artifacts.

        Returns:
            List of candidate ID strings.
        """
        if not self.root.is_dir():
            return []
        return [
            d.name
            for d in sorted(self.root.iterdir())
            if d.is_dir() and d.name != "latest_by_pattern"
        ]

    def list_runs(self, candidate_id: UUID) -> list[str]:
        """List all run IDs for a candidate.

        Args:
            candidate_id: Candidate identifier.

        Returns:
            List of run ID strings.
        """
        cand_dir = self.candidate_dir(candidate_id)
        if not cand_dir.is_dir():
            return []
        return [d.name for d in sorted(cand_dir.iterdir()) if d.is_dir()]


__all__: list[str] = [
    "ServiceArtifactStore",
    "ModelArtifactStoreConfig",
]
