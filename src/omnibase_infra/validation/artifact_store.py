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
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Default root for artifact storage
DEFAULT_ARTIFACT_ROOT = Path.home() / ".claude" / "validation"


class ModelArtifactStoreConfig(BaseModel):
    """Configuration for the artifact store.

    Attributes:
        root_dir: Root directory for artifact storage.
        create_dirs: Whether to create directories on write.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    root_dir: str = Field(
        default=str(DEFAULT_ARTIFACT_ROOT),
        description="Root directory for artifact storage.",
    )
    create_dirs: bool = Field(
        default=True,
        description="Whether to create directories on write.",
    )


class ArtifactStore:
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
            filename: Filename within the artifacts directory.
            content: File content (string or bytes).

        Returns:
            Path to the written artifact file.
        """
        artifacts = self.artifacts_dir(candidate_id, run_id)
        if self._create_dirs:
            artifacts.mkdir(parents=True, exist_ok=True)

        artifact_path = artifacts / filename
        # Ensure parent dir exists for nested filenames (e.g., "logs/check.log")
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
        result: dict[str, Any] = yaml.safe_load(content)
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
        result: dict[str, Any] = yaml.safe_load(content)
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
        target = self.run_dir(candidate_id, run_id)

        # Use relative target for portability
        try:
            relative_target = Path("..") / str(candidate_id) / str(run_id)
        except ValueError:
            relative_target = target

        # Remove existing symlink if present
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()

        symlink_path.symlink_to(relative_target)
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
    "ArtifactStore",
    "ModelArtifactStoreConfig",
    "DEFAULT_ARTIFACT_ROOT",
]
