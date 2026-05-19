# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for runtime source-hash attestation — OMN-9139.

Compares the ``runtime_source_hash`` embedded in a
``ModelRuntimeBootedEvent`` against the current ``main`` HEAD of the
repo via ``git ls-remote``.

Drift policy:
    * ``runtime_source_hash`` is ``"unknown"`` or empty → verdict ``unknown_hash``,
      friction always emitted (treat as infinite distance).
    * Hash matches main HEAD exactly → verdict ``compliant``.
    * Hash is reachable but behind HEAD by ≤ drift_threshold commits →
      verdict ``compliant`` (within tolerance).
    * Hash is behind HEAD by > drift_threshold commits → verdict ``drifted``,
      friction emitted.

The ``commit_distance`` field in the result is ``-1`` when the distance
cannot be computed (e.g. ``git ls-remote`` unavailable in CI sandbox).
In that case the handler compares only whether the short hash matches
main HEAD; if it does not match, verdict is ``drifted``.

Friction files are written to ``.onex_state/friction/`` relative to the
process working directory (or the path supplied via ``friction_dir``
constructor arg for testing).
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.model_runtime_booted_event import (
    ModelRuntimeBootedEvent,
)

logger = logging.getLogger(__name__)

# Sentinel values that indicate a build without attestation
_INVALID_HASHES: frozenset[str] = frozenset({"unknown", "", "dev"})

_DEFAULT_REPO_URL = "https://github.com/OmniNode-ai/omnibase_infra.git"
_DEFAULT_DRIFT_THRESHOLD = 5
_DEFAULT_FRICTION_DIR = Path(".onex_state/friction")


class ModelSourceAttestationResult(BaseModel):
    """Result of a source-hash attestation check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    container_ref: str = Field(..., description="Container that was checked.")
    runtime_source_hash: str = Field(..., description="Hash embedded in the image.")
    verdict: str = Field(
        ...,
        description=(
            "One of: 'compliant' | 'drifted' | 'unknown_hash'. "
            "'unknown_hash' means the build lacked attestation entirely."
        ),
    )
    commit_distance: int = Field(
        default=-1,
        description=(
            "Number of commits between runtime_source_hash and main HEAD. "
            "-1 when distance cannot be computed."
        ),
    )
    friction_path: str | None = Field(
        default=None,
        description="Path to the written friction file, or None if not emitted.",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the attestation check.",
    )


class HandlerSourceAttestation:
    """Compare runtime_source_hash against main HEAD and emit friction on drift.

    Constructor args:
        repo_url: GitHub repo URL to resolve main HEAD via git ls-remote.
        drift_threshold: Max allowed commits behind main before alerting.
        friction_dir: Directory for friction YAML files (default .onex_state/friction).
    """

    def __init__(
        self,
        repo_url: str = _DEFAULT_REPO_URL,
        drift_threshold: int = _DEFAULT_DRIFT_THRESHOLD,
        friction_dir: Path = _DEFAULT_FRICTION_DIR,
    ) -> None:
        self._repo_url = repo_url
        self._drift_threshold = drift_threshold
        self._friction_dir = friction_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attest(self, event: ModelRuntimeBootedEvent) -> ModelSourceAttestationResult:
        """Attest the source hash in *event* and return the verdict.

        This is a synchronous handler — git ls-remote is a short-lived
        subprocess, not a long-running I/O loop.
        """
        runtime_hash = event.runtime_source_hash.strip()

        if not runtime_hash or runtime_hash in _INVALID_HASHES:
            friction_path = self._emit_friction(
                container_ref=event.container_ref,
                runtime_hash=runtime_hash,
                reason="runtime_source_hash is unknown/empty — build lacked attestation",
                commit_distance=-1,
            )
            return ModelSourceAttestationResult(
                container_ref=event.container_ref,
                runtime_source_hash=runtime_hash,
                verdict="unknown_hash",
                commit_distance=-1,
                friction_path=str(friction_path) if friction_path else None,
            )

        main_head = self._resolve_main_head()
        if main_head is None:
            # git ls-remote unavailable — compare short hashes
            logger.warning(
                "git ls-remote unavailable; falling back to short-hash comparison"
            )
            head_short = "unknown"
            distance = -1
        else:
            head_short = main_head[:7]
            distance = self._compute_distance(runtime_hash, main_head)

        runtime_short = runtime_hash[:7]
        is_exact_match = main_head is not None and (
            main_head.startswith(runtime_hash) or runtime_hash.startswith(main_head[:7])
        )

        if is_exact_match or (distance != -1 and distance <= self._drift_threshold):
            return ModelSourceAttestationResult(
                container_ref=event.container_ref,
                runtime_source_hash=runtime_hash,
                verdict="compliant",
                commit_distance=distance,
                friction_path=None,
            )

        # Drifted
        reason = f"runtime_source_hash={runtime_short!r} is " + (
            f"{distance} commits behind main HEAD={head_short!r}"
            if distance != -1
            else f"not at main HEAD={head_short!r} (distance unknown)"
        )
        friction_path = self._emit_friction(
            container_ref=event.container_ref,
            runtime_hash=runtime_hash,
            reason=reason,
            commit_distance=distance,
        )
        return ModelSourceAttestationResult(
            container_ref=event.container_ref,
            runtime_source_hash=runtime_hash,
            verdict="drifted",
            commit_distance=distance,
            friction_path=str(friction_path) if friction_path else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_main_head(self) -> str | None:
        """Return full commit SHA of origin/main, or None on failure."""
        try:
            result = subprocess.run(
                ["git", "ls-remote", self._repo_url, "refs/heads/main"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            # Output format: "<sha>\trefs/heads/main"
            sha = result.stdout.strip().split()[0]
            return sha if len(sha) >= 7 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            logger.debug("git ls-remote failed", exc_info=True)
            return None

    def _compute_distance(self, runtime_hash: str, main_head: str) -> int:
        """Return commit distance between *runtime_hash* and *main_head*.

        Returns -1 if the local repo is not available or the hash is not found.
        This is best-effort; the git subprocess may not be present in all
        environments. Callers must handle -1 gracefully.
        """
        if main_head.startswith(runtime_hash) or runtime_hash.startswith(main_head[:7]):
            return 0
        try:
            result = subprocess.run(
                [
                    "git",
                    "rev-list",
                    "--count",
                    f"{runtime_hash}..{main_head}",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return int(result.stdout.strip())
        except (OSError, subprocess.TimeoutExpired, ValueError):
            logger.debug("git rev-list distance calculation failed", exc_info=True)
        return -1

    def _emit_friction(
        self,
        *,
        container_ref: str,
        runtime_hash: str,
        reason: str,
        commit_distance: int,
    ) -> Path | None:
        """Write a friction YAML file and return its path."""
        try:
            self._friction_dir.mkdir(parents=True, exist_ok=True)
            slug = container_ref.replace("/", "-").replace(":", "-")
            friction_path = self._friction_dir / f"runtime-source-drift-{slug}.yaml"
            payload: dict[str, object] = {
                "type": "runtime_source_drift",
                "container": container_ref,
                "runtime_source_hash": runtime_hash,
                "reason": reason,
                "commit_distance": commit_distance,
                "ticket": "OMN-9139",
                "emitted_at": datetime.now(UTC).isoformat(),
            }
            friction_path.write_text(
                yaml.dump(payload, sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
            logger.warning(
                "Source drift friction emitted: container=%s reason=%s path=%s",
                container_ref,
                reason,
                friction_path,
            )
            return friction_path
        except OSError:
            logger.exception(
                "Failed to write friction file for container=%s", container_ref
            )
            return None


__all__ = ["HandlerSourceAttestation", "ModelSourceAttestationResult"]
