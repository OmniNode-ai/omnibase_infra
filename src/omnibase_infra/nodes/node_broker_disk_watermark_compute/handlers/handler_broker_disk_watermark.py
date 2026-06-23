# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerBrokerDiskWatermark — probes docker data-root + broker volumes.

Algorithm
---------
1. Resolve docker data-root via ``docker info --format '{{json .}}'`` →
   parse ``DockerRootDir``.  On error the data-root probe is skipped and
   ``probe_error`` is set in the output.
2. Probe the data-root filesystem with ``shutil.disk_usage``.
3. For each named broker volume, resolve its ``Mountpoint`` via
   ``docker volume inspect <name>`` and probe that path.
4. Classify each probe result against ``warn_threshold`` / ``p0_threshold``
   and the ``min_free_bytes_floor``.
5. Aggregate into ``ModelBrokerDiskWatermarkOutput``.

docker info / shutil.disk_usage are injectable seams so the handler is
fully testable without a live Docker daemon.

Ticket: OMN-13009
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Callable

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.enum_disk_severity import (
    EnumDiskSeverity,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_check import (
    ModelBrokerDiskCheck,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_watermark_input import (
    ModelBrokerDiskWatermarkInput,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_watermark_output import (
    ModelBrokerDiskWatermarkOutput,
)

logger = logging.getLogger(__name__)

# Severity rank for aggregation — higher is worse.
_SEVERITY_RANK: dict[EnumDiskSeverity, int] = {
    EnumDiskSeverity.CLEAN: 0,
    EnumDiskSeverity.WARN: 1,
    EnumDiskSeverity.P0: 2,
}
_RANK_SEVERITY: dict[int, EnumDiskSeverity] = {v: k for k, v in _SEVERITY_RANK.items()}


def _classify(
    usage_pct: float,
    free_bytes: int,
    *,
    warn_threshold: float,
    p0_threshold: float,
    min_free_bytes_floor: int,
) -> tuple[EnumDiskSeverity, bool]:
    """Classify a single probe result.

    Returns ``(severity, below_min_free_floor)``.  The ``min_free_bytes_floor``
    cross-check independently escalates severity to P0: at 0 free bytes (the
    exact demo-day condition) the broker refuses to start regardless of the
    overall disk percentage.
    """
    below_floor = free_bytes <= min_free_bytes_floor

    if usage_pct >= p0_threshold or below_floor:
        return EnumDiskSeverity.P0, below_floor
    if usage_pct >= warn_threshold:
        return EnumDiskSeverity.WARN, below_floor
    return EnumDiskSeverity.CLEAN, below_floor


def _real_docker_info() -> dict[str, object]:
    """Run ``docker info --format '{{json .}}'`` and return parsed JSON."""
    result = subprocess.run(
        ["docker", "info", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker info failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return dict(json.loads(result.stdout))


def _real_disk_usage(path: str) -> tuple[int, int, int]:
    """Return ``(total, used, free)`` bytes via ``shutil.disk_usage``."""
    usage = shutil.disk_usage(path)
    return usage.total, usage.used, usage.free


def _resolve_volume_mountpoint(volume_name: str) -> str | None:
    """Resolve a Docker volume mountpoint via ``docker volume inspect``."""
    try:
        result = subprocess.run(
            ["docker", "volume", "inspect", volume_name, "--format", "{{.Mountpoint}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "docker volume inspect failed for %r: %s",
                volume_name,
                result.stderr.strip(),
            )
            return None
        mp = result.stdout.strip()
        return mp if mp else None
    except Exception:  # noqa: BLE001 — best-effort; probe continues on failure
        logger.warning(
            "docker volume inspect raised for %r", volume_name, exc_info=True
        )
        return None


class HandlerBrokerDiskWatermark:
    """Handler that probes docker data-root + broker volumes against watermarks.

    Dependencies (injected via constructor):
        - docker_info_runner: callable() -> dict[str, object]
        - disk_usage_runner: callable(path: str) -> (total, used, free)

    Both seams default to the real subprocess / shutil implementations so
    production use requires no configuration.
    """

    def __init__(
        self,
        *,
        docker_info_runner: Callable[[], dict[str, object]] | None = None,
        disk_usage_runner: Callable[[str], tuple[int, int, int]] | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            docker_info_runner: Injectable seam for docker info. Defaults to
                real subprocess call.
            disk_usage_runner: Injectable seam for disk usage probing. Defaults
                to real ``shutil.disk_usage``.
        """
        self._docker_info_runner = docker_info_runner or _real_docker_info
        self._disk_usage_runner = disk_usage_runner or _real_disk_usage

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure compute (no external I/O side-effects)."""
        return EnumHandlerTypeCategory.COMPUTE

    def probe_disk_watermark(
        self, inp: ModelBrokerDiskWatermarkInput
    ) -> ModelBrokerDiskWatermarkOutput:
        """Probe docker data-root + broker volumes and classify severity.

        Args:
            inp: Watermark probe configuration.

        Returns:
            Output containing per-path checks, max severity, and P0/WARN labels.
        """
        # Use injected seams from input when provided (test path).
        docker_info_fn = inp.docker_info_runner or self._docker_info_runner
        disk_usage_fn = inp.disk_usage_runner or self._disk_usage_runner

        checks: list[ModelBrokerDiskCheck] = []
        probe_error = ""
        docker_root_dir = ""

        # --- Step 1: resolve docker data-root ---
        try:
            info = docker_info_fn()
            docker_root_dir = str(info.get("DockerRootDir", ""))
        except Exception as exc:  # noqa: BLE001
            probe_error = f"docker info unavailable: {exc}"
            logger.warning("HandlerBrokerDiskWatermark: %s", probe_error)

        # --- Step 2: probe docker data-root filesystem ---
        if docker_root_dir:
            check = self._probe_path(
                label="docker-data-root",
                path=docker_root_dir,
                disk_usage_fn=disk_usage_fn,
                warn_threshold=inp.warn_threshold,
                p0_threshold=inp.p0_threshold,
                min_free_bytes_floor=inp.min_free_bytes_floor,
            )
            if check is not None:
                checks.append(check)

        # --- Step 3: probe named broker volumes ---
        for volume_name in inp.broker_volume_names:
            mountpoint = _resolve_volume_mountpoint(volume_name)
            if mountpoint is None:
                logger.warning(
                    "HandlerBrokerDiskWatermark: skipping volume %r — mountpoint not found",
                    volume_name,
                )
                continue
            check = self._probe_path(
                label=volume_name,
                path=mountpoint,
                disk_usage_fn=disk_usage_fn,
                warn_threshold=inp.warn_threshold,
                p0_threshold=inp.p0_threshold,
                min_free_bytes_floor=inp.min_free_bytes_floor,
            )
            if check is not None:
                checks.append(check)

        # --- Step 4: aggregate severity ---
        max_rank = max((_SEVERITY_RANK[c.severity] for c in checks), default=0)
        max_severity = _RANK_SEVERITY[max_rank]
        p0_labels = tuple(c.label for c in checks if c.severity == EnumDiskSeverity.P0)
        warn_labels = tuple(
            c.label for c in checks if c.severity == EnumDiskSeverity.WARN
        )

        return ModelBrokerDiskWatermarkOutput(
            correlation_id=inp.correlation_id,
            checks=tuple(checks),
            max_severity=max_severity,
            p0_labels=p0_labels,
            warn_labels=warn_labels,
            docker_root_dir=docker_root_dir,
            probe_error=probe_error,
        )

    def _probe_path(
        self,
        *,
        label: str,
        path: str,
        disk_usage_fn: Callable[[str], tuple[int, int, int]],
        warn_threshold: float,
        p0_threshold: float,
        min_free_bytes_floor: int,
    ) -> ModelBrokerDiskCheck | None:
        """Probe a single path and return a classified check, or None on error."""
        try:
            total, used, free = disk_usage_fn(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HandlerBrokerDiskWatermark: disk_usage failed for %r (%s): %s",
                label,
                path,
                exc,
            )
            return None

        if total == 0:
            logger.warning(
                "HandlerBrokerDiskWatermark: total=0 for %r (%s), skipping", label, path
            )
            return None

        usage_pct = used / total
        severity, below_floor = _classify(
            usage_pct=usage_pct,
            free_bytes=free,
            warn_threshold=warn_threshold,
            p0_threshold=p0_threshold,
            min_free_bytes_floor=min_free_bytes_floor,
        )

        return ModelBrokerDiskCheck(
            label=label,
            path=path,
            total_bytes=total,
            used_bytes=used,
            free_bytes=free,
            usage_pct=usage_pct,
            severity=severity,
            min_free_bytes_floor=min_free_bytes_floor,
            below_min_free_floor=below_floor,
        )


__all__ = ["HandlerBrokerDiskWatermark"]
