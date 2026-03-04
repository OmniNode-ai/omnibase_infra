# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that validates all prerequisites before platform provisioning.

Performs 7 preflight checks: docker_version, compose_version, python_version,
postgres_password_set, docker_daemon, omnibase_dir, port_availability.

Invariants:
    I3 — Monkeypatch discipline: import subprocess at module top (not from X import Y).
         Module-level helpers are used for testable sys.version_info and socket usage.
    I4 — Port semantics: checks ports are FREE (connect_ex != 0).

Ticket: OMN-3492
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_check_result import (
    ModelPreflightCheckResult,
)
from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_effect_output import (
    ModelPreflightEffectOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

# --- Module-level helpers (I3: extract for testability) ---


def _get_python_version_info() -> tuple[int, int, int]:
    """Wraps sys.version_info for testability."""
    return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def _check_port_free(host: str, port: int) -> bool:
    """Returns True if port is NOT in use. I4: Preflight = FREE check."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        result = sock.connect_ex((host, port))
        return result != 0
    finally:
        sock.close()


def _omnibase_dir() -> Path:
    """Returns the omnibase directory path from env or default."""
    env = os.environ.get("OMNIBASE_DIR")  # ONEX_EXCLUDE
    return Path(env) if env else Path.home() / ".omnibase"


def _parse_semver(version_str: str) -> tuple[int, int, int]:
    """Parse a semver string into a (major, minor, patch) tuple."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str.strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)


# Local service ports to check for availability
_LOCAL_SERVICE_PORTS: tuple[int, ...] = (5436, 19092, 16379)
_MIN_DOCKER_VERSION: tuple[int, int, int] = (24, 0, 0)
_MIN_COMPOSE_VERSION: tuple[int, int, int] = (2, 20, 0)
_MIN_PYTHON_VERSION: tuple[int, int, int] = (3, 12, 0)


def _check_docker_version() -> ModelPreflightCheckResult:
    """Check Docker Engine version >= 24.0.0."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ModelPreflightCheckResult(
                check_key="docker_version",
                passed=False,
                message="docker version command failed",
                detail=result.stderr or result.stdout,
            )
        version_str = result.stdout.strip()
        parsed = _parse_semver(version_str)
        if parsed < _MIN_DOCKER_VERSION:
            return ModelPreflightCheckResult(
                check_key="docker_version",
                passed=False,
                message=f"Docker {version_str} < required {_MIN_DOCKER_VERSION}",
                detail=None,
            )
        return ModelPreflightCheckResult(
            check_key="docker_version",
            passed=True,
            message=f"Docker {version_str} ok",
            detail=None,
        )
    except FileNotFoundError:
        return ModelPreflightCheckResult(
            check_key="docker_version",
            passed=False,
            message="docker not found in PATH",
            detail=None,
        )


def _check_compose_version() -> ModelPreflightCheckResult:
    """Check Docker Compose version >= 2.20.0."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version", "--short"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ModelPreflightCheckResult(
                check_key="compose_version",
                passed=False,
                message="docker compose version command failed",
                detail=result.stderr or result.stdout,
            )
        version_str = result.stdout.strip()
        parsed = _parse_semver(version_str)
        if parsed < _MIN_COMPOSE_VERSION:
            return ModelPreflightCheckResult(
                check_key="compose_version",
                passed=False,
                message=f"Compose {version_str} < required {_MIN_COMPOSE_VERSION}",
                detail=None,
            )
        return ModelPreflightCheckResult(
            check_key="compose_version",
            passed=True,
            message=f"Compose {version_str} ok",
            detail=None,
        )
    except FileNotFoundError:
        return ModelPreflightCheckResult(
            check_key="compose_version",
            passed=False,
            message="docker not found in PATH",
            detail=None,
        )


def _check_python_version() -> ModelPreflightCheckResult:
    """Check Python version >= 3.12.0."""
    ver = _get_python_version_info()
    if ver < _MIN_PYTHON_VERSION:
        return ModelPreflightCheckResult(
            check_key="python_version",
            passed=False,
            message=f"Python {ver} < required {_MIN_PYTHON_VERSION}",
            detail=None,
        )
    return ModelPreflightCheckResult(
        check_key="python_version",
        passed=True,
        message=f"Python {'.'.join(str(v) for v in ver)} ok",
        detail=None,
    )


def _check_postgres_password() -> ModelPreflightCheckResult:
    """Check POSTGRES_PASSWORD env var is set and non-empty."""
    val = os.environ.get("POSTGRES_PASSWORD")  # ONEX_EXCLUDE
    if not val:
        return ModelPreflightCheckResult(
            check_key="postgres_password_set",
            passed=False,
            message="POSTGRES_PASSWORD is not set or empty",
            detail=None,
        )
    return ModelPreflightCheckResult(
        check_key="postgres_password_set",
        passed=True,
        message="POSTGRES_PASSWORD is set",
        detail=None,
    )


def _check_docker_daemon() -> ModelPreflightCheckResult:
    """Check Docker daemon is running via docker info."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ModelPreflightCheckResult(
                check_key="docker_daemon",
                passed=False,
                message="Docker daemon not running",
                detail=result.stderr,
            )
        return ModelPreflightCheckResult(
            check_key="docker_daemon",
            passed=True,
            message="Docker daemon is running",
            detail=None,
        )
    except FileNotFoundError:
        return ModelPreflightCheckResult(
            check_key="docker_daemon",
            passed=False,
            message="docker not found in PATH",
            detail=None,
        )


def _check_omnibase_dir() -> ModelPreflightCheckResult:
    """Check omnibase directory exists and is writable."""
    target_dir = _omnibase_dir()
    if not target_dir.exists():
        return ModelPreflightCheckResult(
            check_key="omnibase_dir",
            passed=False,
            message=f"omnibase dir does not exist: {target_dir}",
            detail=None,
        )
    if not os.access(target_dir, os.W_OK):
        return ModelPreflightCheckResult(
            check_key="omnibase_dir",
            passed=False,
            message=f"omnibase dir is not writable: {target_dir}",
            detail=None,
        )
    return ModelPreflightCheckResult(
        check_key="omnibase_dir",
        passed=True,
        message=f"omnibase dir ok: {target_dir}",
        detail=None,
    )


def _check_port_availability() -> ModelPreflightCheckResult:
    """Check that local service ports are free."""
    occupied: list[int] = []
    for port in _LOCAL_SERVICE_PORTS:
        if not _check_port_free("localhost", port):
            occupied.append(port)
    if occupied:
        return ModelPreflightCheckResult(
            check_key="port_availability",
            passed=False,
            message=f"Ports already in use: {occupied}",
            detail=f"Expected free: {list(_LOCAL_SERVICE_PORTS)}, occupied: {occupied}",
        )
    return ModelPreflightCheckResult(
        check_key="port_availability",
        passed=True,
        message=f"All required ports free: {list(_LOCAL_SERVICE_PORTS)}",
        detail=None,
    )


def _run_all_checks() -> tuple[ModelPreflightCheckResult, ...]:
    """Run all 7 preflight checks and return results."""
    return (
        _check_docker_version(),
        _check_compose_version(),
        _check_python_version(),
        _check_postgres_password(),
        _check_docker_daemon(),
        _check_omnibase_dir(),
        _check_port_availability(),
    )


class HandlerPreflightCheck:
    """Validates all prerequisites before platform provisioning.

    Performs 7 checks: docker_version, compose_version, python_version,
    postgres_password_set, docker_daemon, omnibase_dir, port_availability.
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
        logger.info("HandlerPreflightCheck initialized")

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerPreflightCheck shutdown")

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[ModelPreflightEffectOutput]:
        """Run all preflight checks.

        Envelope keys:
            correlation_id: UUID for tracing.
            checks: Optional list of check names to run (default: all 7).
        """
        correlation_id_raw = envelope.get("correlation_id")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else None,
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="run_preflight",
            target_name="preflight_check",
        )
        corr_id = context.correlation_id
        if corr_id is None:
            raise RuntimeError("correlation_id must not be None")

        start_ms = time.monotonic() * 1000.0
        checks = _run_all_checks()
        duration_ms = time.monotonic() * 1000.0 - start_ms

        all_passed = all(c.passed for c in checks)

        logger.info(
            "Preflight complete: passed=%s, checks=%d, duration_ms=%.1f",
            all_passed,
            len(checks),
            duration_ms,
        )

        result = ModelPreflightEffectOutput(
            passed=all_passed,
            checks=checks,
            correlation_id=corr_id,
            duration_ms=duration_ms,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-preflight-check",
            result=result,
        )


__all__: list[str] = [
    "HandlerPreflightCheck",
    "_check_port_free",
    "_get_python_version_info",
    "_omnibase_dir",
    "_parse_semver",
]
