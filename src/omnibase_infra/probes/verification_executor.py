# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Verification executor for orchestrator step proof conditions (OMN-5266).

Maps check_type values to actual probe functions and returns structured
verification results.
"""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import time
from pathlib import Path

from omnibase_infra.probes.capability_probe import http_health_check, socket_check
from omnibase_infra.probes.model_verification_result import ModelVerificationResult
from omnibase_infra.probes.protocol_verification_spec import VerificationSpec


async def _check_command_exit_0(target: str, timeout: int) -> tuple[bool, str]:
    """Run a shell command and check for exit code 0."""
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                target,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
            timeout=timeout,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            return True, f"Command succeeded: {target}"
        return (
            False,
            f"Command failed with exit code {proc.returncode}: {stderr.decode().strip()}",
        )
    except TimeoutError:
        return False, f"Command timed out after {timeout}s: {target}"
    except OSError as exc:
        return False, f"Command execution error: {exc}"


async def _check_file_exists(target: str, _timeout: int) -> tuple[bool, str]:
    """Check if a file exists at the given path."""
    path = Path(target).expanduser()
    if path.exists():
        return True, f"File exists: {target}"
    return False, f"File not found: {target}"


async def _check_tcp_probe(target: str, timeout: int) -> tuple[bool, str]:
    """Check TCP connectivity to host:port."""
    host, _, port_str = target.partition(":")
    if not host or not port_str:
        return False, f"Invalid target format (expected host:port): {target}"
    try:
        port = int(port_str)
    except ValueError:
        return False, f"Invalid port number: {port_str}"

    reachable = socket_check(host, port, timeout=float(timeout))
    if reachable:
        return True, f"TCP connection succeeded: {target}"
    return False, f"TCP connection failed: {target}"


async def _check_http_health(target: str, timeout: int) -> tuple[bool, str]:
    """Check HTTP health endpoint returns 2xx."""
    healthy = http_health_check(target, timeout=float(timeout))
    if healthy:
        return True, f"HTTP health check passed: {target}"
    return False, f"HTTP health check failed: {target}"


async def _check_python_import(target: str, _timeout: int) -> tuple[bool, str]:
    """Check if a Python module can be imported."""
    try:
        importlib.import_module(target)
        return True, f"Module importable: {target}"
    except ImportError:
        return False, f"Module not importable: {target}"


_HANDLERS = {
    "command_exit_0": _check_command_exit_0,
    "file_exists": _check_file_exists,
    "tcp_probe": _check_tcp_probe,
    "http_health": _check_http_health,
    "python_import": _check_python_import,
}


async def execute_verification(spec: VerificationSpec) -> ModelVerificationResult:
    """Execute a verification check and return the result.

    Args:
        spec: Any object with check_type, target, and timeout_seconds
            attributes. Compatible with ModelStepVerification from
            omnibase_core and ModelVerificationSpec from this package.

    Returns:
        A ModelVerificationResult with the check outcome.

    Raises:
        ValueError: If the check_type is not recognized.
    """
    handler = _HANDLERS.get(spec.check_type)
    if handler is None:
        msg = f"Unknown check_type: {spec.check_type}"
        raise ValueError(msg)

    start = time.monotonic()
    passed, message = await handler(spec.target, spec.timeout_seconds)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    return ModelVerificationResult(
        passed=passed,
        check_type=spec.check_type,
        target=spec.target,
        message=message,
        elapsed_ms=elapsed_ms,
    )


__all__ = ["execute_verification"]
