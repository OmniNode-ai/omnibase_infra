# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OmniGate trusted check executor."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

from omnibase_core.enums.ticket.enum_receipt_status import EnumReceiptStatus
from omnibase_core.models.gate.model_omnigate_check import ModelOmniGateCheck
from omnibase_core.models.gate.model_omnigate_check_result import (
    ModelOmniGateCheckResult,
)
from omnibase_core.models.gate.model_omnigate_config import ModelOmniGateConfig
from omnibase_infra.gate.validator_registry import execute_validator

_BASE_ENV_ALLOWLIST = frozenset({"PATH", "HOME", "USER", "LANG", "LC_ALL"})
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(bearer|basic)\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
)
_PREVIEW_LENGTH = 4096


class SecretBearingOutputError(ValueError):
    """Raised when a check output appears to contain a secret."""


def execute_checks(
    config: ModelOmniGateConfig,
    repo_path: Path,
) -> tuple[ModelOmniGateCheckResult, ...]:
    """Execute trusted OmniGate shell checks, then declared validators."""
    results: list[ModelOmniGateCheckResult] = []
    for check in config.checks:
        results.append(_execute_shell_check(config, check, repo_path))
    for validator_ref in config.validators:
        results.append(execute_validator(validator_ref, repo_path))
    return tuple(results)


def _execute_shell_check(
    config: ModelOmniGateConfig,
    check: ModelOmniGateCheck,
    repo_path: Path,
) -> ModelOmniGateCheckResult:
    start = time.monotonic()
    denied_reason = _denied_command_reason(config, check.run)
    if denied_reason is not None:
        return _build_result(
            name=check.name,
            command=check.run,
            status=EnumReceiptStatus.FAIL,
            duration_ms=int((time.monotonic() - start) * 1000),
            output=denied_reason,
        )

    try:
        proc = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", check.run],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=check.timeout_seconds,
            env=_build_sanitized_env(check),
            check=False,
        )
        status = (
            EnumReceiptStatus.PASS if proc.returncode == 0 else EnumReceiptStatus.FAIL
        )
        output = f"{proc.stdout}{proc.stderr}"
    except subprocess.TimeoutExpired as exc:
        status = EnumReceiptStatus.FAIL
        stdout = _coerce_timeout_output(exc.stdout)
        stderr = _coerce_timeout_output(exc.stderr)
        output = f"{stdout}{stderr}\nCheck timed out after {check.timeout_seconds}s"

    return _build_result(
        name=check.name,
        command=check.run,
        status=status,
        duration_ms=int((time.monotonic() - start) * 1000),
        output=output,
    )


def _build_sanitized_env(check: ModelOmniGateCheck) -> dict[str, str]:
    allowed = _BASE_ENV_ALLOWLIST.union(check.allowed_env)
    return {key: value for key, value in os.environ.items() if key in allowed}


def _denied_command_reason(
    config: ModelOmniGateConfig,
    command: str,
) -> str | None:
    for pattern in getattr(config, "denied_command_patterns", ()):
        try:
            if re.search(pattern, command):
                return f"Command denied by OmniGate pattern: {pattern}"
        except re.error as exc:
            return f"Invalid OmniGate denied command pattern {pattern!r}: {exc}"
    return None


def _coerce_timeout_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _build_result(
    *,
    name: str,
    command: str,
    status: EnumReceiptStatus,
    duration_ms: int,
    output: str,
) -> ModelOmniGateCheckResult:
    stdout_preview, stdout_hash = _stdout_preview_and_hash(output)
    return ModelOmniGateCheckResult(
        name=name,
        command=command,
        status=status,
        duration_ms=duration_ms,
        stdout_preview=stdout_preview,
        stdout_hash=stdout_hash,
    )


def _stdout_preview_and_hash(output: str) -> tuple[str | None, str | None]:
    if _looks_secret_bearing(output):
        msg = "Check output appears to contain a secret; refusing receipt result"
        raise SecretBearingOutputError(msg)
    if output == "":
        return None, None
    digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
    return output[:_PREVIEW_LENGTH], f"sha256:{digest}"


def _looks_secret_bearing(output: str) -> bool:
    return any(pattern.search(output) for pattern in _SECRET_PATTERNS)


_SECRET_REDACTED_PLACEHOLDER = "<redacted: output appears to contain a secret>"


def redact_if_secret_bearing(output: str | None) -> str | None:
    """Public helper: replace a secret-bearing string with a fixed placeholder."""
    if output is None:
        return None
    if _looks_secret_bearing(output):
        return _SECRET_REDACTED_PLACEHOLDER
    return output


__all__ = [
    "SecretBearingOutputError",
    "execute_checks",
    "redact_if_secret_bearing",
]
