# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler check executor protocol and configuration for validation checks.

Defines the interface and configuration for individual check executors
that can run subprocess commands, analyze diffs, or perform static
analysis as part of the validation pipeline.

Ticket: OMN-2151
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.models.validation.model_check_result import ModelCheckResult

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_validation_orchestrator.models.model_pattern_candidate import (
        ModelPatternCandidate,
    )

logger = logging.getLogger(__name__)


class ModelCheckExecutorConfig(BaseModel):
    """Configuration for a check executor.

    Attributes:
        working_dir: Working directory for subprocess execution.
        timeout_ms: Maximum execution time in milliseconds (0 = no limit).
        env_overrides: Additional environment variables for subprocess.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    working_dir: str = Field(
        default=".", description="Working directory for execution."
    )
    timeout_ms: float = Field(
        default=120_000.0, ge=0.0, description="Max execution time in ms."
    )
    env_overrides: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Additional environment variables as (key, value) pairs.",
    )


class HandlerCheckExecutor(ABC):
    """Abstract base for individual check executors.

    Each check in the catalog has a corresponding executor that knows
    how to run the check and interpret its results.
    """

    @property
    @abstractmethod
    def check_code(self) -> str:
        """Unique check identifier (e.g., CHECK-PY-001)."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable check name."""

    @property
    @abstractmethod
    def severity(self) -> EnumCheckSeverity:
        """Check severity classification."""

    @abstractmethod
    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Execute the check against the given candidate.

        Args:
            candidate: Pattern candidate to validate.
            config: Executor configuration.

        Returns:
            Check result with pass/fail status and details.
        """

    async def _run_subprocess(
        self,
        command: str,
        config: ModelCheckExecutorConfig,
    ) -> tuple[int, str, str]:
        """Run a command as a subprocess.

        Uses ``asyncio.create_subprocess_exec`` with ``shlex.split`` to
        avoid shell injection surfaces.

        Args:
            command: Command string to execute (parsed via shlex.split).
            config: Executor configuration with timeout and working dir.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        # 0 means no timeout (subprocess runs until completion)
        timeout_s = config.timeout_ms / 1000.0 if config.timeout_ms > 0 else None
        working_dir = Path(config.working_dir)

        env_dict: dict[str, str] | None = None
        if config.env_overrides:
            env_dict = {**os.environ, **dict(config.env_overrides)}

        argv = shlex.split(command)
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env_dict,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_s,
            )
            return (
                process.returncode or 0,
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
            )
        except TimeoutError:
            logger.warning(
                "Check %s timed out after %.0f ms",
                self.check_code,
                config.timeout_ms,
            )
            if process is not None:
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    logger.warning(
                        "Check %s: process did not exit within 5 s after "
                        "kill; potential zombie (pid=%s)",
                        self.check_code,
                        process.pid,
                    )
                except (OSError, ProcessLookupError):
                    # Process already exited between timeout and kill attempt
                    pass
            return (-1, "", f"Timed out after {config.timeout_ms:.0f}ms")
        except OSError as exc:
            logger.warning("Check %s subprocess error: %s", self.check_code, exc)
            return (-2, "", str(exc))

    def _make_result(
        self,
        passed: bool,
        message: str,
        error_output: str = "",
        duration_ms: float = 0.0,
        skipped: bool = False,
    ) -> ModelCheckResult:
        """Build a ModelCheckResult for this check.

        Args:
            passed: Whether the check passed.
            message: Human-readable result message.
            error_output: Captured stderr/stdout on failure.
            duration_ms: Check execution duration in milliseconds.
            skipped: Whether the check was skipped.

        Returns:
            Populated ModelCheckResult.
        """
        return ModelCheckResult(
            check_code=self.check_code,
            label=self.label,
            severity=self.severity,
            passed=passed,
            skipped=skipped,
            message=message,
            error_output=error_output,
            duration_ms=duration_ms,
            executed_at=datetime.now(tz=UTC),
        )


class HandlerSubprocessCheckExecutor(HandlerCheckExecutor):
    """Check executor that runs a command and checks the exit code.

    Used for simple checks like mypy, ruff, and pytest where
    exit code 0 = pass, non-zero = fail.
    """

    def __init__(
        self,
        check_code: str,
        label: str,
        severity: EnumCheckSeverity,
        command: str,
    ) -> None:
        """Initialize with check metadata and command.

        Args:
            check_code: Unique check identifier.
            label: Human-readable check name.
            severity: Check severity.
            command: Shell command to execute.
        """
        self._check_code = check_code
        self._label = label
        self._severity = severity
        self._command = command

    @property
    def check_code(self) -> str:
        """Return check code."""
        return self._check_code

    @property
    def label(self) -> str:
        """Return check label."""
        return self._label

    @property
    def severity(self) -> EnumCheckSeverity:
        """Return check severity."""
        return self._severity

    async def execute(
        self,
        candidate: ModelPatternCandidate,
        config: ModelCheckExecutorConfig,
    ) -> ModelCheckResult:
        """Run the command and check exit code.

        Args:
            candidate: Pattern candidate to validate.
            config: Executor configuration.

        Returns:
            Check result based on subprocess exit code.
        """
        start = time.monotonic()
        returncode, stdout, stderr = await self._run_subprocess(self._command, config)
        duration_ms = (time.monotonic() - start) * 1000.0

        passed = returncode == 0

        # Use only the command basename to avoid leaking full command
        # strings in check results (information disclosure concern if
        # commands ever become dynamic).
        cmd_basename = self._command.split()[0] if self._command else "(empty)"

        if passed:
            return self._make_result(
                passed=True,
                message=f"{self.check_code} succeeded ({cmd_basename})",
                duration_ms=duration_ms,
            )

        # Failure path: capture and truncate output for diagnostics
        output = stderr if stderr else stdout
        max_output = 4096
        if len(output) > max_output:
            output = (
                output[:max_output] + f"\n... (truncated, {len(output)} bytes total)"
            )

        return self._make_result(
            passed=False,
            message=f"{self.check_code} failed (exit {returncode}, {cmd_basename})",
            error_output=output,
            duration_ms=duration_ms,
        )


__all__: list[str] = [
    "HandlerCheckExecutor",
    "HandlerSubprocessCheckExecutor",
    "ModelCheckExecutorConfig",
]
