# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerLabProofRun — executes a rendered lab proof plan on the host it runs on.

Canonical definition-B handler: ``handle(ModelLabProofPlan) ->
ModelLabProofRunReport``. This is the one place a proof touches the host: it
runs each planned argv (never a shell), applies the step's mechanical
expectations, and records what happened. It decides nothing about the PR; the
verdict handler does, from this report.

EXECUTION RULES
  * setup and prove steps run in order; the first failing ``must_succeed`` step
    stops them, and every later setup and prove step is recorded as not run;
  * teardown and residue steps ALWAYS run, whatever happened before, so a
    failed proof still leaves the host as it found it and says whether it did;
  * a step with ``retry`` is re-run every ``interval_seconds`` until it succeeds
    or ``deadline_seconds`` passes;
  * full output goes to ``<log_dir>/<nn>-<step>.log`` on the host; the report
    carries a tail, except for steps marked ``record_output: false``.

Ticket: OMN-19572
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase
from omnibase_infra.lab_proof.model_lab_proof_observation import (
    ModelLabProofObservation,
)
from omnibase_infra.lab_proof.model_lab_proof_plan import ModelLabProofPlan
from omnibase_infra.lab_proof.model_lab_proof_run_report import (
    ModelLabProofRunReport,
)
from omnibase_infra.lab_proof.model_lab_proof_step import ModelLabProofStep

TAIL_CHARS = 6000
MAX_EXTRACTED = 500
ALWAYS_RUN_PHASES = frozenset(
    {EnumLabProofStepPhase.TEARDOWN, EnumLabProofStepPhase.RESIDUE}
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: object) -> str:
    """Process output as text: a timeout hands back bytes, a finished run a str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


class HandlerLabProofRun:
    """Runs one plan's steps as subprocesses and records each outcome."""

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Take the process runner, sleep and clock as seams so tests need no host."""
        self._runner = runner
        self._sleep = sleep
        self._clock = clock

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler (host I/O)."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: effect, runs processes on the host."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, plan: ModelLabProofPlan) -> ModelLabProofRunReport:
        """Execute the plan and return one observation per step, in plan order."""
        started = _now()
        lane_root = Path(plan.workdir).parent
        lane_root.mkdir(parents=True, exist_ok=True)
        log_dir = Path(plan.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        observations: list[ModelLabProofObservation] = []
        aborted_at: EnumLabProofStepId | None = None
        for index, step in enumerate(plan.steps):
            if aborted_at is not None and step.phase not in ALWAYS_RUN_PHASES:
                observations.append(
                    ModelLabProofObservation(
                        step_id=step.step_id,
                        phase=step.phase,
                        attribution=step.attribution,
                        ran=False,
                        skip_reason=f"not run: {aborted_at} failed",
                    )
                )
                continue
            observation = self._run_step(
                step, log_dir / f"{index:02d}-{step.step_id}.log"
            )
            observations.append(observation)
            if (
                aborted_at is None
                and step.must_succeed
                and not observation.ok
                and step.phase not in ALWAYS_RUN_PHASES
            ):
                aborted_at = step.step_id
        return ModelLabProofRunReport(
            run_key=plan.run_key,
            host=plan.host,
            started_at=started,
            finished_at=_now(),
            aborted_at=aborted_at,
            observations=tuple(observations),
        )

    def _run_step(
        self, step: ModelLabProofStep, log_path: Path
    ) -> ModelLabProofObservation:
        begin = self._clock()
        deadline = begin + (step.retry.deadline_seconds if step.retry else 0)
        attempts = 0
        exit_code: int | None = None
        timed_out = False
        stdout = stderr = ""
        met = False
        while True:
            attempts += 1
            if not Path(step.cwd).is_dir():
                exit_code, stdout, stderr = (
                    None,
                    "",
                    f"working directory missing: {step.cwd}",
                )
            else:
                try:
                    completed = self._runner(
                        list(step.argv),
                        cwd=step.cwd,
                        env={**os.environ, **step.env},
                        capture_output=True,
                        text=True,
                        timeout=step.timeout_seconds,
                        check=False,
                    )
                    exit_code, timed_out = completed.returncode, False
                    stdout, stderr = _text(completed.stdout), _text(completed.stderr)
                except subprocess.TimeoutExpired as exc:
                    exit_code, timed_out = None, True
                    stdout, stderr = _text(exc.stdout), _text(exc.stderr)
                except OSError as exc:
                    exit_code, timed_out = None, False
                    stdout, stderr = "", f"could not start {step.argv[0]}: {exc}"
            met = exit_code == 0 and self._expectations_hold(step, stdout)
            if met or step.retry is None or self._clock() >= deadline:
                break
            self._sleep(step.retry.interval_seconds)
        combined = stdout + "\n" + stderr
        counts = {pattern: combined.count(pattern) for pattern in step.grep_patterns}
        extracted = (
            tuple(
                sorted(set(re.findall(step.extract_pattern, combined)))[:MAX_EXTRACTED]
            )
            if step.extract_pattern
            else ()
        )
        observation = ModelLabProofObservation(
            step_id=step.step_id,
            phase=step.phase,
            attribution=step.attribution,
            ran=True,
            exit_code=exit_code,
            timed_out=timed_out,
            attempts=attempts,
            duration_seconds=round(max(self._clock() - begin, 0.0), 3),
            stdout_tail=stdout[-TAIL_CHARS:] if step.record_output else "",
            stderr_tail=stderr[-TAIL_CHARS:] if step.record_output else "",
            expectation_met=met,
            ok=met,
            pattern_counts=counts,
            extracted=extracted,
            log_path=str(log_path),
        )
        self._write_log(step, observation, stdout, stderr)
        return observation

    @staticmethod
    def _expectations_hold(step: ModelLabProofStep, stdout: str) -> bool:
        out = stdout.strip()
        if step.expect_stdout_equals is not None and out != step.expect_stdout_equals:
            return False
        if step.expect_stdout_empty and out:
            return False
        return not (step.expect_stdout_nonempty and not out)

    @staticmethod
    def _write_log(
        step: ModelLabProofStep,
        observation: ModelLabProofObservation,
        stdout: str,
        stderr: str,
    ) -> None:
        # The argv is logged by name and length only: python -c programs are long,
        # and the plan carries them verbatim for anyone who needs them.
        header = (
            f"step={step.step_id} phase={step.phase} cwd={step.cwd}\n"
            f"argv0={step.argv[0]} argc={len(step.argv)} "
            f"attempts={observation.attempts} exit={observation.exit_code} "
            f"timed_out={observation.timed_out}\n"
        )
        Path(observation.log_path).write_text(
            header + "--- stdout ---\n" + stdout + "\n--- stderr ---\n" + stderr + "\n",
            encoding="utf-8",
        )


__all__ = ["HandlerLabProofRun"]
