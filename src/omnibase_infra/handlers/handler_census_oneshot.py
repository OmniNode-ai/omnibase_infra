# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Grade oneshots and re-sample through the producer's declared completion bound."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_census_oneshot_request import (
    ModelCensusOneshotPolicy,
    ModelCensusOneshotRequest,
    ModelOneshotObservation,
)
from omnibase_infra.models.model_census_oneshot_result import (
    ModelCensusOneshotResult,
    ModelOneshotGrade,
    ModelPendingOneshot,
)


def load_policy(path: Path) -> ModelCensusOneshotPolicy:
    """Load explicitly supplied policy at a composition boundary."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy = ModelCensusOneshotPolicy.model_validate(
        {key: data[key] for key in ModelCensusOneshotPolicy.model_fields}
    )
    if any(
        type(value) is not int or value <= 0 for value in policy.phase_timeouts.values()
    ):
        raise ValueError("invalid census phase limits")
    if not 0 < policy.sample_interval_seconds <= policy.phase_timeouts["runtime"]:
        raise ValueError("invalid census sample interval")
    return policy


def labels_to_dict(labels: object) -> dict[str, str]:
    """Coerce a container's Labels to a mapping.

    The Docker Engine API returns Labels as a real mapping; the Docker CLI
    returns a comma-joined ``k=v`` string. The collector
    (``lane_census_inventory.py``) normalizes both to a mapping, so the mapping
    branch is the live path — the string branch is retained for envelopes
    produced by older callers and for direct CLI-shaped fixtures. Note the string
    form is lossy by construction: a label VALUE may itself contain commas
    (``com.docker.compose.project.config_files``), which a flat split cannot
    recover. Prefer the mapping form (OMN-15466).
    """
    if isinstance(labels, dict):
        return {str(k): str(v) for k, v in labels.items()}
    out: dict[str, str] = {}
    for pair in str(labels or "").split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _instant(value: str) -> datetime | None:
    text = re.sub(r"(\.\d{6})\d+", r"\1", value.strip()).replace("Z", "+00:00")
    try:
        instant = datetime.fromisoformat(text)
    except ValueError:
        return None
    return instant if instant.tzinfo is not None and instant.year >= 2000 else None


def grade_oneshot(request: ModelOneshotObservation) -> ModelOneshotGrade:
    """Pure completion decision; absence and exited-zero remain clean."""
    if not request.present or (request.running and request.keepalive):
        return ModelOneshotGrade()
    if not request.running:
        match = re.search(r"Exited\s*\((\d+)\)", request.status, re.IGNORECASE)
        if match and int(match[1]) != 0:
            return ModelOneshotGrade(
                kind="oneshot_failed",
                detail=f"migration/init container {request.container!r} Exited non-zero (code {match[1]}): {request.status}",
            )
        return ModelOneshotGrade()
    started = _instant(request.started_at)
    observed = request.observed_at
    if (
        started is None
        or observed is None
        or observed.tzinfo is None
        or started > observed
    ):
        return ModelOneshotGrade(
            kind="oneshot_unobservable",
            detail=f"cannot grade completion of running oneshot {request.container!r}: missing, invalid or future StartedAt/observed_at; {request.status}",
        )
    elapsed = (observed - started).total_seconds()
    bound = request.completion_timeout_seconds
    if elapsed < bound:
        return ModelOneshotGrade(
            pending=ModelPendingOneshot(
                lane=request.lane,
                container=request.container,
                started_at=started.isoformat(),
                deadline=(started + timedelta(seconds=bound)).isoformat(),
                remaining_seconds=bound - elapsed,
            )
        )
    return ModelOneshotGrade(
        kind="oneshot_stuck",
        detail=f"migration/init container {request.container!r} is still Running after {elapsed:g}s (completion bound {bound}s): {request.status}",
    )


class HandlerCensusOneshot:
    """Publish each census sample before rechecking still-pending executions."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, request: ModelCensusOneshotRequest) -> ModelCensusOneshotResult:
        """Execute the existing collector through a finite follow-up window."""
        status = 0
        diagnostics: list[str] = []
        deadline = self._clock() + request.policy.phase_timeouts["runtime"]
        arguments = request.arguments
        with TemporaryDirectory(prefix="census-oneshot-") as scratch:
            plan_path = Path(scratch) / "plan.json"
            environment = {
                **os.environ,
                "ONEX_CENSUS_SINGLE_SAMPLE": "1",
                "ONEX_CENSUS_PLAN_OUT": str(plan_path),
            }
            while True:
                plan_path.unlink(missing_ok=True)
                try:
                    sample = subprocess.run(
                        [
                            "bash",
                            str(
                                Path(request.script_directory) / "lane-census-check.sh"
                            ),
                            *arguments,
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=request.policy.phase_timeouts["runtime"],
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    diagnostics.append(f"census sample timed out: {exc}\n")
                    return ModelCensusOneshotResult(
                        exit_code=4, stderr="".join(diagnostics)
                    )
                diagnostics.append(sample.stderr)
                if sample.returncode:
                    if sample.returncode == 8 or status in (0, 30):
                        status = sample.returncode
                if (
                    sample.returncode not in (0, 6, 7, 8, 30, 31)
                    or not plan_path.is_file()
                ):
                    return ModelCensusOneshotResult(
                        exit_code=sample.returncode,
                        stdout=sample.stdout,
                        stderr="".join(diagnostics),
                    )
                if not plan_path.is_file():
                    code = sample.returncode or (
                        0 if set(arguments) & {"--help", "-h"} else 4
                    )
                    return ModelCensusOneshotResult(
                        exit_code=code,
                        stdout=sample.stdout,
                        stderr="".join(diagnostics)
                        + "census sample supplied no completion plan\n",
                    )
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                pending = plan["pending_oneshots"]
                if not pending:
                    return ModelCensusOneshotResult(
                        exit_code=status,
                        stdout=sample.stdout,
                        stderr="".join(diagnostics),
                    )
                if self._clock() >= deadline:
                    diagnostics.append(
                        "ABORT: oneshot follow-up sampling budget exhausted with a new execution pending; completion could not be observed.\n"
                    )
                    return ModelCensusOneshotResult(
                        exit_code=4, stderr="".join(diagnostics)
                    )
                delay = min(
                    request.policy.sample_interval_seconds,
                    *[row["remaining_seconds"] for row in pending],
                )
                diagnostics.append(
                    f"Oneshot completion pending; automatically sampling again in {delay}s.\n"
                )
                self._sleep(delay)
                arguments = tuple(arg for arg in arguments if arg != "--memory")
