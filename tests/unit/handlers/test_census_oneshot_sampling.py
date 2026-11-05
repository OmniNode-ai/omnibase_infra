# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Completion sampling preserves failures and shares one wall-clock budget."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_census_oneshot as handler
from omnibase_infra.models.model_census_oneshot_policy import ModelCensusOneshotPolicy
from omnibase_infra.models.model_census_oneshot_request import ModelCensusOneshotRequest
from omnibase_infra.models.model_census_oneshot_result import ModelCensusOneshotResult

pytestmark = pytest.mark.unit

_POLICY = ModelCensusOneshotPolicy(
    phase_timeouts={"runtime": 10},
    sample_interval_seconds=2,
    dispatch_timeout_seconds=20,
)


class Samples:
    """A collector whose work and sleeps advance the same deterministic clock."""

    def __init__(self, codes: tuple[int, ...], *, cost: float = 0) -> None:
        self.codes = codes
        self.cost = cost
        self.now = 0.0
        self.timeouts: list[float] = []
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def sample(
        self,
        script_directory: str,
        arguments: tuple[str, ...],
        plan_path: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        index = len(self.timeouts)
        self.timeouts.append(timeout)
        if self.cost > timeout:
            self.now += timeout
            raise subprocess.TimeoutExpired("census", timeout)
        self.now += self.cost
        plan_path.write_text(
            json.dumps(
                {"pending_oneshots": [{"remaining_seconds": 299}]}
                if index < len(self.codes) - 1
                else {"pending_oneshots": []}
            )
        )
        return subprocess.CompletedProcess(
            "census", self.codes[index], json.dumps({"sample": index + 1}), ""
        )

    def run(self, monkeypatch: pytest.MonkeyPatch) -> ModelCensusOneshotResult:
        monkeypatch.setattr(handler.collector, "sample", self.sample)
        return handler.HandlerCensusOneshot(clock=self.clock, sleep=self.sleep).handle(
            ModelCensusOneshotRequest(script_directory="unused", policy=_POLICY)
        )


@pytest.mark.parametrize(
    ("codes", "expected"),
    [(codes, 6) for codes in itertools.permutations((30, 31, 6), 3)]
    + [(codes, 8) for codes in itertools.permutations((8, 6, 7), 3)]
    + [(codes, 7) for codes in itertools.permutations((31, 7, 0), 3)]
    + [((6, 0), 6), ((7, 0), 7), ((31, 0), 31), ((8, 30), 8), ((30, 8), 8)],
)
def test_sample_failures_have_order_independent_precedence(
    monkeypatch: pytest.MonkeyPatch, codes: tuple[int, ...], expected: int
) -> None:
    samples = Samples(codes)
    result = samples.run(monkeypatch)
    assert result.exit_code == expected
    assert json.loads(result.stdout) == {"sample": len(codes)}


def test_followup_subprocess_uses_only_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = Samples((0, 0), cost=3)
    result = samples.run(monkeypatch)
    assert result.exit_code == 0
    assert samples.timeouts == [10, 5]
    assert samples.now == 8


@pytest.mark.parametrize("first_code", [0, 30, 8])
def test_sample_timeout_retains_completed_output_and_earlier_alarm(
    monkeypatch: pytest.MonkeyPatch, first_code: int
) -> None:
    samples = Samples((first_code, 0), cost=6)
    result = samples.run(monkeypatch)
    assert samples.timeouts == [10, 2]
    assert samples.now == 10
    assert result.exit_code == (first_code or 4)
    assert json.loads(result.stdout) == {"sample": 1}
    assert "sample timed out" in result.stderr


@pytest.mark.parametrize("first_code", [0, 30, 8])
def test_sleep_cannot_start_another_sample_after_deadline(
    monkeypatch: pytest.MonkeyPatch, first_code: int
) -> None:
    samples = Samples((first_code, 0), cost=9.5)
    result = samples.run(monkeypatch)
    assert samples.timeouts == [10]
    assert samples.sleeps == [0.5]
    assert samples.now == 10
    assert result.exit_code == (first_code or 4)
    assert json.loads(result.stdout) == {"sample": 1}
    assert "sampling budget exhausted" in result.stderr


@pytest.mark.parametrize(("arguments", "expected"), [((), 4), (("--help",), 0)])
def test_missing_plan_refuses_clean_sample_except_help(
    monkeypatch: pytest.MonkeyPatch, arguments: tuple[str, ...], expected: int
) -> None:
    def no_plan(*args: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess("census", 0, "collector output", "")

    monkeypatch.setattr(handler.collector, "sample", no_plan)
    result = handler.HandlerCensusOneshot().handle(
        ModelCensusOneshotRequest(
            script_directory="unused", policy=_POLICY, arguments=arguments
        )
    )
    assert result.exit_code == expected
    assert result.stdout == "collector output"


def test_relocated_deploy_agent_loads_phase_policy_from_installed_package(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    agent_root = tmp_path / "scripts/deploy-agent"
    shutil.copytree(
        repo / "scripts/deploy-agent/deploy_agent", agent_root / "deploy_agent"
    )
    (tmp_path / "config").mkdir()
    shutil.copy2(repo / "config/deploy_lane_routing.yaml", tmp_path / "config")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from deploy_agent.executor import PHASE_TIMEOUTS, Phase; "
            "assert PHASE_TIMEOUTS[Phase.RUNTIME] == 300; "
            'assert set(PHASE_TIMEOUTS) == {"preflight", "git", "compose_gen", "core", "runtime", "verification"}',
        ],
        cwd=agent_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
