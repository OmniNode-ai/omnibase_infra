# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20317: the nightly integration step opts its Kafka suites in.

Four kafka-marked suites call ``require_service_env`` with the
``KAFKA_INTEGRATION_TESTS`` opt-in and fail, rather than skip, in CI when it is
absent (OMN-18781). The nightly selects them by marker and its e2e stack boots a
plaintext Redpanda, but the "Run integration tests" step never exported the
opt-in, so run 45325746514 reported 10 errors and 1 failure on that one cause.

Falsifier: delete ``KAFKA_INTEGRATION_TESTS`` from the step's ``env`` in
nightly-integration.yml and test_run_step_exports_the_kafka_opt_in fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly-integration.yml"
KAFKA_OPT_IN = "KAFKA_INTEGRATION_TESTS"


def _steps() -> list[dict[str, object]]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "integration-tests"
    ]["steps"]


def _step(name: str) -> dict[str, object]:
    return next(step for step in _steps() if step["name"] == name)


def test_run_step_exports_the_kafka_opt_in() -> None:
    env = _step("Run integration tests").get("env", {})
    assert isinstance(env, dict)
    assert env.get(KAFKA_OPT_IN) == "1"


def test_the_opt_in_is_backed_by_the_stack_the_job_boots() -> None:
    names = [step["name"] for step in _steps()]
    spin_up = _step("Spin up e2e stack")["run"]
    assert isinstance(spin_up, str)
    assert "redpanda" in spin_up
    # The broker address the opt-in vouches for is exported by a step that runs
    # after the stack is up and before the step carrying the opt-in, so the
    # opt-in never names a broker the job has not yet started.
    resolve = _step("Resolve reachable e2e connectivity host")["run"]
    assert isinstance(resolve, str)
    assert "KAFKA_BOOTSTRAP_SERVERS=" in resolve
    assert (
        names.index("Spin up e2e stack")
        < names.index("Resolve reachable e2e connectivity host")
        < names.index("Run integration tests")
    )
    env = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["env"]
    assert env["ONEX_EVENT_BUS_TYPE"] == "kafka"
    run_env = _step("Run integration tests").get("env", {})
    assert isinstance(run_env, dict)
    assert run_env.get(KAFKA_OPT_IN) == "1"
