# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The onex-api guard publishes the job-end generation within its wait window.

OMN-20154: a successful job's own recovered recreate may replace the converged
container with the same image and revision. Only that recorded movement rebinds;
every other movement still fails the lab-pass generation check. All reads use
fakes, including the agent's /job response and the container's compose label.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Never

import pytest
import yaml

from scripts.ci import check_dev_lane_staleness as staleness
from scripts.ci import check_lane_onex_api_revision as guard
from scripts.ci import check_lane_sibling_revision as sibling
from scripts.ci.deploy_lane_verify_route import (
    job_env,
    load_table,
    targets_for_receipt_lane,
    write_lane_env,
)
from scripts.ci.lab_pass_receipt import (
    ModelLaneGeneration,
    generation_check,
    parse_generation,
    read_lane_generation,
)

pytestmark = pytest.mark.unit

AGENT = "http://agent.invalid:8098"
CID = "ada00fdd-90ef-430c-9a49-adac38ac7131"
SERVICE = "onex-api"
IMAGE = "sha256:" + "1" * 64
REVISION = "c0dc1d80ae818b95bbc3dcec8a72dcb0fa4b02ff"
OLD_ID = "7b926d92febd" + "0" * 52
NEW_ID = "e311dd332d91" + "0" * 52
START = datetime(2026, 9, 30, 23, 12, 36, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/onex-api-lab-delivery-reusable.yml"


@pytest.fixture(autouse=True)
def _forbid_external_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> Never:
        pytest.fail("unit test attempted a subprocess or live agent read")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(staleness, "_http_get_json", forbidden)


def _generation(
    container_id: str,
    *,
    container: str = SERVICE,
    image: str = IMAGE,
    revision: str = REVISION,
) -> ModelLaneGeneration:
    return ModelLaneGeneration(
        container=container, container_id=container_id, image=image, revision=revision
    )


def _clock_from(
    start: datetime,
) -> tuple[Callable[[], datetime], Callable[[float], None]]:
    now = [start]

    def clock() -> datetime:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += timedelta(seconds=seconds)

    return clock, sleep


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def _publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    status: str = "success",
    service: str = SERVICE,
    outcome: str = "recovered",
    container: str = SERVICE,
    image: str = IMAGE,
    revision: str = REVISION,
    agent_url: str = AGENT,
    correlation_id: str = CID,
) -> tuple[ModelLaneGeneration, ModelLaneGeneration, str]:
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    converged = _generation(OLD_ID, container=container)
    running = _generation(NEW_ID, container=container, image=image, revision=revision)
    reads = iter([converged, running])
    clock, sleep = _clock_from(START)
    requests: list[tuple[str, float]] = []
    body = json.dumps(
        {
            "status": status,
            "completed_at": START.isoformat() if status != "in_progress" else "",
            "verify_recreate": [{"service": service, "outcome": outcome}],
        }
    )

    def opener(url: str, timeout: float) -> tuple[int, str]:
        requests.append((url, timeout))
        # Prove we wait past convergence rather than just read one success record.
        if len(requests) == 1:
            return 200, '{"status": "in_progress"}'
        return 200, body

    guard.publish_onex_api_generation(
        verdict=sibling.Verdict(notes=["onex-api carries the merge"]),
        container=container,
        agent_url=agent_url,
        correlation_id=correlation_id,
        deadline_monotonic=125.0,
        monotonic=lambda: 100.0 + (clock() - START).total_seconds(),
        clock=clock,
        sleep=sleep,
        read_generation=lambda _c: next(reads),
        read_service=lambda _c: SERVICE,
        opener=opener,
        request_timeout_seconds=2.5,
    )
    values = _outputs(output)
    published = parse_generation(values["generation"])
    assert clock() <= START + timedelta(seconds=25)
    if not agent_url or not correlation_id:
        assert requests == []
    else:
        assert requests
        assert all(url == f"{AGENT}/job/{CID}" for url, _ in requests)
        assert all(timeout == 2.5 for _, timeout in requests)
    return published, running, values["evidence"]


@pytest.mark.parametrize("container", [SERVICE, "omninode-dev-202-onex-api"])
def test_recorded_recovered_recreate_publishes_the_job_end_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, container: str
) -> None:
    published, running, evidence = _publish(monkeypatch, tmp_path, container=container)

    assert published == running
    assert generation_check(published, running).ok
    assert "onex-api carries the merge" in evidence
    assert "ended success" in evidence
    assert "rebound" in evidence


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "failed"},
        {"service": "runtime-effects"},
        {"outcome": "still_failing"},
        {"image": "sha256:" + "2" * 64},
        {"revision": "f" * 40},
        {"status": "in_progress"},
        {"agent_url": ""},
        {"correlation_id": ""},
        {"agent_url": "", "correlation_id": ""},
    ],
)
def test_other_movements_keep_the_converged_generation_and_fail_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changes: dict[str, str]
) -> None:
    published, running, _ = _publish(monkeypatch, tmp_path, **changes)

    assert published == _generation(OLD_ID)
    assert not generation_check(published, running).ok


@pytest.mark.parametrize("remaining", [0.0, 0.25, 300.0, -5.0])
def test_job_end_deadline_uses_only_the_remaining_guard_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remaining: float
) -> None:
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github-output"))
    deadlines: list[datetime] = []

    def wait(**kwargs: Any) -> staleness.ModelJobEnd:
        deadlines.append(kwargs["deadline"])
        assert kwargs["clock"]() == START
        assert kwargs["request_timeout_seconds"] == 1.5
        return staleness.ModelJobEnd(ended=False, correlation_id=CID, reason="deadline")

    monkeypatch.setattr(sibling, "wait_for_agent_job_end", wait)
    guard.publish_onex_api_generation(
        verdict=sibling.Verdict(),
        container=SERVICE,
        agent_url=AGENT,
        correlation_id=CID,
        deadline_monotonic=100.0 + remaining,
        monotonic=lambda: 100.0,
        clock=lambda: START,
        sleep=lambda _s: None,
        read_generation=lambda _c: _generation(OLD_ID),
        read_service=lambda _c: SERVICE,
        request_timeout_seconds=1.5,
    )

    assert deadlines == [START + timedelta(seconds=max(0.0, remaining))]
    assert deadlines[0].tzinfo is UTC


def test_unreadable_generation_is_not_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    def unreadable(_container: str) -> ModelLaneGeneration:
        raise ValueError("unreadable container")

    guard.publish_onex_api_generation(
        verdict=sibling.Verdict(notes=["converged"]),
        container=SERVICE,
        agent_url=AGENT,
        correlation_id=CID,
        deadline_monotonic=125.0,
        monotonic=lambda: 100.0,
        clock=lambda: START,
        sleep=lambda _s: None,
        read_generation=unreadable,
        read_service=lambda _c: SERVICE,
    )

    assert _outputs(output) == {"evidence": "converged"}


def test_main_publishes_the_shared_binding_and_single_line_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(guard, "read_compose_project", lambda _c: "omnibase-infra")
    monkeypatch.setattr(
        guard, "read_container_labels", lambda _c: {guard.REVISION_LABEL: REVISION}
    )
    monkeypatch.setattr(guard, "read_containment", lambda _e, _a: "identical")
    monotonic = iter([100.0, 100.0, 115.0, 120.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic))
    calls: list[dict[str, Any]] = []

    def bind(**kwargs: Any) -> sibling.ModelBoundGeneration:
        calls.append(kwargs)
        now = kwargs["clock"]()
        assert now.tzinfo is UTC
        assert now + timedelta(seconds=9) <= kwargs["deadline"]
        assert kwargs["deadline"] <= now + timedelta(seconds=10)
        return sibling.ModelBoundGeneration(_generation(NEW_ID), "job ended\nrebound\r")

    monkeypatch.setattr(guard, "bind_sibling_generation", bind)
    assert (
        guard.main(
            [
                "--expect-revision",
                REVISION,
                "--agent-url",
                AGENT,
                "--correlation-id",
                CID,
                "--agent-timeout-seconds",
                "2.5",
                "--wait-timeout",
                "30s",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0]["agent_url"] == AGENT
    assert calls[0]["correlation_id"] == CID
    assert calls[0]["container"] == SERVICE
    assert calls[0]["request_timeout_seconds"] == 2.5
    assert calls[0]["read_generation"] is read_lane_generation
    assert calls[0]["read_service"] is staleness.read_compose_service
    assert calls[0]["sleep"] is time.sleep
    values = _outputs(output)
    assert parse_generation(values["generation"]) == _generation(NEW_ID)
    assert "job ended rebound" in values["evidence"]
    assert len(output.read_text().splitlines()) == 2


def test_non_ok_main_keeps_the_existing_generation_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(guard, "read_compose_project", lambda _c: "omnibase-infra")
    monkeypatch.setattr(
        guard, "read_container_labels", lambda _c: {guard.REVISION_LABEL: REVISION}
    )
    monkeypatch.setattr(guard, "read_containment", lambda _e, _a: "behind")
    monkeypatch.setattr(guard, "read_lane_generation", lambda _c: _generation(OLD_ID))
    reads: list[str] = []
    original = guard._write_output_generation

    def write(container: str) -> None:
        reads.append(container)
        original(container)

    monkeypatch.setattr(guard, "_write_output_generation", write)
    assert guard.main(["--expect-revision", REVISION, "--wait-timeout", "0s"]) == 1
    assert reads == [SERVICE]
    values = _outputs(output)
    assert parse_generation(values["generation"]) == _generation(OLD_ID)
    assert "ONEX_API_NOT_CONVERGED" in values["evidence"]


def test_agent_cli_defaults_are_optional() -> None:
    args = guard._build_parser().parse_args(["--expect-revision", REVISION])
    assert args.agent_url == ""
    assert args.correlation_id == ""
    assert args.agent_timeout_seconds == 10.0


def test_workflow_wires_job_end_inputs_without_moving_the_ceiling() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    trigger = workflow["jobs"]["trigger-delivery"]
    assert trigger["outputs"]["correlation_id"] == (
        "${{ steps.publish.outputs.correlation_id }}"
    )
    verify = workflow["jobs"]["verify-onex-api-delivered"]
    assert verify["timeout-minutes"] == 45
    steps = verify["steps"]
    converge = next(step for step in steps if step.get("id") == "converge")
    assert '--agent-url "$DEPLOY_AGENT_URL"' in converge["run"]
    assert '--correlation-id "$CORRELATION_ID"' in converge["run"]
    assert converge["env"]["CORRELATION_ID"] == (
        "${{ needs.trigger-delivery.outputs.correlation_id }}"
    )
    assert converge["env"]["DEPLOY_AGENT_URL"] == "${{ env.DEPLOY_AGENT_URL }}"
    lane_env = next(step for step in steps if "--write-lane-env" in step.get("run", ""))
    assert steps.index(lane_env) < steps.index(converge)
    assert "uv run python scripts/ci/check_dev_lane_staleness.py" in lane_env["run"]
    assert "--lane compose-dev" in lane_env["run"]


def test_lane_env_collision_preserves_the_probe_explicit_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    assert write_lane_env(receipt_lane="compose-dev") == 0
    exported = _outputs(env_file)
    targets = targets_for_receipt_lane(load_table(), "compose-dev")
    assert exported == job_env(targets)
    assert exported["DEPLOY_AGENT_URL"] == targets.deploy_agent_url
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["verify-onex-api-delivered"]["steps"]
    probe = next(step for step in steps if step.get("id") == "probe")
    collisions = exported.keys() & probe["env"].keys()
    assert collisions == {
        "DEV_LANE_MAIN_URL",
        "DEV_LANE_EFFECTS_URL",
        "DEV_LANE_PROJECTION_URL",
    }
    effective = exported | probe["env"]
    assert effective["DEV_LANE_CONTAINER"] == SERVICE
    for name in collisions:
        assert effective[name] == probe["env"][name]
