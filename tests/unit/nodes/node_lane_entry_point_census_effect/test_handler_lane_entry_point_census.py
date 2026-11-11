# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane entry-point census records each runtime-image command (OMN-21923, T24)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers import handler_lane_entry_point_census as handler_module
from omnibase_infra.handlers.handler_lane_entry_point_census import (
    HandlerLaneEntryPointCensus,
    entry_point_of,
    load_policy,
)
from omnibase_infra.models.lane_entry_point_census import (
    ModelLaneEntryPointCensusRequest,
    ModelLaneEntryPointCensusResult,
)

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "inspect_planted.json"
POLICY = (
    Path(__file__).parents[4] / "deploy" / "lane-census" / "runtime-entry-points.yaml"
)
PLANTED = "proj-a-ctx-consumer"
WRITER = "proj-a-delegation-writer"


def _run(policy: Path = POLICY, **extra: str) -> ModelLaneEntryPointCensusResult:
    request = ModelLaneEntryPointCensusRequest(
        policy_path=str(policy), inspect_json_path=str(FIXTURE), **extra
    )
    return HandlerLaneEntryPointCensus().handle(request)


def test_planted_python_dash_m_container_is_reported_by_name() -> None:
    result = _run()
    assert result.success, result.error_message
    lane_a = next(c for c in result.censuses if c.lane == "proj-a")
    assert PLANTED in lane_a.remove
    assert WRITER in lane_a.remove
    planted = next(e for e in lane_a.containers if e.container == PLANTED)
    assert planted.verdict == "remove"
    assert planted.entry_point.startswith("python -m ")
    assert planted.command.startswith(
        "/usr/bin/tini -- /app/entrypoint-runtime.sh python -m "
    )


def test_every_runtime_image_container_lists_its_command() -> None:
    result = _run()
    lane_a = next(c for c in result.censuses if c.lane == "proj-a")
    assert lane_a.container_count == 6  # postgres is not a runtime-image container
    assert lane_a.allowed_count == 3
    assert lane_a.remove_count == 3
    assert all(e.command and e.cmd for e in lane_a.containers)
    assert "proj-a-postgres" not in {e.container for e in lane_a.containers}


def test_lane_without_a_python_dash_m_command_holds_only_the_forwarder_to_remove() -> (
    None
):
    result = _run()
    lane_b = next(c for c in result.censuses if c.lane == "proj-b")
    assert lane_b.remove == ("proj-b-gateway-forwarder",)
    assert (lane_b.container_count, lane_b.allowed_count) == (2, 1)


def test_runtime_image_run_under_an_unmatched_tag_is_found_by_its_launcher() -> None:
    forwarder = next(
        e
        for c in _run().censuses
        for e in c.containers
        if e.container == "proj-b-gateway-forwarder"
    )
    assert forwarder.image == "gateway-image:build-1"
    assert forwarder.entry_point == "onex-gateway-forwarder"
    assert forwarder.verdict == "remove"


def test_launcher_matching_is_a_declared_value_not_code(tmp_path: Path) -> None:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    policy["launcher_marks_runtime_image"] = False
    off = tmp_path / "policy.yaml"
    off.write_text(yaml.safe_dump(policy), encoding="utf-8")
    lane_b = next(c for c in _run(off).censuses if c.lane == "proj-b")
    assert lane_b.container_count == 1
    assert lane_b.remove == ()


def test_table_is_regenerated_from_the_census() -> None:
    table = _run().table_markdown
    assert "| `onex-runtime` | 3 |" in table
    assert "| `python -m ...consumer` | 1 | proj-a-ctx-consumer |" in table
    assert (
        "| `python -m omnimarket...handler_*` | 1 | proj-a-delegation-writer |" in table
    )
    assert "| `uvicorn` | 1 | proj-a-resolver | entry point to remove |" in table


def test_allowed_list_in_data_changes_the_verdict_without_a_code_change(
    tmp_path: Path,
) -> None:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    policy["allowed_commands"] = ["onex-runtime", "uvicorn"]
    widened = tmp_path / "policy.yaml"
    widened.write_text(yaml.safe_dump(policy), encoding="utf-8")
    lane_a = next(c for c in _run(widened).censuses if c.lane == "proj-a")
    assert "proj-a-resolver" not in lane_a.remove
    assert lane_a.remove_count == 2


def test_manifest_names_the_lane_by_its_compose_project(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"lanes": {"dev": {"compose_project": "proj-a"}}}),
        encoding="utf-8",
    )
    lanes = {c.lane for c in _run(manifest_path=str(manifest)).censuses}
    assert lanes == {"dev", "proj-b"}


def test_unreadable_policy_is_a_refusal_not_an_empty_census(tmp_path: Path) -> None:
    result = _run(tmp_path / "absent.yaml")
    assert result.success is False
    assert result.censuses == ()
    assert result.error_message


def test_an_undeclared_policy_key_is_refused(tmp_path: Path) -> None:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    policy["surprise"] = True
    bad = tmp_path / "policy.yaml"
    bad.write_text(yaml.safe_dump(policy), encoding="utf-8")
    assert _run(bad).success is False


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("tini", "--", "/app/entrypoint-runtime.sh", "onex-runtime"), "onex-runtime"),
        (("python3.12", "-m", "pkg.mod", "--x"), "python -m pkg.mod"),
        (("/usr/local/bin/uvicorn", "app:a"), "uvicorn"),
        ((), ""),
    ],
)
def test_entry_point_of(argv: tuple[str, ...], expected: str) -> None:
    prefix = ("tini", "--", "/app/entrypoint-runtime.sh")
    assert entry_point_of(argv, prefix) == expected


def test_fixture_is_the_recorded_inspect_shape() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert {"Name", "Config", "State"} <= set(data[0])


def _completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_live_read_keeps_the_containers_inspect_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = json.loads(FIXTURE.read_text(encoding="utf-8"))[:2]
    replies = [
        _completed("aaa\nbbb\ngone\n"),
        _completed(json.dumps(kept), 1, "error: no such object: gone"),
    ]
    monkeypatch.setattr(handler_module, "_docker", lambda *_: replies.pop(0))
    assert handler_module.read_live_inspection("", None) == kept


def test_live_read_refuses_when_inspect_prints_no_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = [_completed("aaa\n"), _completed("", 1, "daemon unreachable")]
    monkeypatch.setattr(handler_module, "_docker", lambda *_: replies.pop(0))
    with pytest.raises(ValueError, match="docker inspect exited 1"):
        handler_module.read_live_inspection("", None)


def test_launcher_is_matched_by_program_basename() -> None:
    prefix = ("/usr/bin/tini", "--", "/app/entrypoint-runtime.sh")
    argv = ("tini", "--", "/app/entrypoint-runtime.sh", "onex-runtime")
    assert entry_point_of(argv, prefix) == "onex-runtime"
    assert entry_point_of(("tini", "x", "onex-runtime"), prefix) == "tini"


def test_bus_request_needs_no_docker_or_host_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_policy(POLICY)
    inspection = tuple(json.loads(FIXTURE.read_text()))

    def unexpected_io(*_: object) -> None:
        pytest.fail("an inline bus request must not read host files or Docker")

    monkeypatch.setattr(handler_module, "read_live_inspection", unexpected_io)
    monkeypatch.setattr(handler_module, "load_policy", unexpected_io)
    result = HandlerLaneEntryPointCensus().handle(
        ModelLaneEntryPointCensusRequest(policy=policy, inspection=inspection)
    )
    assert result.success, result.error_message
    assert len(result.censuses) == 2
    assert PLANTED in next(c for c in result.censuses if c.lane == "proj-a").remove


def test_missing_or_ambiguous_policy_is_a_refusal() -> None:
    handler = HandlerLaneEntryPointCensus()
    assert not handler.handle(ModelLaneEntryPointCensusRequest(inspection=())).success
    assert not handler.handle(
        ModelLaneEntryPointCensusRequest(
            policy=load_policy(POLICY), policy_path=str(POLICY), inspection=()
        )
    ).success


def test_ambiguous_or_malformed_inspection_is_a_refusal(tmp_path: Path) -> None:
    assert (
        not HandlerLaneEntryPointCensus()
        .handle(
            ModelLaneEntryPointCensusRequest(
                policy=load_policy(POLICY),
                inspection=(),
                inspect_json_path=str(FIXTURE),
            )
        )
        .success
    )
    malformed = tmp_path / "inspect.json"
    malformed.write_text("[1]")
    assert (
        not HandlerLaneEntryPointCensus()
        .handle(
            ModelLaneEntryPointCensusRequest(
                policy_path=str(POLICY), inspect_json_path=str(malformed)
            )
        )
        .success
    )
