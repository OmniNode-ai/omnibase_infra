# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Lane container memory event: builder, and the census pass's exit status (OMN-19959).

On 2026-09-28 every OOM kill on the .202 lab host was a cgroup kill that no
monitor read, and two lane consumers ran pinned at their 256 MiB limit
(``memory.events`` ``max 1053`` and ``max 565``) with no signal anywhere. These
tests pin the event that makes that visible and the exit codes that make a
failing pass loud:

* one record per lane container carrying every schema field, with the CI jobs
  that ran on this host's runners in the same window named by repo and run id;
* ``record_key`` and ``alert_key`` identical when the same fixture is built twice;
* an ``oom_kill`` rise, or a ``max`` rise in two consecutive passes, exits 31;
* an unset broker container, or a produce that fails, exits 6;
* a pass over unchanged counters exits 0, so the alert clears.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_CENSUS_SH = _SCRIPTS / "lane-census-check.sh"
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lane_container_memory"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mem = _load("lane_container_memory_event", _SCRIPTS / "lane_container_memory_event.py")
fake = _load(
    "fake_docker_engine_omn19959",
    Path(__file__).resolve().parent / "fake_docker_engine_omn19959.py",
)

BOOT_ID = fake.BOOT_ID
REDPANDA_ID = "a03d4887a7266a85e1df700cb8b9b0004084b2322718f674b5c9abbf3dcefafa"
CONSUMER_ID = "5e1f0c2b7d9a4e6f8b3c1d0a2e4f6b8c9d1e3f5a7b9c2d4e6f8a1b3c5d7e9f0a"


def _runs(*logs: dict[str, str]) -> list[dict[str, str | None]]:
    """Parse fixture worker logs the way the collector does as it streams them."""
    runs = []
    for log in logs:
        run = mem.parse_worker_log(
            log["text"], runner_name=log["runner_name"], log_name=log["log_name"]
        )
        if run is not None:
            runs.append(run)
    return runs


def _observation() -> dict[str, Any]:
    return {
        "host_boot_id": BOOT_ID,
        "boot_time": "2026-09-28T12:10:00Z",
        "read_at": "2026-09-28T19:00:00.000000Z",
        "containers": [
            {
                "container_id": REDPANDA_ID,
                "container_name": "omnibase-infra-sim-202-redpanda",
                "lane": "sim-202",
                "started_at": "2026-09-28T12:11:26.55947631Z",
                "memory_max": "max\n",
                "memory_peak": "2114498560\n",
                "memory_events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
            },
            {
                "container_id": CONSUMER_ID,
                "container_name": "omninode-sim-202-runtime-effects",
                "lane": "sim-202",
                "started_at": "2026-09-28T12:12:00Z",
                "memory_max": "268435456\n",
                "memory_peak": "268435456\n",
                "memory_events": "low 0\nhigh 0\nmax 1060\noom 1\noom_kill 1\noom_group_kill 0\n",
            },
        ],
        "journal_oom_kills": [],
        "worker_runs": _runs(
            {
                "runner_name": "omnipc2-ci-runner-13",
                "log_name": "Worker_20260928-185104-utc.log",
                "text": fake.worker_log(
                    repo="OmniNode-ai/knowledge-base-internal",
                    run_id="36468089372",
                    started="2026-09-28 18:51:04Z",
                    completed="2026-09-28 18:51:33Z",
                ),
            },
            {
                "runner_name": "omnipc2-ci-runner-6",
                "log_name": "Worker_20260928-185900-utc.log",
                "text": fake.worker_log(
                    repo="OmniNode-ai/omnimarket",
                    run_id="36454760449",
                    started="2026-09-28 18:59:00Z",
                    completed=None,
                ),
            },
            {
                # Wholly before the window: completed 17:05, window opens 18:00.
                "runner_name": "omnipc2-ci-runner-13",
                "log_name": "Worker_20260928-170000-utc.log",
                "text": fake.worker_log(
                    repo="OmniNode-ai/omninode_infra",
                    run_id="36425734123",
                    started="2026-09-28 17:00:00Z",
                    completed="2026-09-28 17:05:00Z",
                ),
            },
        ),
    }


def _previous_state() -> dict[str, Any]:
    return {
        "host": "omnipc2",
        "host_boot_id": BOOT_ID,
        "window_end": "2026-09-28T18:00:00.000000Z",
        "containers": {
            REDPANDA_ID: {"max_total": 0, "oom_kill_total": 0, "max_delta": 0},
            CONSUMER_ID: {"max_total": 1053, "oom_kill_total": 0, "max_delta": 5},
        },
    }


def _build(
    observation: dict[str, Any] | None = None,
    previous: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    return mem.build_event(
        host="omnipc2",
        observation=observation if observation is not None else _observation(),
        previous_state=previous if previous is not None else _previous_state(),
    )


def _record(event: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [r for r in event["records"] if r["container_name"] == name]
    assert len(matches) == 1, event["records"]
    return matches[0]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_builder_emits_one_record_per_container_with_every_field() -> None:
    event, _state, _alerts = _build()

    assert set(event) == set(mem.ENVELOPE_FIELDS)
    assert event["schema_version"] == "1.0.0"
    assert event["event_type"] == "lane-container-memory-observation"
    assert event["host"] == "omnipc2"
    assert event["host_boot_id"] == BOOT_ID
    assert event["window_start"] == "2026-09-28T18:00:00.000000Z"
    assert event["window_end"] == "2026-09-28T19:00:00.000000Z"
    assert len(event["records"]) == 2
    for record in event["records"]:
        assert set(record) == set(mem.RECORD_FIELDS)

    redpanda = _record(event, "omnibase-infra-sim-202-redpanda")
    assert redpanda["limit_bytes"] is None, "memory.max 'max' is no limit, not 0"
    assert redpanda["peak_bytes"] == 2114498560
    # The peak is a lifetime high-water mark: its window opens at the start.
    assert redpanda["peak_window_start"] == "2026-09-28T12:11:26.559476Z"
    assert redpanda["peak_window_start"] == redpanda["container_started_at"]
    assert redpanda["peak_window_end"] == event["window_end"]

    consumer = _record(event, "omninode-sim-202-runtime-effects")
    assert consumer["limit_bytes"] == 268435456
    assert consumer["max_total"] == 1060
    assert consumer["max_delta"] == 7
    assert consumer["oom_kill_total"] == 1
    assert consumer["oom_kill_delta"] == 1


def test_ci_runs_are_attributed_from_the_runner_worker_logs() -> None:
    event, _state, _alerts = _build()
    runs = {(r["repo"], r["run_id"]): r for r in event["ci_runs"]}

    completed = runs[("OmniNode-ai/knowledge-base-internal", "36468089372")]
    assert completed["runner_name"] == "omnipc2-ci-runner-13"
    assert completed["job_started_at"] == "2026-09-28T18:51:04.000000Z"
    assert completed["job_completed_at"] == "2026-09-28T18:51:33.000000Z"

    running = runs[("OmniNode-ai/omnimarket", "36454760449")]
    assert running["job_completed_at"] is None, "a running job has no completion"

    assert ("OmniNode-ai/omninode_infra", "36425734123") not in runs, (
        "a job that completed before the window opened was attributed to it"
    )
    assert len(event["ci_runs"]) == 2


def test_building_the_same_fixture_twice_gives_the_same_keys() -> None:
    first, first_state, first_alerts = _build()
    second, second_state, second_alerts = _build()
    assert first == second
    assert first_state == second_state
    assert first_alerts == second_alerts

    consumer = _record(first, "omninode-sim-202-runtime-effects")
    expected_record = hashlib.sha256(
        f"{BOOT_ID}|{CONSUMER_ID}|{first['window_end']}".encode()
    ).hexdigest()
    expected_alert = hashlib.sha256(
        f"{BOOT_ID}|{CONSUMER_ID}|1|1060".encode()
    ).hexdigest()
    assert consumer["record_key"] == expected_record
    assert consumer["alert_key"] == expected_alert


def test_oom_kill_and_a_sustained_limit_hit_alert() -> None:
    _event, _state, alerts = _build()
    joined = "\n".join(alerts)
    assert (
        "OOM_KILL lane=sim-202 container=omninode-sim-202-runtime-effects delta=1"
        in joined
    )
    assert "LIMIT_HIT lane=sim-202 container=omninode-sim-202-runtime-effects" in joined
    assert "peak/limit=268435456/268435456" in joined
    assert "omnibase-infra-sim-202-redpanda" not in joined


def test_a_single_limit_hit_is_not_yet_sustained() -> None:
    previous = _previous_state()
    previous["containers"][CONSUMER_ID]["max_delta"] = 0
    observation = _observation()
    observation["containers"][1]["memory_events"] = "max 1060\noom 0\noom_kill 0\n"
    previous["containers"][CONSUMER_ID]["oom_kill_total"] = 0
    _event, state, alerts = _build(observation, previous)
    assert alerts == []
    assert state["containers"][CONSUMER_ID]["max_delta"] == 7, (
        "the delta must be carried, so the NEXT rise is the sustained one"
    )


def test_flat_counters_are_quiet_so_the_alert_clears() -> None:
    event, state, _alerts = _build()
    later = _observation()
    later["read_at"] = "2026-09-28T20:00:00.000000Z"
    event2, _state2, alerts2 = _build(later, state)
    assert alerts2 == []
    consumer = _record(event2, "omninode-sim-202-runtime-effects")
    assert consumer["max_delta"] == 0 and consumer["oom_kill_delta"] == 0
    assert event2["window_start"] == event["window_end"]


def test_first_pass_after_a_boot_opens_the_window_at_boot() -> None:
    previous = _previous_state()
    previous["host_boot_id"] = "0f0f0f0f-0000-0000-0000-000000000000"
    event, _state, alerts = _build(previous=previous)
    assert event["window_start"] == "2026-09-28T12:10:00.000000Z"
    consumer = _record(event, "omninode-sim-202-runtime-effects")
    # Started after the boot, so every count since then is in this window.
    assert consumer["max_delta"] == 1060
    assert consumer["oom_kill_delta"] == 1
    assert any(a.startswith("OOM_KILL") for a in alerts)


def test_an_unreadable_counter_is_refused_never_a_zero() -> None:
    observation = _observation()
    observation["containers"][0]["memory_events"] = "low 0\nhigh 0\nmax 3\n"
    with pytest.raises(mem.MemoryObservationError, match="oom_kill"):
        _build(observation)


def test_a_completed_worker_log_that_names_no_job_is_refused() -> None:
    text = (
        "[2026-09-28 18:51:04Z INFO HostContext] start\n"
        "[2026-09-28 18:51:33Z INFO Worker] Job completed.\n"
    )
    with pytest.raises(mem.MemoryObservationError, match="names no repository"):
        mem.parse_worker_log(
            text,
            runner_name="omnipc2-ci-runner-13",
            log_name="Worker_20260928-185104-utc.log",
        )


def test_a_worker_log_still_starting_is_read_next_pass() -> None:
    assert (
        mem.parse_worker_log(
            "[2026-09-28 18:59:00Z INFO HostContext] start\n",
            runner_name="omnipc2-ci-runner-6",
            log_name="Worker_20260928-185900-utc.log",
        )
        is None
    )


def test_a_malformed_worker_run_is_refused() -> None:
    observation = _observation()
    del observation["worker_runs"][0]["run_id"]
    with pytest.raises(mem.MemoryObservationError, match="missing"):
        _build(observation)


def test_a_kill_hidden_by_a_restart_is_counted_from_the_journal() -> None:
    """The restarted cgroup reads oom_kill 0; the kernel journal named it three times."""
    observation = _observation()
    observation["containers"][0]["memory_events"] = "max 0\noom 0\noom_kill 0\n"
    observation["journal_oom_kills"] = [
        {
            "container_id": REDPANDA_ID,
            "container_name": "omnibase-infra-sim-202-redpanda",
            "lane": "sim-202",
            "count": 3,
        }
    ]
    event, _state, alerts = _build(observation)
    redpanda = _record(event, "omnibase-infra-sim-202-redpanda")
    assert redpanda["oom_kill_delta"] == 3
    assert redpanda["oom_kill_total"] == 0, "the total stays the cgroup's own counter"
    assert any(
        a.startswith(
            "OOM_KILL lane=sim-202 container=omnibase-infra-sim-202-redpanda delta=3"
        )
        for a in alerts
    )


def test_a_killed_container_that_is_not_running_is_still_alerted() -> None:
    observation = _observation()
    observation["journal_oom_kills"] = [
        {
            "container_id": "d" * 64,
            "container_name": "omninode-sim-202-runtime",
            "lane": "sim-202",
            "count": 1,
        }
    ]
    _event, _state, alerts = _build(observation)
    assert (
        "OOM_KILL lane=sim-202 container=omninode-sim-202-runtime delta=1 "
        "(container not running)"
    ) in alerts


def test_the_event_matches_the_published_v1_fixture() -> None:
    """Task 5b builds its wire model against this file; a drift here is a schema change."""
    event, _state, _alerts = _build()
    published = json.loads((_FIXTURES / "event.v1.json").read_text(encoding="utf-8"))
    assert event == published
    mem.validate_event(published)


def test_validate_event_refuses_an_extra_or_missing_key() -> None:
    event, _state, _alerts = _build()
    extra = copy.deepcopy(event)
    extra["records"][0]["usage_bytes"] = 1
    with pytest.raises(mem.MemoryObservationError, match="unexpected"):
        mem.validate_event(extra)
    missing = copy.deepcopy(event)
    del missing["records"][0]["peak_bytes"]
    with pytest.raises(mem.MemoryObservationError, match="missing"):
        mem.validate_event(missing)


# ---------------------------------------------------------------------------
# lane-census-check.sh --memory, end to end against a fake host
# ---------------------------------------------------------------------------


def _containers(consumer_events: str) -> list[Any]:
    return [
        fake.FakeContainer(
            cid=REDPANDA_ID,
            name="omnibase-infra-sim-202-redpanda",
            pid=4101,
            project=fake.SIM_PROJECT,
        ),
        fake.FakeContainer(
            cid=CONSUMER_ID,
            name="omninode-sim-202-runtime-effects",
            pid=4102,
            project=fake.SIM_PROJECT,
            memory_max="268435456\n",
            memory_peak="268435456\n",
            memory_events=consumer_events,
            image="omninode-runtime:x",
        ),
        fake.FakeContainer(
            cid="c" * 64,
            name="omnipc2-ci-runner-13",
            pid=4103,
            project="omnipc2-ci-runner",
            image="omninode-runner:x",
            worker_logs={
                "Worker_20260928-185104-utc.log": (
                    fake.worker_log(
                        repo="OmniNode-ai/knowledge-base-internal",
                        run_id="36468089372",
                        started="2026-09-28 18:51:04Z",
                        completed="2026-09-28 18:51:33Z",
                    ),
                    1790621493,
                ),
            },
        ),
    ]


class _Pass:
    """Run the real census script with --memory against a fake host."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.produced = tmp_path / "produced.ndjson"
        self.docker_calls = tmp_path / "docker-calls.log"
        shim = self.bin / "docker"
        shim.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$*" >> "{self.docker_calls}"\n'
            'if [[ "$1" == exec ]]; then\n'
            f'  cat >> "{self.produced}"\n'
            '  exit "${STUB_PRODUCE_RC:-0}"\n'
            "fi\n"
            "exit 1\n"
        )
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self.state = self.home / ".local/state/onex/lane-container-memory-state.json"
        self.journal = tmp_path / "journal.txt"
        self.journal.write_text("")
        fake.write_journalctl_stub(self.bin, self.journal)

    def run(
        self,
        host: Any,
        *,
        broker: str | None = "omnibase-infra-dev-202-redpanda",
        produce_rc: int = 0,
        extra: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "HOME": str(self.home),
            "LANE_CENSUS_PYTHON": sys.executable,
            "LANE_CENSUS_HOST": "omnipc2",
            "LANE_MANIFEST": str(_FIXTURES / "lane-manifest.yaml"),
            "LANE_MEMORY_RUNNER_FLEET_CONFIG": str(_FIXTURES / "runner_fleet.yaml"),
            "STUB_PRODUCE_RC": str(produce_rc),
            **host.env(),
        }
        if broker is not None:
            env["LANE_MEMORY_BROKER_CONTAINER"] = broker
        return subprocess.run(
            ["/bin/bash", str(_CENSUS_SH), "--memory", *(extra or [])],
            capture_output=True,
            text=True,
            env=env,
            cwd=_REPO,
            timeout=120,
            check=False,
        )


_CLEAN = "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n"


@pytest.fixture
def census(tmp_path: Path) -> _Pass:
    return _Pass(tmp_path)


def _host(tmp_path: Path, consumer_events: str) -> Any:
    return fake.FakeHost(tmp_path / "host", _containers(consumer_events))


def test_a_clean_pass_publishes_one_event_and_exits_0(
    census: _Pass, tmp_path: Path
) -> None:
    host = _host(tmp_path, _CLEAN)
    try:
        result = census.run(host)
    finally:
        host.close()
    assert result.returncode == 0, result.stderr
    lines = census.produced.read_text().splitlines()
    assert len(lines) == 1, "one event per host per pass"
    event = json.loads(lines[0])
    mem.validate_event(event)
    assert event["host"] == "omnipc2"
    assert {r["container_name"] for r in event["records"]} == {
        "omnibase-infra-sim-202-redpanda",
        "omninode-sim-202-runtime-effects",
    }, "runner containers are not lane containers"
    assert [(r["repo"], r["run_id"]) for r in event["ci_runs"]] == [
        ("OmniNode-ai/knowledge-base-internal", "36468089372")
    ]
    call = census.docker_calls.read_text()
    assert "exec -i omnibase-infra-dev-202-redpanda sh -c" in call
    assert (
        'rpk topic produce "onex.evt.omnibase-infra.lane-container-memory.v1"' in call
    )
    assert "DEV_KAFKA_SASL_USERNAME" in call, "the SASL pair is passed by NAME"
    assert census.state.exists(), "a published pass advances the state"


def test_an_oom_kill_fails_the_pass_naming_the_container(
    census: _Pass, tmp_path: Path
) -> None:
    host = _host(tmp_path, "max 3\noom 1\noom_kill 1\n")
    try:
        result = census.run(host)
    finally:
        host.close()
    assert result.returncode == 31, result.stderr
    assert (
        "OOM_KILL lane=sim-202 container=omninode-sim-202-runtime-effects"
        in result.stderr
    )


def test_a_kill_the_restart_hid_from_the_cgroup_fails_the_pass(
    census: _Pass, tmp_path: Path
) -> None:
    """The lab case: counters read zero after the restart; the journal names the kill."""
    host = _host(tmp_path, _CLEAN)
    census.journal.write_text(fake.oom_kill_line(CONSUMER_ID, 1790625222.276671))
    try:
        result = census.run(host)
    finally:
        host.close()
    assert result.returncode == 31, result.stderr
    assert (
        "OOM_KILL lane=sim-202 container=omninode-sim-202-runtime-effects"
        in result.stderr
    )


def test_a_sustained_limit_hit_fails_and_a_restored_limit_clears(
    census: _Pass, tmp_path: Path
) -> None:
    host = _host(tmp_path, "max 10\noom 0\noom_kill 0\n")
    try:
        first = census.run(host)
        assert first.returncode == 0, (
            "one pass at the limit is not yet sustained: " + first.stderr
        )

        host.containers[1].memory_events = "max 20\noom 0\noom_kill 0\n"
        host.write_tree()
        second = census.run(host)
        assert second.returncode == 31, second.stderr
        assert (
            "LIMIT_HIT lane=sim-202 container=omninode-sim-202-runtime-effects"
            in second.stderr
        )

        third = census.run(host)  # counters unchanged: the limit was restored
        assert third.returncode == 0, "the alert did not clear: " + third.stderr
    finally:
        host.close()


def test_an_unset_broker_container_exits_6_and_keeps_the_window(
    census: _Pass, tmp_path: Path
) -> None:
    host = _host(tmp_path, _CLEAN)
    try:
        result = census.run(host, broker=None)
    finally:
        host.close()
    assert result.returncode == 6, result.stderr
    assert "LANE_MEMORY_BROKER_CONTAINER is unset" in result.stderr
    assert not census.state.exists(), "an unpublished window must not advance the state"


def test_a_failed_produce_exits_6(census: _Pass, tmp_path: Path) -> None:
    host = _host(tmp_path, _CLEAN)
    try:
        result = census.run(host, produce_rc=1)
    finally:
        host.close()
    assert result.returncode == 6, result.stderr
    assert not census.state.exists()


def test_an_unreadable_counter_exits_7(census: _Pass, tmp_path: Path) -> None:
    host = _host(tmp_path, _CLEAN)
    try:
        (
            host.sysfs_root / f"system.slice/docker-{CONSUMER_ID}.scope/memory.events"
        ).unlink()
        result = census.run(host)
    finally:
        host.close()
    assert result.returncode == 7, result.stderr
    assert "memory.events" in result.stderr
    assert not census.produced.exists(), "nothing is published from a partial read"


def test_dry_run_neither_publishes_nor_advances(census: _Pass, tmp_path: Path) -> None:
    host = _host(tmp_path, _CLEAN)
    out = tmp_path / "event.json"
    try:
        result = census.run(host, extra=["--dry-run", "--memory-event-out", str(out)])
    finally:
        host.close()
    assert result.returncode == 0, result.stderr
    assert not census.produced.exists()
    assert not census.state.exists()
    mem.validate_event(json.loads(out.read_text()))


def test_without_memory_the_census_is_unchanged(tmp_path: Path) -> None:
    """The lane-census-refresh workflow runs no --memory and must see no memory exit."""
    body = _CENSUS_SH.read_text(encoding="utf-8")
    assert 'if [[ "$MEMORY" == true && $MEMORY_RC -eq 0 ]]; then' in body
    assert "MEMORY=false" in body
