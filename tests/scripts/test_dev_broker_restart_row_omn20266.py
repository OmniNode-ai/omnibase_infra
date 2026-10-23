# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20266: dev broker crashes between ticks must reach the system alert.

Extract the deployed function verbatim and drive its declared inspect and clock
seams. No test contacts Docker, the dev host, or Slack. The measured incident
and the following tick must both stay CRITICAL so confirmation can page.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCRIPT = REPO_ROOT / "deploy/maintenance/omninode-system-slack-report.sh"
FUNCTION_NAME = "check_dev_broker"
STARTED_AT = "2026-10-23T08:06:32.496581573Z"


def _function_source() -> str:
    """The function as it stands in the deployed script, not a copy of it."""
    text = REPORT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^{FUNCTION_NAME}\(\) \{{.*?^\}}$", text, re.MULTILINE | re.DOTALL
    )
    assert match, f"{FUNCTION_NAME} is no longer defined in {REPORT_SCRIPT}"
    return match.group(0)


def _epoch(timestamp: str) -> str:
    return str(int(datetime.fromisoformat(timestamp).replace(tzinfo=UTC).timestamp()))


def _payload(
    restarts: int = 12,
    started_at: str = STARTED_AT,
    memory_arg: str = "--memory 6G",
) -> str:
    cmd = ["redpanda", "start", "--mode dev-container", "--smp 1"]
    if memory_arg:
        cmd.append(memory_arg)
    cmd.append("--default-log-level=warn")
    return f"{restarts}|{started_at}|running|{json.dumps(cmd)}\n"


def _run(
    tmp_path: Path,
    payload: str,
    *,
    now: str = "2026-10-23T08:20:00Z",
    env: dict[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Run the real function with nounset, errexit and pipefail enabled."""
    body_file = tmp_path / "inspect.txt"
    body_file.write_text(payload, encoding="utf-8")
    script = tmp_path / "driver.sh"
    script.write_text(
        "set -euo pipefail\n" + _function_source() + f"\n{FUNCTION_NAME}\n",
        encoding="utf-8",
    )
    full_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "OMNINODE_DEV_BROKER_INSPECT_CMD": f"cat {body_file}",
        "OMNINODE_ALERT_DEV_BROKER_NOW_EPOCH": _epoch(now),
    }
    full_env.update(env or {})
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=full_env,
    )
    assert completed.returncode == 0, completed.stderr
    rows: dict[str, tuple[str, str]] = {}
    for row in completed.stdout.splitlines():
        parts = row.split("|")
        assert len(parts) == 4, f"not a four-field resource row: {row!r}"
        domain, status, key, detail = parts
        assert domain == "broker"
        assert key not in rows, f"duplicate row: {row!r}"
        rows[key] = (status, detail)
    assert rows, "the broker check emitted no verdict"
    return rows


@pytest.mark.unit
def test_check_is_called_on_its_own_line_inside_collect() -> None:
    text = REPORT_SCRIPT.read_text(encoding="utf-8")
    collect = re.search(r"^collect\(\) \{.*?^\}$", text, re.MULTILINE | re.DOTALL)
    assert collect
    assert re.search(
        r"^    check_census_drift\n    check_dev_broker$", collect[0], re.M
    )


@pytest.mark.unit
def test_defaults_match_the_dev_broker_and_confirmation_window() -> None:
    source = _function_source()
    assert "${OMNINODE_ALERT_DEV_BROKER_CONTAINER:-omnibase-infra-redpanda}" in source
    assert "${OMNINODE_ALERT_DEV_BROKER_RESTART_HOLD_SECONDS:-3600}" in source
    assert "${OMNINODE_ALERT_DEV_BROKER_MEMORY_FLOOR_MB:-12288}" in source


@pytest.mark.unit
@pytest.mark.parametrize("now", ["2026-10-23T08:20:00Z", "2026-10-23T08:35:00Z"])
def test_measured_incident_stays_critical_across_confirmation_ticks(
    tmp_path: Path, now: str
) -> None:
    rows = _run(tmp_path, _payload(), now=now)
    assert set(rows) == {"dev-broker-restart", "dev-broker-memory"}
    status, detail = rows["dev-broker-restart"]
    assert status == "CRITICAL"
    for expected in ("omnibase-infra-redpanda", "RestartCount=12", STARTED_AT, "3600s"):
        assert expected in detail
    status, detail = rows["dev-broker-memory"]
    assert status == "WARNING"
    for expected in (
        "6G",
        "6144 MB",
        "12288 MB",
        "docker/docker-compose.infra.yml",
        "--memory ${DEV_REDPANDA_MEMORY:-12G}",
    ):
        assert expected in detail


@pytest.mark.unit
@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-10-23T09:06:31Z", "CRITICAL"),
        ("2026-10-23T09:06:32Z", "OK"),
        ("2026-10-23T09:06:33Z", "OK"),
        ("2026-10-23T10:06:32Z", "OK"),
    ],
)
def test_restart_hold_expires_at_the_boundary(
    tmp_path: Path, now: str, expected: str
) -> None:
    status, detail = _run(tmp_path, _payload(), now=now)["dev-broker-restart"]
    assert status == expected
    if expected == "OK":
        assert detail == f"restart_count=12 started_at={STARTED_AT}"
    else:
        assert "restart within hold window 3600s" in detail


@pytest.mark.unit
def test_fresh_recreate_does_not_alarm(tmp_path: Path) -> None:
    rows = _run(tmp_path, _payload(restarts=0, started_at="2026-10-23T08:15:00Z"))
    assert rows["dev-broker-restart"] == (
        "OK",
        "restart_count=0 started_at=2026-10-23T08:15:00Z",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("memory_arg", "expected"),
    [
        ("--memory 12G", "OK"),
        ("--memory=16384M", "OK"),
        ("--memory 12g", "OK"),
        ("--memory=12288m", "OK"),
        ("--memory 12287M", "WARNING"),
        ("", "WARNING"),
        ("--memory nonsense", "WARNING"),
        ("--memory=6T", "WARNING"),
    ],
)
def test_running_memory_is_compared_to_floor(
    tmp_path: Path, memory_arg: str, expected: str
) -> None:
    rows = _run(tmp_path, _payload(memory_arg=memory_arg))
    status, detail = rows["dev-broker-memory"]
    assert status == expected
    if memory_arg in ("", "--memory nonsense", "--memory=6T"):
        assert "could not be read" in detail


@pytest.mark.unit
@pytest.mark.parametrize("payload", ["", " \n"])
def test_empty_inspect_is_one_critical_row(tmp_path: Path, payload: str) -> None:
    rows = _run(tmp_path, payload)
    assert list(rows) == ["dev-broker-restart"]
    status, detail = rows["dev-broker-restart"]
    assert status == "CRITICAL"
    assert "could not inspect omnibase-infra-redpanda;" in detail


@pytest.mark.unit
def test_failed_inspect_discards_even_partial_output(tmp_path: Path) -> None:
    failing = tmp_path / "failed-inspect.sh"
    failing.write_text(f"printf '%s' {shlex.quote(_payload())}\nexit 1\n")
    rows = _run(
        tmp_path,
        _payload(),
        env={"OMNINODE_DEV_BROKER_INSPECT_CMD": f"bash {failing}"},
    )
    assert list(rows) == ["dev-broker-restart"]
    assert rows["dev-broker-restart"][0] == "CRITICAL"
    assert "could not inspect" in rows["dev-broker-restart"][1]


@pytest.mark.unit
@pytest.mark.parametrize("started_at", ["not-a-date", ""])
def test_unparseable_started_at_is_critical(tmp_path: Path, started_at: str) -> None:
    rows = _run(tmp_path, _payload(started_at=started_at))
    assert rows["dev-broker-restart"][0] == "CRITICAL"
    assert "StartedAt=" in rows["dev-broker-restart"][1]
    assert "could not be parsed" in rows["dev-broker-restart"][1]
    assert rows["dev-broker-memory"][0] == "WARNING"


@pytest.mark.unit
@pytest.mark.parametrize("restart_count", ["not-a-number", "", "-1"])
def test_unparseable_restart_count_is_critical(
    tmp_path: Path, restart_count: str
) -> None:
    payload = _payload().replace("12|", f"{restart_count}|", 1)
    rows = _run(tmp_path, payload)
    assert rows["dev-broker-restart"][0] == "CRITICAL"
    assert (
        f"RestartCount={restart_count} could not be parsed"
        in rows["dev-broker-restart"][1]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "state", ["created", "restarting", "paused", "exited", "dead", "removing", ""]
)
@pytest.mark.parametrize("restarts", [0, 12])
def test_non_running_broker_is_critical_even_after_hold_expires(
    tmp_path: Path, state: str, restarts: int
) -> None:
    payload = _payload(restarts=restarts).replace("|running|", f"|{state}|")
    rows = _run(tmp_path, payload, now="2026-10-23T10:06:32Z")
    assert set(rows) == {"dev-broker-restart"}
    status, detail = rows["dev-broker-restart"]
    assert status == "CRITICAL"
    assert f"state={state or 'unknown'}; dev broker is not running" in detail


@pytest.mark.unit
def test_old_fractional_start_does_not_extend_restart_hold(tmp_path: Path) -> None:
    started_at = "2026-10-23T07:06:32.999999999Z"
    rows = _run(tmp_path, _payload(started_at=started_at))
    assert rows["dev-broker-restart"] == (
        "OK",
        f"restart_count=12 started_at={started_at}",
    )


@pytest.mark.unit
def test_thresholds_and_container_are_env_overridable(tmp_path: Path) -> None:
    rows = _run(
        tmp_path,
        _payload(),
        env={
            "OMNINODE_ALERT_DEV_BROKER_CONTAINER": "custom-broker",
            "OMNINODE_ALERT_DEV_BROKER_RESTART_HOLD_SECONDS": "600",
            "OMNINODE_ALERT_DEV_BROKER_MEMORY_FLOOR_MB": "6144",
        },
    )
    assert rows["dev-broker-restart"][0] == "OK"
    assert rows["dev-broker-memory"][0] == "OK"
    assert "custom-broker" in rows["dev-broker-memory"][1]


@pytest.mark.unit
def test_broker_rows_have_a_digest_section_and_reach_active_issues() -> None:
    text = REPORT_SCRIPT.read_text(encoding="utf-8")
    digest = re.search(r"^format_digest\(\) \{.*?^\}$", text, re.MULTILINE | re.DOTALL)
    assert digest
    assert "*Dev broker*\n$broker_lines\n" in digest[0]
    assert '$1=="broker" {printf' in digest[0]
    assert '$1=="census" || $1=="broker") && ($2=="WARNING"' in digest[0]


@pytest.mark.unit
def test_the_check_can_be_switched_off(tmp_path: Path) -> None:
    script = tmp_path / "driver.sh"
    script.write_text(
        "set -euo pipefail\n" + _function_source() + f"\n{FUNCTION_NAME}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={
            "PATH": "/usr/bin:/bin",
            "OMNINODE_ALERT_DEV_BROKER_CHECK_ENABLED": "0",
            "OMNINODE_DEV_BROKER_INSPECT_CMD": "false",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
