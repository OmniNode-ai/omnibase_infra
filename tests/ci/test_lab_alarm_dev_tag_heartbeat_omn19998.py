# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Controls for the lab alarm's dev-tag heartbeat condition (OMN-19998).

The deploy agent beats against each tag-followed repository's newest dev tag
and serves the latest beat on ``GET /dev-tag-heartbeat``. Nothing read it: a
lane that stopped beating, or one that beat while running behind the newest
tag, raised nothing. The stale fixture is the negative control -- the same
document as the fresh one with one beat gone quiet.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from scripts.lab_alarm import (
    EnumAlarmCondition,
    EnumConditionOutcome,
    ModelAlarmConfig,
    evaluate_dev_tag_heartbeat,
    read_dev_tag_heartbeat,
)
from tests.helpers.lane_census_fixtures import use_neutral_lane_census

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "lab_alarm.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FRESH = FIXTURES / "dev_tag_heartbeat_fresh_omn19998.json"
STALE = FIXTURES / "dev_tag_heartbeat_stale_omn19998.json"

NOW = datetime(2026, 10, 17, 2, 0, 0, tzinfo=UTC)
MAX_AGE = timedelta(minutes=30)
REPOSITORIES = ("omnimarket",)
TWO_REPOSITORIES = ("omnimarket", "omnibase_infra")


def _document(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _both(document: dict[str, Any]) -> dict[str, Any]:
    """The document with the followed repository's beat copied under a second repository."""
    beats = document["heartbeat"]
    beats["omnibase_infra"] = {**beats["omnimarket"], "repository": "omnibase_infra"}
    return document


def _evaluate(
    document: dict[str, Any] | Exception,
    repositories: tuple[str, ...] = REPOSITORIES,
) -> Any:
    def reader() -> dict[str, Any]:
        if isinstance(document, Exception):
            raise document
        return document

    return evaluate_dev_tag_heartbeat(
        reader, repositories=repositories, now=NOW, max_age=MAX_AGE
    )


@pytest.mark.unit
def test_a_fresh_heartbeat_for_the_followed_repository_is_ok() -> None:
    """Positive control: the stale fixture differs from this one in observed_at alone."""
    report = _evaluate(_document(FRESH))
    assert report.condition is EnumAlarmCondition.DEV_TAG_HEARTBEAT
    assert report.outcome is EnumConditionOutcome.OK
    assert report.alarms == ()
    assert "omnimarket" in report.evidence


@pytest.mark.unit
def test_a_stale_heartbeat_alarms_and_names_the_repository_and_age() -> None:
    report = _evaluate(_document(STALE))
    assert report.outcome is EnumConditionOutcome.ALARM
    assert len(report.alarms) == 1
    alarm = report.alarms[0]
    assert alarm.condition is EnumAlarmCondition.DEV_TAG_HEARTBEAT
    assert alarm.subject == "omnimarket"
    assert "stale" in alarm.detail
    # 02:00 minus 22:10 the evening before is 3h50m.
    assert "3:50:00" in alarm.detail


@pytest.mark.unit
def test_a_heartbeat_exactly_at_the_bound_is_not_stale() -> None:
    document = _document(FRESH)
    beat = document["heartbeat"]["omnimarket"]
    beat["observed_at"] = (NOW - MAX_AGE).isoformat().replace("+00:00", "Z")
    assert _evaluate(document).outcome is EnumConditionOutcome.OK
    beat["observed_at"] = (
        (NOW - MAX_AGE - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    )
    assert _evaluate(document).outcome is EnumConditionOutcome.ALARM


@pytest.mark.unit
def test_a_followed_repository_with_no_beat_alarms() -> None:
    report = _evaluate(_document(FRESH), TWO_REPOSITORIES)
    assert report.outcome is EnumConditionOutcome.ALARM
    assert [a.subject for a in report.alarms] == ["omnibase_infra"]
    assert "no beat" in report.alarms[0].detail


@pytest.mark.unit
def test_a_repository_the_config_does_not_follow_is_not_graded() -> None:
    document = _both(_document(FRESH))
    document["heartbeat"]["omnibase_infra"]["observed_at"] = "2026-10-01T00:00:00Z"
    assert _evaluate(document).outcome is EnumConditionOutcome.OK


@pytest.mark.unit
def test_a_beat_that_reports_the_lane_behind_its_newest_tag_alarms() -> None:
    document = _document(FRESH)
    beat = document["heartbeat"]["omnimarket"]
    beat["alarm"] = True
    beat["lag_seconds"] = 9000
    beat["running_ref"] = "1" * 40
    report = _evaluate(document)
    assert report.outcome is EnumConditionOutcome.ALARM
    assert [a.subject for a in report.alarms] == ["omnimarket"]
    assert "behind" in report.alarms[0].detail
    assert "dev-20261017014500-0123456789ab" in report.alarms[0].detail


@pytest.mark.unit
def test_one_repository_stale_and_one_behind_raises_two_alarms() -> None:
    document = _both(_document(STALE))
    document["heartbeat"]["omnibase_infra"]["observed_at"] = "2026-10-17T01:58:00Z"
    document["heartbeat"]["omnibase_infra"]["alarm"] = True
    report = _evaluate(document, TWO_REPOSITORIES)
    assert sorted(a.subject for a in report.alarms) == ["omnibase_infra", "omnimarket"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "document",
    [
        OSError("connection refused"),
        ValueError("not json"),
        {"heartbeat": None, "reason": "this agent has no heartbeat to report"},
        {"heartbeat": ["not", "an", "object"]},
        {"no": "heartbeat key"},
    ],
    ids=["unreachable", "bad-json", "agent-reports-none", "wrong-shape", "no-key"],
)
def test_an_unreadable_surface_is_indeterminate_not_an_alarm(
    document: dict[str, Any] | Exception,
) -> None:
    report = _evaluate(document)
    assert report.outcome is EnumConditionOutcome.INDETERMINATE
    assert report.alarms == ()


@pytest.mark.unit
def test_an_unparseable_observed_at_is_indeterminate() -> None:
    document = _document(FRESH)
    document["heartbeat"]["omnimarket"]["observed_at"] = "yesterday"
    assert _evaluate(document).outcome is EnumConditionOutcome.INDETERMINATE


@contextmanager
def _agent(status: int, body: bytes) -> Iterator[str]:
    """A real local HTTP server answering GET /dev-tag-heartbeat; yields its URL."""
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append(self.path)
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
        assert seen == ["/dev-tag-heartbeat"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.unit
def test_the_reader_fetches_the_agent_endpoint() -> None:
    with _agent(200, FRESH.read_bytes()) as url:
        document = read_dev_tag_heartbeat(url)
    assert set(document["heartbeat"]) == set(REPOSITORIES)


@pytest.mark.unit
def test_the_reader_raises_on_a_refused_status_so_the_condition_reads_indeterminate() -> (
    None
):
    with _agent(503, b"{}") as url:
        with pytest.raises(OSError, match="HTTP 503"):
            read_dev_tag_heartbeat(url)


@pytest.mark.unit
def test_the_reader_refuses_a_non_http_agent_url() -> None:
    with pytest.raises(ValueError, match="not an http URL"):
        read_dev_tag_heartbeat("file:///etc/passwd")


@pytest.mark.unit
def test_the_shipped_config_declares_the_followed_repositories_and_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_neutral_lane_census(monkeypatch)
    monkeypatch.setenv("ONEX_INFRA_HOST", "lab.invalid")
    monkeypatch.setenv("ONEX_RUNTIME_SSH_HOST", "user@lab.invalid")
    monkeypatch.setenv("OMNI_HOME", "/nonexistent")
    config = ModelAlarmConfig.load(CONFIG)
    assert config.dev_tag_heartbeat_repositories == REPOSITORIES
    # The agent re-reads every 60 s and publishes every 15 min; 30 min is two
    # publish periods, well inside the 2 h bound after which a beat alarms.
    assert config.dev_tag_heartbeat_max_age == MAX_AGE
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("dev_tag_heartbeat_repositories", "dev_tag_heartbeat_max_age_minutes"):
        assert any(k.startswith("_comment_") and key in v for k, v in payload.items())
