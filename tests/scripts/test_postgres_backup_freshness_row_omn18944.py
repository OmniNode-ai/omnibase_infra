# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18944 -- the production backup's failure must reach a person.

WHAT IS UNDER TEST
    The `check_postgres_backup_freshness` function inside
    ``deploy/maintenance/omninode-system-slack-report.sh`` -- the artifact that
    actually runs on the `.201` host every fifteen minutes, extracted verbatim
    from that file rather than re-implemented (memory
    ``feedback_test_the_artifact_that_runs``).

WHY IT EXISTS
    `omninode_infra`'s nightly Postgres backup CronJob carries a bounded
    failed-job history limit and nothing else. No alert rule covers CronJob
    failure and the alert-rule files that repo carries are applied by no
    workflow, so a failed nightly backup and a successful one were the same
    event from outside the cluster. Operator ruling 2026-09-20, firm: a gate
    whose failure reaches no person and blocks nothing is not acceptable.

THE ASYMMETRY THESE TESTS PIN
    Three different silences must produce three different rows, because they
    have three different remedies and the whole defect class is them rendering
    identically:

      the backup is missing        -> CRITICAL, the gate ran and found nothing
      the gate stopped running     -> CRITICAL, nobody is checking
      we could not look            -> WARNING, not evidence the backup is gone

    A fourth case matters as much and is the easiest to get wrong: a
    `pull_request` run of the freshness workflow exercises that gate's own unit
    tests and makes NO claim about the live backup. Counting one as a verdict
    would let a green pull request paper over a cluster that had stopped
    backing up, so `test_a_pull_request_run_is_not_a_verdict` is load-bearing.

HERMETICITY
    Every test drives the function through its declared fetch seam against a
    recorded payload. Nothing reaches the GitHub API, `.201`, Slack or AWS. The
    live leg was exercised once by hand during development against a real
    workflow and recorded on the ticket; a seam that only ever resolves to the
    real command would mean these tests all took the "could not look" branch
    while believing they had tested the verdict rules.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCRIPT = REPO_ROOT / "deploy/maintenance/omninode-system-slack-report.sh"
FUNCTION_NAME = "check_postgres_backup_freshness"


def _function_source() -> str:
    """The function as it stands in the deployed script, not a copy of it."""
    text = REPORT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^{FUNCTION_NAME}\(\) \{{.*?^\}}$", text, re.MULTILINE | re.DOTALL
    )
    assert match, f"{FUNCTION_NAME} is no longer defined in {REPORT_SCRIPT}"
    return match.group(0)


def _run(
    tmp_path: Path,
    payload: object | None,
    *,
    # None means "use a placeholder"; the empty string is the meaningful case,
    # exercising the no-token branch. A literal default here would read as a
    # hardcoded credential to the lint that guards real ones.
    token: str | None = None,
    stale_hours: int = 9,
    env: dict[str, str] | None = None,
) -> str:
    """Drive the real function against a recorded payload. Returns its row."""
    body_file = tmp_path / "body.json"
    if payload is None:
        body_file.write_text("", encoding="utf-8")
    elif isinstance(payload, str):
        body_file.write_text(payload, encoding="utf-8")
    else:
        body_file.write_text(json.dumps(payload), encoding="utf-8")

    script = tmp_path / "driver.sh"
    script.write_text(
        "set -u\n" + _function_source() + f"\n{FUNCTION_NAME}\n", encoding="utf-8"
    )

    full_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "GH_PAT": "placeholder-not-a-real-value" if token is None else token,
        "OMNINODE_BACKUP_GATE_FETCH_CMD": f"cat {body_file}",
        "OMNINODE_BACKUP_GATE_STALE_HOURS": str(stale_hours),
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
    return completed.stdout.strip()


def _run_payload(hours_ago: float, **fields: object) -> dict:
    started = datetime.now(UTC) - timedelta(hours=hours_ago)
    run = {
        "event": "schedule",
        "conclusion": "success",
        "status": "completed",
        "run_started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "html_url": "https://github.com/OmniNode-ai/omninode_infra/actions/runs/1",
    }
    run.update(fields)
    return {"total_count": 1, "workflow_runs": [run]}


def _status(row: str) -> str:
    parts = row.split("|")
    assert len(parts) >= 4, f"row is not the four-field domain shape: {row!r}"
    assert parts[0] == "backup", f"row is not in the backup domain: {row!r}"
    assert parts[2] == "postgres-freshness", f"unexpected key: {row!r}"
    return parts[1]


@pytest.mark.unit
class TestTheWiring:
    def test_the_check_is_called_by_the_collector(self) -> None:
        """A function nobody calls is the defect one level down."""
        text = REPORT_SCRIPT.read_text(encoding="utf-8")
        calls = [line for line in text.splitlines() if line.strip() == FUNCTION_NAME]
        assert calls, (
            f"{FUNCTION_NAME} is defined but never invoked, so it emits no row "
            "and reaches nobody"
        )

    def test_the_digest_renders_the_backup_row_even_when_clean(self) -> None:
        """ "Every cluster is fresh" and "nothing looked" must not both be blank.

        That identical rendering is precisely the defect this row closes, so a
        digest section that only appears on failure would reintroduce it.
        """
        text = REPORT_SCRIPT.read_text(encoding="utf-8")
        assert "backup_lines=" in text
        assert "No backup freshness verdict this tick" in text, (
            "the digest no longer distinguishes a clean backup verdict from an "
            "absent one"
        )
        assert "*Production database backup*" in text

    def test_a_backup_issue_row_reaches_the_active_issues_list(self) -> None:
        """The four-field domain shape must be in the issue_lines selector.

        OMN-15525 is the precedent: a selector that disagreed with the row
        shape computed an EMPTY issue list, took the "clean" branch, and paged
        nobody while three lanes were dead.
        """
        text = REPORT_SCRIPT.read_text(encoding="utf-8")
        selector = next(
            line
            for line in text.splitlines()
            if line.strip().startswith("issue_lines=")
        )
        assert '$1=="backup"' in selector, (
            "a backup WARNING/CRITICAL row would be rendered by the generic "
            "column-2 branch with the wrong fields, or not at all"
        )


@pytest.mark.unit
class TestTheThreeSilencesAreDistinguishable:
    def test_a_failed_freshness_run_is_critical_and_names_the_run(
        self, tmp_path: Path
    ) -> None:
        row = _run(tmp_path, _run_payload(1, conclusion="failure"))
        assert _status(row) == "CRITICAL"
        assert "FAILED its freshness check" in row
        assert "actions/runs/1" in row

    def test_a_gate_that_stopped_running_is_critical(self, tmp_path: Path) -> None:
        """Distinct from a failure: nobody is checking, rather than a bad result."""
        row = _run(tmp_path, _run_payload(73))
        assert _status(row) == "CRITICAL"
        assert "may have stopped" in row
        assert "73h ago" in row

    def test_a_deleted_workflow_is_critical_and_says_so(self, tmp_path: Path) -> None:
        row = _run(tmp_path, {"message": "Not Found"})
        assert _status(row) == "CRITICAL"
        assert "has been removed" in row

    def test_an_unreachable_api_is_a_warning_not_a_pass(self, tmp_path: Path) -> None:
        """ "Could not look" must not render as "nothing wrong" -- nor as a page."""
        row = _run(tmp_path, None)
        assert _status(row) == "WARNING"
        assert "state unknown" in row

    def test_a_missing_token_is_a_warning_not_a_pass(self, tmp_path: Path) -> None:
        row = _run(
            tmp_path,
            _run_payload(1),
            token="",
            env={"GITHUB_TOKEN": ""},
        )
        assert _status(row) == "WARNING"
        assert "state unknown" in row

    def test_an_unparseable_body_is_a_warning_not_a_pass(self, tmp_path: Path) -> None:
        row = _run(tmp_path, "{not json at all")
        assert _status(row) == "WARNING"
        assert "state unknown" in row


@pytest.mark.unit
class TestTheVerdictSelection:
    def test_a_fresh_successful_run_is_ok(self, tmp_path: Path) -> None:
        """Positive control for every finding above."""
        row = _run(tmp_path, _run_payload(1))
        assert _status(row) == "OK"
        assert "fresh Postgres backup" in row

    def test_a_pull_request_run_is_not_a_verdict(self, tmp_path: Path) -> None:
        """A green pull request must not paper over a dead backup.

        The freshness workflow runs on pull requests to prove its OWN logic,
        with no AWS leg and no claim about the live backup. If that counted,
        every pull request touching the gate would mint a fake green.
        """
        payload = {
            "total_count": 1,
            "workflow_runs": [
                {
                    "event": "pull_request",
                    "conclusion": "success",
                    "status": "completed",
                    "run_started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "html_url": "https://github.com/x/y/actions/runs/9",
                }
            ],
        }
        row = _run(tmp_path, payload)
        assert _status(row) == "CRITICAL"
        assert "NO scheduled run" in row

    def test_the_newest_scheduled_run_decides_not_the_first_listed(
        self, tmp_path: Path
    ) -> None:
        now = datetime.now(UTC)

        def at(hours: float, conclusion: str) -> dict:
            return {
                "event": "schedule",
                "conclusion": conclusion,
                "status": "completed",
                "run_started_at": (now - timedelta(hours=hours)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "html_url": "https://github.com/x/y/actions/runs/1",
            }

        # An older success listed after a newer failure must not win.
        payload = {
            "total_count": 3,
            "workflow_runs": [at(1, "failure"), at(20, "success"), at(9, "success")],
        }
        row = _run(tmp_path, payload)
        assert _status(row) == "CRITICAL"
        assert "FAILED its freshness check" in row

    def test_an_in_flight_run_inside_the_window_is_not_an_alarm(
        self, tmp_path: Path
    ) -> None:
        row = _run(
            tmp_path,
            _run_payload(1, status="in_progress", conclusion=None),
        )
        assert _status(row) == "OK"

    def test_an_in_flight_run_wedged_past_the_window_is_critical(
        self, tmp_path: Path
    ) -> None:
        """A job stuck for longer than the bar is the same silence as none."""
        row = _run(
            tmp_path,
            _run_payload(40, status="in_progress", conclusion=None),
        )
        assert _status(row) == "CRITICAL"
        assert "unverified" in row

    def test_an_empty_run_list_is_critical_not_ok(self, tmp_path: Path) -> None:
        row = _run(tmp_path, {"total_count": 0, "workflow_runs": []})
        assert _status(row) == "CRITICAL"
        assert "NO scheduled run" in row


@pytest.mark.unit
class TestNoBypass:
    def test_the_check_cannot_be_silenced_into_a_green(self, tmp_path: Path) -> None:
        """Disabling the check removes the row; it must never fake an OK one.

        A disable that emitted OK would be strictly worse than no check at all,
        because the digest would then assert a freshness nobody measured.
        """
        row = _run(
            tmp_path,
            _run_payload(1, conclusion="failure"),
            env={"OMNINODE_BACKUP_GATE_CHECK_ENABLED": "0"},
        )
        assert row == "", "a disabled check emitted a row"
