# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""An unchanged finding inside the dedup window does not re-alert (OMN-18942).

MEASURED, 2026-09-23. SLACK_BOT_TOKEN and SLACK_CHANNEL_ID were set on this
repository at 10:43Z. After that, every scheduled run of
`pr-ci-zombie-detector.yml` re-posted the same unchanged findings to the
channel (runs 35850665037, 35851297482, 35852673260), because the evaluator
called `post_slack_alert` whenever there was a finding, with no prior-alert
state and no window. A channel that repeats itself every ten minutes is muted
on day one, which is the third defect the ticket names.

WHAT IS PINNED
    * unchanged-or-lower inside the window: suppressed, and the suppression is
      RECORDED in the report artifact and the job summary (a silent drop is
      indistinguishable from a run that never happened -- AC-3's falsifier);
    * worsened, new, and window-expired: posted;
    * resolved: not posted, recorded;
    * prior state unreadable: POST, with a marker saying why (fail open to the
      channel, never to silence);
    * a failed delivery does not advance the state, so it is retried;
    * a missing destination still refuses exactly as before.

Every suppression test has a positive control over the same inputs with only
the prior state changed (rule 16): an alerter that never posts would pass the
suppression tests alone.

The end-to-end tests drive the real `main()` with `_gh_json`, `_gh_bytes` and
`urlopen` replaced at the module boundary, so the fetch, the decision, the
report and the summary are all the production code path.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "nonrequired_check_failure_rate.py"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-ci-zombie-detector.yml"

OWNER = "OmniNode-ai"
REPO = "omnibase_infra"
SLUG = f"{OWNER}/{REPO}"
CHECK = "Test-Failure Ratchet Gate"
KEY = f"check|{SLUG}|{CHECK}"
WORKFLOW_FILE = "pr-ci-zombie-detector.yml"
CURRENT_RUN_ID = 999
PRIOR_RUN_ID = 111


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "nonrequired_check_failure_rate_dedup", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _prior_report(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "nonrequired_check_report/v1",
        "repos": {},
        "alert_state": {
            "schema": "nonrequired_check_alert_state/v1",
            "entries": entries,
        },
    }


def _zip(payload: Any, *, raw: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "nonrequired-check-report.json",
            raw if raw is not None else json.dumps(payload),
        )
    return buffer.getvalue()


class _Harness:
    """The GitHub and Slack boundaries, faked at the module's own seams."""

    def __init__(
        self,
        module: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        failing: dict[str, int],
        prior_zip: bytes | None,
        runs_error: str | None = None,
        slack_ok: bool = True,
        with_destination: bool = True,
    ) -> None:
        self.module = module
        self.tmp_path = tmp_path
        self.posts: list[dict[str, Any]] = []
        self.runs_queries: list[str] = []
        self.summary = tmp_path / "summary.md"
        self.report = tmp_path / "report.json"
        heads = [f"sha{i}" for i in range(max([*failing.values(), 1]))]

        def gh_json(path: str, token: str | None) -> Any:
            if path.startswith(f"repos/{SLUG}/branches/"):
                return {"contexts": ["CI Summary"]}
            if path.startswith(f"repos/{SLUG}/pulls"):
                return [{"head": {"sha": sha}} for sha in heads]
            if "/check-runs" in path:
                index = heads.index(path.split("/commits/")[1].split("/")[0])
                return {
                    "check_runs": [
                        {
                            "name": name,
                            "conclusion": "failure" if index < count else "success",
                        }
                        for name, count in failing.items()
                    ]
                }
            if path.startswith(f"repos/{SLUG}/actions/workflows/{WORKFLOW_FILE}/runs"):
                self.runs_queries.append(path)
                if runs_error:
                    raise RuntimeError(runs_error)
                return {"workflow_runs": [{"id": CURRENT_RUN_ID}, {"id": PRIOR_RUN_ID}]}
            if path.startswith(f"repos/{SLUG}/actions/runs/{PRIOR_RUN_ID}/artifacts"):
                if prior_zip is None:
                    return {"artifacts": []}
                return {
                    "artifacts": [
                        {
                            "id": 5,
                            "name": "nonrequired-check-report",
                            "expired": False,
                        }
                    ]
                }
            if path.startswith(f"repos/{SLUG}/actions/runs/{CURRENT_RUN_ID}/"):
                raise AssertionError("the current run's own artifact was read")
            raise AssertionError(f"unexpected gh api path {path}")

        def gh_bytes(path: str, token: str | None) -> bytes:
            assert path == f"repos/{SLUG}/actions/artifacts/5/zip", path
            assert prior_zip is not None
            return prior_zip

        class _Response:
            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *_: Any) -> bool:
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "ok": slack_ok,
                        **({} if slack_ok else {"error": "not_in_channel"}),
                    }
                ).encode()

        def urlopen(request: Any, timeout: int = 10) -> _Response:
            self.posts.append(json.loads(request.data.decode()))
            return _Response()

        monkeypatch.setattr(module, "_gh_json", gh_json)
        monkeypatch.setattr(module, "_gh_bytes", gh_bytes)
        monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(self.summary))
        monkeypatch.setenv("GITHUB_REPOSITORY", SLUG)
        monkeypatch.setenv("GITHUB_RUN_ID", str(CURRENT_RUN_ID))
        monkeypatch.setenv("GITHUB_REF_NAME", "dev")
        if with_destination:
            monkeypatch.setenv("SLACK_BOT_TOKEN", "test-bot-token-not-real")
            monkeypatch.setenv("SLACK_CHANNEL_ID", "C0TEST")
        else:
            monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
            monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)

    def run(self, now: datetime) -> int:
        self.module._now = lambda: now
        return int(
            self.module.main(
                [
                    "--owner",
                    OWNER,
                    "--repo",
                    REPO,
                    "--no-scheduled",
                    "--policy",
                    str(POLICY),
                    "--report",
                    str(self.report),
                    "--prior-state-workflow",
                    WORKFLOW_FILE,
                ]
            )
        )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.report.read_text(encoding="utf-8"))

    @property
    def summary_text(self) -> str:
        return self.summary.read_text(encoding="utf-8")


NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC)


def _entry(failures: int, posted_at: datetime) -> dict[str, Any]:
    return {"posted_failures": failures, "last_posted_at": _iso(posted_at)}


# ---------------------------------------------------------------------------
# The falsifier: an unchanged finding inside the window does not re-alert.
# ---------------------------------------------------------------------------


def test_unchanged_finding_inside_the_window_is_suppressed_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    prior = _zip(_prior_report({KEY: _entry(3, NOW - timedelta(hours=1))}))
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    assert harness.run(NOW) == 0
    assert harness.posts == [], "an unchanged finding inside the window re-posted"

    dedup = harness.payload["dedup"]
    assert dedup["prior_state"]["status"] == "read"
    assert dedup["prior_state"]["source_run_id"] == PRIOR_RUN_ID
    assert [s["key"] for s in dedup["suppressed"]] == [KEY]
    assert dedup["suppressed"][0]["failures"] == 3
    assert dedup["suppressed"][0]["posted_failures"] == 3
    assert dedup["due"] == []
    # The window is measured from the LAST POST, so a standing finding is
    # re-surfaced once per window rather than never.
    carried = harness.payload["alert_state"]["entries"][KEY]
    assert carried["last_posted_at"] == _iso(NOW - timedelta(hours=1))
    assert "SUPPRESSED" in harness.summary_text
    assert CHECK in harness.summary_text


def test_positive_control_same_finding_with_no_prior_post_is_posted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rule 16: the suppression above is not an alerter that never posts."""
    module = _module()
    prior = _zip(_prior_report({}))
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1
    assert CHECK in harness.posts[0]["text"]
    dedup = harness.payload["dedup"]
    assert [p["reason"] for p in dedup["due"]] == ["new"]
    assert harness.payload["alert_state"]["entries"][KEY] == {
        "posted_failures": 3,
        "last_posted_at": _iso(NOW),
    }


def test_a_lower_count_inside_the_window_is_suppressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    prior = _zip(_prior_report({KEY: _entry(5, NOW - timedelta(hours=2))}))
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    assert harness.run(NOW) == 0
    assert harness.posts == []
    assert [s["key"] for s in harness.payload["dedup"]["suppressed"]] == [KEY]


def test_a_worsened_count_re_posts_inside_the_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    prior = _zip(_prior_report({KEY: _entry(3, NOW - timedelta(hours=1))}))
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 4}, prior_zip=prior
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1
    posted = harness.payload["dedup"]["due"]
    assert posted[0]["reason"] == "worsened"
    assert posted[0]["previous_failures"] == 3
    assert "worsened" in harness.posts[0]["text"]
    assert harness.payload["alert_state"]["entries"][KEY]["posted_failures"] == 4


def test_a_new_finding_posts_and_a_resolved_one_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    gone = f"check|{SLUG}|A Check That Recovered"
    prior = _zip(
        _prior_report(
            {
                KEY: _entry(3, NOW - timedelta(hours=1)),
                gone: _entry(4, NOW - timedelta(hours=1)),
            }
        )
    )
    other = "Brand New Flake"
    harness = _Harness(
        module,
        monkeypatch,
        tmp_path,
        failing={CHECK: 3, other: 3},
        prior_zip=prior,
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1
    text = harness.posts[0]["text"]
    assert other in text
    assert "Recovered" not in text, "a resolved finding was posted"
    dedup = harness.payload["dedup"]
    assert [p["key"] for p in dedup["due"]] == [f"check|{SLUG}|{other}"]
    assert [s["key"] for s in dedup["suppressed"]] == [KEY]
    assert dedup["resolved"] == [gone]
    # A resolved key is dropped, so a recurrence reads as new, not suppressed.
    assert gone not in harness.payload["alert_state"]["entries"]
    # The post names what it left out, so the channel is not read as the
    # whole picture.
    assert "1 unchanged finding(s) suppressed" in text


def test_window_expiry_re_posts_an_unchanged_finding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    window = float(module.load_policy(POLICY)["alert_dedup_window_hours"])
    prior = _zip(
        _prior_report({KEY: _entry(3, NOW - timedelta(hours=window, seconds=1))})
    )
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1
    assert harness.payload["dedup"]["due"][0]["reason"] == "window_expired"
    assert harness.payload["alert_state"]["entries"][KEY]["last_posted_at"] == _iso(NOW)


def test_just_inside_the_window_is_still_suppressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Boundary control for the expiry test above."""
    module = _module()
    window = float(module.load_policy(POLICY)["alert_dedup_window_hours"])
    prior = _zip(
        _prior_report({KEY: _entry(3, NOW - timedelta(hours=window, seconds=-60))})
    )
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    assert harness.run(NOW) == 0
    assert harness.posts == []


# ---------------------------------------------------------------------------
# Prior state that cannot be read fails OPEN to the channel, with a marker.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prior_zip", "runs_error", "status"),
    [
        (_zip(None, raw=b"{not json"), None, "unreadable"),
        (b"this is not a zip archive", None, "unreadable"),
        (
            _zip({"alert_state": {"schema": "something/else", "entries": {}}}),
            None,
            "unreadable",
        ),
        (None, "gh api runs failed: HTTP 403", "unreadable"),
    ],
    ids=["corrupt-json", "corrupt-zip", "wrong-schema", "runs-listing-refused"],
)
def test_unreadable_prior_state_posts_with_a_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prior_zip: bytes | None,
    runs_error: str | None,
    status: str,
) -> None:
    module = _module()
    harness = _Harness(
        module,
        monkeypatch,
        tmp_path,
        failing={CHECK: 3},
        prior_zip=prior_zip,
        runs_error=runs_error,
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1, "unreadable prior state silently suppressed"
    assert "prior alert state unreadable" in harness.posts[0]["text"]
    dedup = harness.payload["dedup"]
    assert dedup["prior_state"]["status"] == status
    assert dedup["prior_state"]["reason"]
    assert [p["reason"] for p in dedup["due"]] == ["prior_state_unreadable"]
    assert "prior alert state UNREADABLE" in harness.summary_text


def test_absent_prior_state_posts_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No earlier artifact (first run, or every recent run lacks one) is absent,
    not unreadable -- and it posts too."""
    module = _module()
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=None
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1
    assert harness.payload["dedup"]["prior_state"]["status"] == "absent"
    assert harness.payload["dedup"]["due"][0]["reason"] == "prior_state_absent"


def test_a_prior_report_predating_alert_state_reads_as_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every artifact uploaded before this change has no alert_state key."""
    module = _module()
    prior = _zip({"schema": "nonrequired_check_report/v1", "repos": {}})
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1
    assert harness.payload["dedup"]["prior_state"]["status"] == "absent"


# ---------------------------------------------------------------------------
# Delivery and destination.
# ---------------------------------------------------------------------------


def test_a_failed_delivery_does_not_advance_the_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Slack answers HTTP 200 with ok:false on a refused post. Recording that
    as delivered would suppress an alert nobody ever received."""
    module = _module()
    prior = _zip(_prior_report({}))
    harness = _Harness(
        module,
        monkeypatch,
        tmp_path,
        failing={CHECK: 3},
        prior_zip=prior,
        slack_ok=False,
    )
    assert harness.run(NOW) == 0
    assert len(harness.posts) == 1
    assert harness.payload["dedup"]["delivered"] is False
    assert KEY not in harness.payload["alert_state"]["entries"]


def test_a_missing_destination_still_refuses_even_when_everything_is_suppressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fail-closed behaviour is unchanged by the dedup: findings plus no
    destination is red whatever the prior state says."""
    module = _module()
    prior = _zip(_prior_report({KEY: _entry(3, NOW - timedelta(hours=1))}))
    harness = _Harness(
        module,
        monkeypatch,
        tmp_path,
        failing={CHECK: 3},
        prior_zip=prior,
        with_destination=False,
    )
    assert harness.run(NOW) == 1
    assert harness.posts == []
    assert harness.report.is_file()


def test_a_suppressed_finding_raises_no_warning_annotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    prior = _zip(_prior_report({KEY: _entry(3, NOW - timedelta(hours=1))}))
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    harness.run(NOW)
    out = capsys.readouterr().out
    assert "::warning title=Non-required check failing::" not in out
    assert f"SUPPRESSED {KEY}" in out


def test_a_posted_finding_still_raises_its_warning_annotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Control for the test above."""
    module = _module()
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=None
    )
    harness.run(NOW)
    assert "::warning title=Non-required check failing::" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Declared, not defaulted; and wired.
# ---------------------------------------------------------------------------


def test_the_policy_declares_the_dedup_window() -> None:
    module = _module()
    assert module.load_policy(POLICY)["alert_dedup_window_hours"] == 24


def test_the_policy_fails_loudly_without_the_dedup_window(tmp_path: Path) -> None:
    module = _module()
    doc = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    del doc["route"]["nonrequired_check_alert"]["alert_dedup_window_hours"]
    partial = tmp_path / "policy.yaml"
    partial.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(KeyError, match="alert_dedup_window_hours"):
        module.load_policy(partial)


def test_the_workflow_reads_prior_state_from_its_own_run_history() -> None:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = doc["jobs"]["nonrequired-check-failure-rate"]["steps"]
    evaluate = next(
        s for s in steps if s.get("name") == "Evaluate non-required check failure rates"
    )
    assert f"--prior-state-workflow {WORKFLOW_FILE}" in evaluate["run"]
    # The fetch rides the minted App token, which already carries actions:read
    # on this repository; no new credential.
    assert evaluate["env"]["GH_TOKEN"] == "${{ steps.app-token.outputs.token }}"
    mint = next(s for s in steps if s.get("id") == "app-token")
    assert mint["with"]["permission-actions"] == "read"
    assert "omnibase_infra" in mint["with"]["repositories"]


# ---------------------------------------------------------------------------
# State that must not be lost, and state that must not be trusted.
# ---------------------------------------------------------------------------


def _alert(module: Any, check: str, failures: int, repo: str = SLUG) -> Any:
    return module.Alert(repo=repo, check=check, failures=failures, observed=10)


def test_an_unreadable_repository_carries_its_state_instead_of_resolving() -> None:
    """A transient 403 on one repository must not drop its findings' state,
    or they would all re-post as new the next time it reads."""
    module = _module()
    other = "OmniNode-ai/omniweb"
    other_key = f"check|{other}|Web Flake"
    prior = module.PriorAlertState(
        "read",
        "",
        {
            KEY: _entry(3, NOW - timedelta(hours=1)),
            other_key: _entry(4, NOW - timedelta(hours=1)),
        },
        PRIOR_RUN_ID,
    )
    window = timedelta(hours=24)
    decision = module.decide_alerts(
        [_alert(module, CHECK, 3)], prior, NOW, window, frozenset({other})
    )
    assert decision.resolved == []
    assert decision.carried == [other_key]
    state = module.next_alert_state(decision, prior, NOW, delivered=False)
    assert state[other_key] == prior.entries[other_key]

    # Control: the same key with its repository READABLE is resolved and dropped.
    control = module.decide_alerts([_alert(module, CHECK, 3)], prior, NOW, window)
    assert control.resolved == [other_key]
    assert other_key not in module.next_alert_state(control, prior, NOW, True)


def test_a_post_recorded_in_the_future_does_not_suppress() -> None:
    module = _module()
    prior = module.PriorAlertState(
        "read", "", {KEY: _entry(3, NOW + timedelta(hours=2))}, PRIOR_RUN_ID
    )
    decision = module.decide_alerts(
        [_alert(module, CHECK, 3)], prior, NOW, timedelta(hours=24)
    )
    assert [reason for _a, reason, _p in decision.to_post] == ["prior_entry_in_future"]
    assert decision.suppressed == []


def test_a_dry_run_posts_nothing_and_advances_no_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    harness = _Harness(
        module,
        monkeypatch,
        tmp_path,
        failing={CHECK: 3},
        prior_zip=_zip(_prior_report({})),
    )
    module._now = lambda: NOW
    code = module.main(
        [
            "--owner", OWNER, "--repo", REPO, "--no-scheduled",
            "--policy", str(POLICY), "--report", str(harness.report),
            "--prior-state-workflow", WORKFLOW_FILE, "--dry-run",
        ]
    )  # fmt: skip
    assert code == 0
    assert harness.posts == []
    assert harness.payload["dedup"]["dry_run"] is True
    assert harness.payload["dedup"]["delivered"] is None
    assert harness.payload["alert_state"]["entries"] == {}
    assert "dry run, nothing sent" in harness.summary_text


def test_summary_table_cells_escape_the_key_separator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unescaped '|' in a markdown table cell splits the column."""
    module = _module()
    prior = _zip(_prior_report({KEY: _entry(3, NOW - timedelta(hours=1))}))
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=prior
    )
    harness.run(NOW)
    assert f"| check\\|{OWNER}/{REPO}\\|{CHECK} |" in harness.summary_text


def test_the_prior_run_lookup_is_scoped_to_the_runs_own_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A feature-branch dispatch must not read or seed dev's alert state."""
    module = _module()
    harness = _Harness(
        module, monkeypatch, tmp_path, failing={CHECK: 3}, prior_zip=None
    )
    harness.run(NOW)
    assert harness.runs_queries, "the prior-run listing was never queried"
    assert all("&branch=dev" in q for q in harness.runs_queries)
    # Not filtered to completed: the previous run may still be in progress
    # (its other job running) after its report was uploaded.
    assert not any("status=" in q for q in harness.runs_queries)


def test_the_alerter_job_cannot_overlap_itself() -> None:
    """Overlapping runs would each read the run BEFORE the other and re-post."""
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = doc["jobs"]["nonrequired-check-failure-rate"]
    assert job["concurrency"]["cancel-in-progress"] is False
    assert "github.ref" in job["concurrency"]["group"]
