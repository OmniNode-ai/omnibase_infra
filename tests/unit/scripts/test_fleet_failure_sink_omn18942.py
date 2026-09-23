# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The fleet failure sink (OMN-18942).

WHAT IS BEING PINNED, AND WHY EACH TEST IS THE FALSIFIER OF A REAL FAILURE

The ticket's finding is not "no alerter exists". One exists, it is correct, it
runs every ten minutes, and it names fifteen continuously-red scheduled
workflows into a destination that was never configured. So the tests here are
about the HOP, not about the evaluation:

* a fleet row must reach the digest AND the active-issue list. The OMN-15525
  defect in this very reporter was a row shape whose status lived in a column
  no selector read, so a dead runtime lane rendered in the message body while
  the alert branch computed zero issues and paged nobody. A new row family is
  the same hazard, so it is asserted on both surfaces.
* an unreadable source must be a row that names itself. An empty green and a
  sweep that never ran are the two states this whole family exists to tell
  apart.
* a dry run must not disarm the next real alert. Found while building this.
* severity must be derived from the report's own counters and the policy's own
  threshold, never from a rate with no denominator behind it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTER = REPO_ROOT / "deploy" / "maintenance" / "omninode-system-slack-report.sh"
SYNC = REPO_ROOT / "deploy" / "maintenance" / "omninode-host-maintenance-sync.sh"
PROBE = REPO_ROOT / "scripts" / "omninode-fleet-failure-probe.py"
EVALUATOR = REPO_ROOT / "scripts" / "ci" / "nonrequired_check_failure_rate.py"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
ZOMBIE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-ci-zombie-detector.yml"
RUNTIME_POLICY_ENV = REPO_ROOT / "docker" / "runtime-policy.env"


def _load_probe() -> Any:
    """Import the probe by path: its filename is not an identifier."""
    spec = importlib.util.spec_from_file_location("fleet_failure_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe_module = _load_probe()


# ---------------------------------------------------------------------------
# A minimal reporter sandbox.
#
# Deliberately NOT imported from the sibling suite: these tests assert on one
# row family and a state-file write, and a harness shared with the lane-probe
# tests would couple this file to fixtures about disk classification and
# container health that it has no opinion about.
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _stub_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    # Healthy everywhere: this suite is about fleet rows, and a stubbed outage
    # would only add unrelated CRITICAL rows to every assertion.
    _write(
        bin_dir / "curl",
        """#!/usr/bin/env bash
out=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  [[ "${args[$i]}" == "-o" ]] && out="${args[$((i+1))]}"
done
[[ -n "$out" ]] && printf '%s' '{"status":"healthy","healthy":true,"is_running":true}' >"$out"
printf '200'
exit 0
""",
    )
    _write(bin_dir / "docker", "#!/usr/bin/env bash\nexit 0\n")
    _write(
        bin_dir / "df",
        '#!/usr/bin/env bash\necho "Filesystem 1G-blocks Used Avail Use% Mounted"\n'
        'echo "target 1832G 75G 1664G 5%"\n',
    )
    _write(bin_dir / "hostname", "#!/usr/bin/env bash\necho omninode-pc\n")
    _write(bin_dir / "flock", "#!/usr/bin/env bash\nexit 0\n")
    if shutil.which("sha256sum") is None:
        _write(bin_dir / "sha256sum", "#!/usr/bin/env bash\nshasum -a 256\n")
    # `timeout(1)` is shimmed UNCONDITIONALLY, and the reason is worth stating
    # rather than leaving as a quirk. The reporter bounds the probe with it; the
    # `.201` host is Linux and has it; macOS ships none, and where a developer
    # has one it is under a Homebrew prefix this harness deliberately does not
    # put on PATH (no machine-specific absolute paths). Without the shim every
    # fleet row here arrives as "probe exited non-zero" -- the correct
    # fail-closed answer to a missing binary, and no evidence at all about the
    # rows these tests are for.
    #
    # The consequence, said plainly: this suite does NOT exercise the bounded
    # call, on any platform. What it exercises is row selection, rendering and
    # the state-file write. The timeout branch's own behaviour is asserted by
    # `test_probe_exiting_nonzero_is_a_warning_not_a_green`, which reaches the
    # same code path through a probe that exits non-zero by itself.
    _write(bin_dir / "timeout", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    return bin_dir


def _stage(tmp_path: Path, bin_dir: Path) -> Path:
    staged = tmp_path / "staged-reporter.sh"
    lines = REPORTER.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("PATH="):
            lines.insert(index + 1, f'PATH="{bin_dir}:$PATH"\n')
            break
    else:  # pragma: no cover - the artifact pins PATH; a miss is a real defect
        raise AssertionError("no PATH assignment in the reporter")
    staged.write_text("".join(lines))
    staged.chmod(0o755)
    return staged


def _run_reporter(
    tmp_path: Path, *, mode: str = "dry-run", extra_env: dict[str, str] | None = None
) -> str:
    bin_dir = _stub_bin(tmp_path)
    state = tmp_path / "state"
    logs = tmp_path / "logs"
    env = {
        "PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
        "OMNINODE_ALERT_STATE_DIR": str(state),
        "OMNINODE_ALERT_LOG_DIR": str(logs),
        "OMNINODE_ALERT_LOCK_FILE": str(tmp_path / "lock"),
        "OMNINODE_INFRA_REPO_ROOT": str(REPO_ROOT),
        "OMNINODE_RUNTIME_POLICY_ENV": str(RUNTIME_POLICY_ENV),
        # The neighbouring probes acquire network dependencies this suite has
        # no business on; their own rows are asserted in their own suites.
        # OMN-18944 added the third of them, whose row would otherwise land in
        # this suite's warning count and in its active-issue list.
        "OMNINODE_CI_PROBE_ENABLED": "0",
        # OMN-18949: the reporter also reads the lane census probe from
        # collect(), and on a runner there is no live snapshot, so it emits a
        # WARNING row and moves this file's issue counts. These cases are about
        # the FLEET rows; the census rows are asserted in
        # test_lane_census_sink_omn18949.py, which switches it back on.
        "OMNINODE_CENSUS_PROBE_ENABLED": "0",
        "OMNINODE_RUNNER_TREE_CHECK_ENABLED": "0",
        "OMNINODE_BACKUP_GATE_CHECK_ENABLED": "0",
        "SLACK_BOT_TOKEN": "x",
        "SLACK_CHANNEL_ID": "C-TEST",
    }
    env.update(extra_env or {})
    proc = subprocess.run(
        ["bash", str(_stage(tmp_path, bin_dir)), "--mode", mode],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    out = proc.stdout
    if not out.strip():
        found = sorted(logs.glob("*.log"))
        if found:
            out = found[-1].read_text()
    assert out.strip(), f"no report: rc={proc.returncode} stderr={proc.stderr[-1500:]}"
    return out


def _stub_probe(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "stub-fleet-probe.py"
    _write(path, f"#!/usr/bin/env python3\nprint({rows!r}, end='')\n")
    return path


# ---------------------------------------------------------------------------
# The hop: a fleet row must reach BOTH the section and the active-issue list
# ---------------------------------------------------------------------------


def test_fleet_rows_reach_the_section_and_the_active_issue_list(
    tmp_path: Path,
) -> None:
    """The OMN-15525 defect, prevented for a new row family.

    A row rendering in the message body while contributing nothing to the issue
    counters is exactly how this reporter once printed three CRITICAL lanes
    under the header `0 critical` and paged nobody.
    """
    rows = (
        "fleet|OK|scheduled-workflows|scanned 13 repos, 77 workflows; 2 above threshold\n"
        "fleet|CRITICAL|sched/omni_home/scheduled-gap-detect.yml|failed 100.0% (8/8) over 7d\n"
        "fleet|WARNING|lab-pass/dev-head|NO passing lab-pass receipt for dev head abc123def456\n"
    )
    report = _run_reporter(
        tmp_path,
        extra_env={"OMNINODE_FLEET_PROBE_SCRIPT": str(_stub_probe(tmp_path, rows))},
    )

    assert "*Fleet failures*" in report
    assert "scheduled-gap-detect.yml" in report
    # The clean heartbeat renders too: "swept and found nothing" must be
    # distinguishable from "the sweep never ran".
    assert "scanned 13 repos" in report

    issues = report.split("*Active issues*", 1)[1]
    assert "scheduled-gap-detect.yml" in issues, (
        "a CRITICAL fleet row rendered in the digest but never reached the "
        "active-issue list -- the OMN-15525 shape"
    )
    assert "lab-pass/dev-head" in issues
    assert "scheduled-workflows" not in issues, "an OK row must not read as an issue"
    assert "*1 critical*" in report
    assert "*1 warning*" in report


def test_a_clean_tick_still_renders_the_fleet_section(tmp_path: Path) -> None:
    """An absent section reads as health. It must say it found nothing."""
    report = _run_reporter(tmp_path, extra_env={"OMNINODE_FLEET_PROBE_ENABLED": "0"})
    assert "*Fleet failures*" in report
    assert "No fleet failure rows this tick" in report


# ---------------------------------------------------------------------------
# Fail-closed: every "could not look" is a row that names itself
# ---------------------------------------------------------------------------


def test_missing_probe_is_a_named_row_not_silence(tmp_path: Path) -> None:
    report = _run_reporter(
        tmp_path,
        extra_env={"OMNINODE_FLEET_PROBE_SCRIPT": str(tmp_path / "does-not-exist.py")},
    )
    assert "probe script missing or unreadable" in report
    assert "probe script missing or unreadable" in report.split("*Active issues*", 1)[1]


def test_probe_producing_no_rows_is_a_warning_not_a_green(tmp_path: Path) -> None:
    report = _run_reporter(
        tmp_path,
        extra_env={"OMNINODE_FLEET_PROBE_SCRIPT": str(_stub_probe(tmp_path, ""))},
    )
    assert "probe produced no rows" in report
    assert "fleet failure state unknown" in report


def test_probe_exiting_nonzero_is_a_warning_not_a_green(tmp_path: Path) -> None:
    failing = tmp_path / "failing-probe.py"
    _write(failing, "#!/usr/bin/env python3\nimport sys; sys.exit(3)\n")
    report = _run_reporter(
        tmp_path, extra_env={"OMNINODE_FLEET_PROBE_SCRIPT": str(failing)}
    )
    assert "probe exited non-zero or timed out" in report


# ---------------------------------------------------------------------------
# A dry run must not disarm the next real alert
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write_the_alert_state_file(tmp_path: Path) -> None:
    """The defect found while building this.

    The per-key decisions are computed in every mode but only `alert` posts, so
    persisting them from a dry run records `notified=1` for keys nobody was
    told about and the next real tick stays silent. The documented way to
    inspect this reporter would swallow the first alert for every standing
    issue.
    """
    rows = (
        "fleet|CRITICAL|sched/omni_home/scheduled-gap-detect.yml|failed 100.0% (8/8)\n"
    )
    probe = _stub_probe(tmp_path, rows)
    _run_reporter(tmp_path, extra_env={"OMNINODE_FLEET_PROBE_SCRIPT": str(probe)})
    state_file = tmp_path / "state" / "omninode-system-alert-keys.tsv"
    assert not state_file.exists(), (
        "a dry run wrote the alert state file; the next real tick would read "
        "these keys as already notified and stay silent"
    )


def test_alert_mode_does_write_the_state_file(tmp_path: Path) -> None:
    """The positive control for the test above.

    Without it, a guard that disabled the state machine entirely would pass the
    dry-run assertion and silently break de-duplication in production.
    """
    rows = (
        "fleet|CRITICAL|sched/omni_home/scheduled-gap-detect.yml|failed 100.0% (8/8)\n"
    )
    probe = _stub_probe(tmp_path, rows)
    _run_reporter(
        tmp_path,
        mode="alert",
        extra_env={"OMNINODE_FLEET_PROBE_SCRIPT": str(probe)},
    )
    state_file = tmp_path / "state" / "omninode-system-alert-keys.tsv"
    assert state_file.exists(), "alert mode must persist per-key alert state"
    assert "sched/omni_home/scheduled-gap-detect.yml" in state_file.read_text()


# ---------------------------------------------------------------------------
# Severity is derived, never a second tuned number
# ---------------------------------------------------------------------------


def _report_with(workflow: str, failures: int, observed: int) -> dict[str, Any]:
    rate = round(100.0 * failures / observed, 1) if observed else 0.0
    return {
        "schema": "nonrequired_check_report/v1",
        "repos": {
            "OmniNode-ai/omni_home": {
                "scheduled": {
                    "window_days": 7,
                    "threshold_pct": 10.0,
                    "workflows": {workflow: {}},
                    "alerts": [
                        {
                            "workflow": workflow,
                            "failures": failures,
                            "observed": observed,
                            "rate_pct": rate,
                            "last_failure_url": "https://example.invalid/1",
                        }
                    ],
                }
            }
        },
    }


def test_never_green_above_the_threshold_is_critical() -> None:
    rows, heartbeat = probe_module._rows_from_report(
        _report_with(".github/workflows/scheduled-gap-detect.yml", 8, 8), 3
    )
    assert [r["status"] for r in rows] == ["CRITICAL"]
    assert rows[0]["key"] == "sched/omni_home/scheduled-gap-detect.yml"
    assert "has not succeeded once" in rows[0]["detail"]
    assert "1 above threshold" in heartbeat


def test_a_hundred_percent_rate_over_one_run_is_not_critical() -> None:
    """A rate with no denominator is not evidence of a dead surface.

    Two such rows were in the first live sweep. Escalating them would page on a
    workflow that has fired once, all week, and happened to fail.
    """
    rows, _ = probe_module._rows_from_report(
        _report_with(".github/workflows/promotion-pr-opener.yml", 1, 1), 3
    )
    assert [r["status"] for r in rows] == ["WARNING"]
    assert "has not succeeded once" not in rows[0]["detail"]


def test_a_flaky_workflow_above_the_rate_threshold_is_a_warning() -> None:
    rows, _ = probe_module._rows_from_report(
        _report_with(".github/workflows/chain-canary.yml", 91, 95), 3
    )
    assert [r["status"] for r in rows] == ["WARNING"]


def test_the_critical_denominator_is_the_policys_own_failure_threshold() -> None:
    """Not an independent constant: the policy already declares this number."""
    block = yaml.safe_load(POLICY.read_text())["route"]["nonrequired_check_alert"]
    threshold = int(block["failure_threshold"])
    at = probe_module._rows_from_report(
        _report_with("w.yml", threshold, threshold), threshold
    )[0]
    below = probe_module._rows_from_report(
        _report_with("w.yml", threshold - 1, threshold - 1), threshold
    )[0]
    assert at[0]["status"] == "CRITICAL"
    assert below[0]["status"] == "WARNING"


# ---------------------------------------------------------------------------
# An unreadable sweep must not read as a recovery
# ---------------------------------------------------------------------------


def test_an_unreadable_sweep_keeps_standing_findings_standing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dropping the rows would be worse than serving them stale.

    `row_key()` keys on the label, so a key that vanishes is read by the
    reporter's state machine as a RECOVERY and posts "resolved" for a finding
    nobody fixed.
    """
    cache = tmp_path / "fleet-scheduled-sweep.json"
    cache.write_text(
        json.dumps(
            {
                "evaluated_at": 0,  # ancient, so the refresh path is taken
                "heartbeat": "scanned 13 repos; 1 above threshold",
                "rows": [
                    {
                        "status": "CRITICAL",
                        "key": "sched/omni_home/scheduled-gap-detect.yml",
                        "detail": "failed 100.0% (8/8) over 7d",
                    }
                ],
            }
        )
    )
    monkey_env = {
        "OMNINODE_FLEET_SWEEP_CACHE": str(cache),
        # Unresolvable evaluator: the "could not look" branch.
        "OMNINODE_FLEET_EVALUATOR": str(tmp_path / "absent.py"),
        "OMNINODE_FLEET_POLICY_FILE": str(POLICY),
    }
    import os

    previous = {k: os.environ.get(k) for k in monkey_env}
    os.environ.update(monkey_env)
    try:
        probe_module.check_scheduled_workflows(tmp_path)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    printed = capsys.readouterr().out
    assert "source unreadable" in printed
    assert "sched/omni_home/scheduled-gap-detect.yml" in printed, (
        "a standing finding was dropped on an unreadable sweep; the reporter "
        "would post a resolution for it"
    )


# ---------------------------------------------------------------------------
# One declared repository list, one evaluator
# ---------------------------------------------------------------------------


def test_policy_declares_the_fleet_repo_list() -> None:
    block = yaml.safe_load(POLICY.read_text())["route"]["nonrequired_check_alert"]
    repos = block["fleet_repos"]
    assert isinstance(repos, list) and len(repos) >= 9
    # The registry repository is the named case: its cross-repo drift detector
    # was red 12 of 12 daily runs and was outside the fleet's only alerter by
    # construction.
    assert "omni_home" in repos
    assert len(set(repos)) == len(repos), "the fleet list carries a duplicate"


def test_load_policy_requires_the_fleet_repo_list(tmp_path: Path) -> None:
    """Rule 8: a silently-defaulted list is how this read three repos for a month."""
    spec = importlib.util.spec_from_file_location("nrcfr", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    doc = yaml.safe_load(POLICY.read_text())
    del doc["route"]["nonrequired_check_alert"]["fleet_repos"]
    stripped = tmp_path / "policy.yaml"
    stripped.write_text(yaml.safe_dump(doc))
    with pytest.raises(KeyError, match="fleet_repos"):
        module.load_policy(stripped)


def _evaluator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EVALUATOR), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _load_evaluator() -> Any:
    """Import the evaluator by path: its directory is not a package."""
    spec = importlib.util.spec_from_file_location(
        "nonrequired_check_failure_rate", EVALUATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator_module = _load_evaluator()


# ---------------------------------------------------------------------------
# The destination must exist, or the job must say so in its exit code.
#
# Measured 2026-09-21, live: no chat secret of any name exists at OmniNode-ai
# organisation scope or in omnibase_infra / omninode_infra repository scope.
# This job reaches post_slack_alert ONLY when it has findings to deliver, and
# with no credential it printed a line and returned, leaving the run green. A
# finding that reaches nobody while the job reports success is the exact defect
# class this ticket was filed about, reproduced by its own fix.
#
# The asymmetry that stays: raising an alert is still not this job failing.
# What fails is having an alert and no destination for it.
# ---------------------------------------------------------------------------


class _Alertish:
    def __init__(self, detail: str) -> None:
        self.detail = detail


def test_findings_with_no_destination_credential_refuse(monkeypatch) -> None:
    """RED test. An undeliverable finding may not exit zero."""
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)
    with pytest.raises(evaluator_module.AlertDestinationMissingError) as excinfo:
        evaluator_module.post_slack_alert([_Alertish("omnibase_infra never green")])
    assert "SLACK_BOT_TOKEN" in str(excinfo.value)


def test_an_empty_destination_credential_is_treated_as_absent(monkeypatch) -> None:
    """A secret that resolves to the empty string is the same hole as no secret."""
    monkeypatch.setenv("SLACK_BOT_TOKEN", "")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")
    with pytest.raises(evaluator_module.AlertDestinationMissingError):
        evaluator_module.post_slack_alert([_Alertish("x")])


def test_a_missing_channel_refuses_as_loudly_as_a_missing_token(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-not-a-real-token")
    monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)
    with pytest.raises(evaluator_module.AlertDestinationMissingError) as excinfo:
        evaluator_module.post_slack_alert([_Alertish("x")])
    assert "SLACK_CHANNEL_ID" in str(excinfo.value)


def test_the_refusal_reason_is_one_line(monkeypatch) -> None:
    """A multi-line reason in an ::error:: annotation is truncated to noise."""
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)
    with pytest.raises(evaluator_module.AlertDestinationMissingError) as excinfo:
        evaluator_module.post_slack_alert([_Alertish("x")])
    assert "\n" not in str(excinfo.value)


def test_the_refusal_names_no_credential_value(monkeypatch) -> None:
    """NEGATIVE CONTROL: the reason names the reference, never the secret."""
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-a-real-looking-value")
    monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)
    with pytest.raises(evaluator_module.AlertDestinationMissingError) as excinfo:
        evaluator_module.post_slack_alert([_Alertish("x")])
    assert "xoxb-a-real-looking-value" not in str(excinfo.value)


def test_a_present_credential_is_unchanged(monkeypatch) -> None:
    """POSITIVE CONTROL: with a destination, the post path runs as before.

    The refusal must be reachable ONLY through the absent-credential branch;
    if it fired with both values present, the change would have converted a
    silent success into a permanent red.
    """
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-not-a-real-token")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")
    posted: list[Any] = []
    monkeypatch.setattr(
        evaluator_module.urllib.request,
        "urlopen",
        lambda request, timeout=10: posted.append(request) or _NullResponse(),
    )
    evaluator_module.post_slack_alert([_Alertish("x")])
    assert len(posted) == 1


class _NullResponse:
    def __enter__(self) -> _NullResponse:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def read(self) -> bytes:
        return b"{}"


def test_no_findings_means_no_destination_is_required() -> None:
    """The call site only delivers when there is something to deliver.

    A clean sweep with no chat secret is not a silent failure -- nothing
    failed to reach anybody -- so the guard must sit on the delivery path and
    not on the evaluation path.
    """
    source = EVALUATOR.read_text()
    assert "if combined_alerts and not args.dry_run:" in source


def test_selecting_no_repositories_is_an_error_not_an_empty_sweep() -> None:
    """An empty repo list would exit 0 having looked at nothing."""
    result = _evaluator("--policy", str(POLICY), "--report", "/dev/null")
    assert result.returncode != 0
    assert "no repositories selected" in result.stderr


def test_repo_and_repos_from_policy_are_mutually_exclusive() -> None:
    result = _evaluator(
        "--policy",
        str(POLICY),
        "--report",
        "/dev/null",
        "--repo",
        "omnibase_infra",
        "--repos-from-policy",
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_scheduled_only_and_no_scheduled_are_mutually_exclusive() -> None:
    result = _evaluator(
        "--policy",
        str(POLICY),
        "--report",
        "/dev/null",
        "--repos-from-policy",
        "--scheduled-only",
        "--no-scheduled",
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


# ---------------------------------------------------------------------------
# Wiring that a refactor must not silently undo
# ---------------------------------------------------------------------------


def test_the_reporter_calls_the_fleet_probe_from_collect() -> None:
    body = REPORTER.read_text()
    collect = body.split("\ncollect() {", 1)[1].split("\nsnapshot=$(collect)", 1)[0]
    assert "check_fleet_failures" in collect, (
        "the probe is defined but never called -- a detector wired to nothing"
    )


def test_the_scheduled_sweep_has_exactly_one_caller() -> None:
    """Two surfaces evaluating the same runs into different destinations is the
    divergence a shared evaluator exists to prevent. Since the registry
    repository moved to the Actions job (scheduled_actions_repos), "one caller"
    is per repository; test_fleet_scheduled_share_omn18942 pins the partition."""
    assert "--no-scheduled" in ZOMBIE_WORKFLOW.read_text(), (
        "the GitHub Actions job evaluates its --repo list's scheduled runs "
        "again; the .201 reporter already owns those repositories' sweep"
    )


def test_the_maintenance_manifest_installs_the_probe_and_what_it_reads() -> None:
    """An artifact absent from the manifest is a merged, never-installed probe."""
    manifest = SYNC.read_text()
    for relpath in (
        "scripts/omninode-fleet-failure-probe.py",
        "scripts/ci/nonrequired_check_failure_rate.py",
        "scripts/ci/lab_pass_receipt.py",
        "config/runner_routing_policy.yaml",
    ):
        assert f'"{relpath}|' in manifest, f"{relpath} is not in the sync manifest"


def test_the_probe_resolves_its_companions_in_both_layouts() -> None:
    """The repo nests them under scripts/ci/; the host sync installs them flat."""
    assert (
        probe_module._resolve_sibling("nonrequired_check_failure_rate.py", "__unset__")
        == EVALUATOR
    )
    assert (
        probe_module._resolve_sibling("lab_pass_receipt.py", "__unset__")
        == REPO_ROOT / "scripts" / "ci" / "lab_pass_receipt.py"
    )


def test_the_probe_emits_no_credential_name_with_a_value() -> None:
    """The probe normalises token names; it must never print one."""
    body = PROBE.read_text()
    assert "print(token" not in body
    assert "SLACK_BOT_TOKEN" in body, "the sanity of this test depends on the name"
    # Named only where it is REMOVED from the child environment.
    for line in body.splitlines():
        if "SLACK_BOT_TOKEN" in line:
            assert "pop(" in line, line


# ---------------------------------------------------------------------------
# One unreadable repository must not blank the fleet
# ---------------------------------------------------------------------------


def _report_with_error(readable_alerts: int = 1) -> dict[str, Any]:
    repos: dict[str, Any] = {
        "OmniNode-ai/omni_home": {
            "error": "gh api ... failed: gh: Not Found (HTTP 404)"
        },
        "OmniNode-ai/omnibase_infra": {
            "scheduled": {
                "window_days": 7,
                "threshold_pct": 10.0,
                "workflows": {"a.yml": {}},
                "alerts": [
                    {
                        "workflow": ".github/workflows/chain-canary.yml",
                        "failures": 91,
                        "observed": 95,
                        "rate_pct": 95.8,
                        "last_failure_url": "https://example.invalid/2",
                    }
                ][:readable_alerts],
            }
        },
    }
    return {"schema": "nonrequired_check_report/v1", "repos": repos}


def test_an_unreadable_repo_is_its_own_row_and_the_others_still_report() -> None:
    """Measured on `.201`, 2026-09-20.

    The host token reads twelve of the thirteen fleet repositories and returns
    404 on the registry repository -- the one whose red scheduled workflow this
    ticket names. Before the per-repo split, that single 404 aborted the sweep
    and produced no rows at all for any repository.
    """
    rows, _ = probe_module._rows_from_report(_report_with_error(), 3)
    keys = [r["key"] for r in rows]
    assert "sched/omni_home/source-unreadable" in keys, (
        "an unreadable repository vanished; a clean board and an unscanned "
        "repository would be indistinguishable"
    )
    assert "sched/omnibase_infra/chain-canary.yml" in keys, (
        "one unreadable repository blanked the repositories that WERE read"
    )
    assert (
        "UNKNOWN, not clean"
        in rows[keys.index("sched/omni_home/source-unreadable")]["detail"]
    )


def test_the_heartbeat_states_the_denominator_not_just_the_findings() -> None:
    _, heartbeat = probe_module._rows_from_report(_report_with_error(), 3)
    assert "1/2 repos" in heartbeat, heartbeat
    assert "1 repo(s) unreadable" in heartbeat, heartbeat


def test_the_evaluator_reports_a_read_failure_as_a_repo_error_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    """The exit code is for CI; the report is for a renderer. Both must be true.

    Driven with a `gh` stub on PATH that 404s for one repository and returns an
    empty workflow list for the other, so this asserts the real subprocess path
    rather than a patched function.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    _write(
        stub_bin / "gh",
        """#!/usr/bin/env bash
if [[ "$*" == *"/omni_home/"* ]]; then
  echo "gh: Not Found (HTTP 404)" >&2
  exit 1
fi
echo '{"workflows": []}'
""",
    )
    policy_doc = yaml.safe_load(POLICY.read_text())
    policy_doc["route"]["nonrequired_check_alert"]["fleet_repos"] = [
        "omni_home",
        "omnibase_infra",
    ]
    # Both repositories in the host's share, so the host selection reads both.
    policy_doc["route"]["nonrequired_check_alert"]["scheduled_actions_repos"] = []
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump(policy_doc))
    report = tmp_path / "report.json"

    import os

    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env['PATH']}"
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--repos-from-policy",
            "--scheduled-only",
            "--policy",
            str(policy),
            "--report",
            str(report),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert result.returncode != 0, (
        "a repository that could not be read exited 0; CI would read an "
        "unevaluated repository as a clean one"
    )
    assert report.is_file(), "the report was not written, so the partial result is lost"
    payload = json.loads(report.read_text())
    assert "error" in payload["repos"]["OmniNode-ai/omni_home"]
    assert "error" not in payload["repos"]["OmniNode-ai/omnibase_infra"]
