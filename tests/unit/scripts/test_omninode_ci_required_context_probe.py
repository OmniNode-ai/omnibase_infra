# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15550 -- a required status check that never reports must alarm.

WHAT IS UNDER TEST
    ``scripts/omninode-ci-required-context-probe.py`` and its
    fold-in to ``deploy/maintenance/omninode-system-slack-report.sh``. Every
    test drives those real files -- the ones installed on ``.201`` -- via
    ``subprocess``, never a Python re-implementation of their logic
    (``feedback_test_the_artifact_that_runs``).

THE REPLAY IS REAL RECORDED STATE, NOT A MOCK
    Fixtures under ``tests/fixtures/omn15550/`` were captured from the live
    GitHub API during and after the 2026-07-30 outage (OMN-15536). The only
    transformation is a point-in-time filter: drop check-runs not yet started
    at T, and render those not yet finished at T as ``in_progress`` with no
    conclusion. That reconstructs exactly what the API would have returned at
    T -- so "would this have caught the outage?" is answered against the real
    thing, not against a hand-authored approximation of it.

    ``absent/`` is the load-bearing one: PR #2575 head ``2173c0c5`` at
    2026-07-30T20:00:00Z, mid-outage. **61 check-runs present, none failing,
    and ``CI Summary`` -- the sole required context on that branch -- simply
    not there.** A green board and a wedged repo at the same time.

WHY THE NEGATIVE CASES CARRY EQUAL WEIGHT
    A detector that alarms on the outage but also on healthy states is not a
    detector, it is noise that gets muted and then misses the next outage. Four
    controls are asserted here, all from real heads:
      * ``healed``     -- same head after the repin; context present.
      * ``pending``    -- context present but ``in_progress``. Must NEVER alarm
                          at any age; ageing pending checks is OMN-12560's job.
      * ``late_start`` -- context genuinely absent 32 min in, with runs still
                          in flight. On head ``39c88810`` ``CI Summary`` really
                          did start 47 min after CI began, so alarming here
                          would be a false positive on an ordinary slow
                          pipeline.
      * ``paginated``  -- context present, but only on page 2 of the check-runs
                          list. A probe that reads one page calls it absent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = REPO_ROOT / "scripts" / "omninode-ci-required-context-probe.py"
REPORTER = REPO_ROOT / "deploy" / "maintenance" / "omninode-system-slack-report.sh"
SYNC = REPO_ROOT / "deploy" / "maintenance" / "omninode-host-maintenance-sync.sh"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn15550"

# The real head/PR/run identities the fixtures were captured from. Asserted on
# so a regenerated fixture that quietly points somewhere else fails loudly
# rather than passing against different data.
OUTAGE_SHA = "2173c0c5"
OUTAGE_PR = 2575
LATE_START_PR = 2581
ZEROJOB_RUN = "30583527197"
HEALTHY_RUN = "30586119120"
REQUIRED_CONTEXT = "CI Summary"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to drive the reporter"
)


def run_probe(
    scenario: str,
    now: str,
    *,
    extra_env: dict[str, str] | None = None,
    args: tuple[str, ...] = ("--skip-zero-job-scan",),
) -> list[str]:
    """Drive the real probe against a recorded scenario; return its rows."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # this Mac's ambient PYTHONPATH shadows stdlib
    env.update(
        {
            "OMNINODE_CI_PROBE_FIXTURE_DIR": str(FIXTURES / scenario),
            "OMNINODE_CI_PROBE_REPOS": "omnibase_infra",
            "OMNINODE_CI_PROBE_NOW": now,
        }
    )
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["python3", str(PROBE), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\nstderr:\n{proc.stderr[-2000:]}"
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def criticals(rows: list[str]) -> list[str]:
    return [r for r in rows if r.split("|")[1] == "CRITICAL"]


def warnings(rows: list[str]) -> list[str]:
    return [r for r in rows if r.split("|")[1] == "WARNING"]


def heartbeat(rows: list[str]) -> str:
    matches = [r for r in rows if r.split("|")[2] == "required-contexts"]
    assert len(matches) == 1, f"expected exactly one heartbeat row, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# The outage itself
# ---------------------------------------------------------------------------


def test_fixture_reproduces_the_outage_condition() -> None:
    """Provenance guard: the replay input must really be the invisible state.

    If the recorded head ever stops showing "many contexts present, required
    one absent", every assertion below becomes vacuous while still passing.
    """
    payloads = list((FIXTURES / "absent").glob("*check_runs*.json"))
    assert payloads, "absent scenario has no check-runs fixture"
    names = [
        run["name"]
        for path in payloads
        for run in json.loads(path.read_text())["check_runs"]
    ]
    assert len(names) >= 50, (
        f"expected a densely-reporting head, got {len(names)} check-runs -- "
        "the point of this replay is that the board looked healthy"
    )
    assert REQUIRED_CONTEXT not in names, (
        f"{REQUIRED_CONTEXT!r} is present in the mid-outage fixture; the replay "
        "no longer reproduces the absent condition it exists to prove"
    )


def test_outage_replay_alarms_naming_pr_and_context() -> None:
    """The 2026-07-30 condition must produce a CRITICAL that names the culprit."""
    rows = run_probe("absent", "2026-07-30T20:00:00Z")
    crit = criticals(rows)
    assert len(crit) == 1, "expected exactly one CRITICAL, got:\n" + "\n".join(rows)
    row = crit[0]
    assert f"#{OUTAGE_PR}" in row, row
    assert REQUIRED_CONTEXT in row, row
    assert OUTAGE_SHA in row, row
    # An operator reading the Slack line must be able to tell this apart from a
    # red check without opening GitHub.
    assert "never reported" in row, row
    assert "absent=1" in heartbeat(rows)


def test_absent_row_key_is_stable_across_ticks() -> None:
    """De-duplication identity must not carry the age, or every tick re-pages.

    ``row_key()`` in the reporter de-duplicates on ``ci|<key>``, so a key that
    embeds a changing number defeats state-change suppression and turns one
    stranded PR into an alert every 15 minutes.
    """
    first = criticals(run_probe("absent", "2026-07-30T20:00:00Z"))[0].split("|")[2]
    later = criticals(run_probe("absent", "2026-07-30T21:00:00Z"))[0].split("|")[2]
    assert first == later, f"key changed between ticks: {first!r} -> {later!r}"
    assert not re.search(r"\d+m", first), f"key embeds an age: {first!r}"


# ---------------------------------------------------------------------------
# False-positive controls -- each is a real head that must stay quiet
# ---------------------------------------------------------------------------


def test_healed_head_is_quiet() -> None:
    """Same head after the repin: context present and successful."""
    rows = run_probe("healed", "2026-07-30T22:00:00Z")
    assert criticals(rows) == [], "\n".join(rows)
    assert "absent=0" in heartbeat(rows)


def test_pending_required_context_never_alarms() -> None:
    """Reported-but-running is visible today and belongs to OMN-12560, not here."""
    rows = run_probe("pending", "2026-07-31T00:00:00Z")
    assert criticals(rows) == [], "\n".join(rows)
    hb = heartbeat(rows)
    assert "absent=0" in hb, hb
    assert "pending=1" in hb, f"pending context was not counted: {hb}"


def test_pending_does_not_alarm_even_when_very_old() -> None:
    """The PENDING/ABSENT boundary is categorical, not a longer timeout.

    Ageing a pending check into a CRITICAL here would double-page with
    OMN-12560 for a single stalled check.
    """
    rows = run_probe(
        "pending",
        "2026-07-31T12:00:00Z",
        extra_env={"OMNINODE_CI_ABSENT_CEILING_MINUTES": "5"},
    )
    assert criticals(rows) == [], (
        "a pending required context alarmed after 12h; PENDING must never "
        "escalate to ABSENT:\n" + "\n".join(rows)
    )


def test_in_flight_runs_suppress_a_late_starting_required_context() -> None:
    """A required context that has not started yet, while CI is still running.

    Real case: on head ``39c88810`` ``CI Summary`` started 47 minutes after the
    first check-run. At 32 minutes in it is genuinely absent and past the grace
    window -- alarming would be a false positive on an ordinary pipeline.
    """
    rows = run_probe("late_start", "2026-07-30T23:40:00Z")
    assert criticals(rows) == [], (
        "alarmed on a legitimately late required context while runs were still "
        "in flight:\n" + "\n".join(rows)
    )


def test_pagination_prevents_a_phantom_absence() -> None:
    """The required context lives on page 2. One-page reads invent absences."""
    rows = run_probe("paginated", "2026-07-30T22:00:00Z")
    assert criticals(rows) == [], (
        "reported a context absent that is present on page 2 of the check-runs "
        "list -- the probe is not paginating:\n" + "\n".join(rows)
    )


def test_context_satisfied_only_by_a_commit_status_is_not_absent() -> None:
    """Required contexts can be satisfied on either of two GitHub surfaces.

    A check-runs-only probe reports every commit-status context permanently
    absent. This is not hypothetical: on these repos ``CodeRabbit`` posts a
    legacy commit status and appears in the check-runs list zero times, and
    ``gate / CodeRabbit Thread Check`` is a live required context on
    omnibase_core / omnimarket / omniclaude dev.

    The status payload here is a real capture; the required-context list is set
    to that status's name so the union is what the assertion turns on.
    """
    scenario = FIXTURES / "status_surface"
    checkruns = json.loads(next(scenario.glob("*check_runs*.json")).read_text())
    assert not [r for r in checkruns["check_runs"] if r["name"] == "CodeRabbit"], (
        "CodeRabbit appears as a check-run in this fixture; the test no longer "
        "isolates the commit-status surface"
    )
    rows = run_probe("status_surface", "2026-07-30T22:00:00Z")
    assert criticals(rows) == [], (
        "a required context present as a commit status was reported absent -- "
        "the probe is reading check-runs only:\n" + "\n".join(rows)
    )


def test_pagination_fixture_actually_splits_the_required_context() -> None:
    """Non-vacuity guard for the test above.

    If page 1 ever contains the required context, the pagination test passes
    without exercising pagination at all.
    """
    page1 = json.loads(
        (
            FIXTURES
            / "paginated"
            / "repos_OmniNode_ai_omnibase_infra_commits_2173c0c53a46624be1efd4cff91cd07f3f8df13b_check_runs_per_page_100.json"
        ).read_text()
    )
    names1 = [r["name"] for r in page1["check_runs"]]
    assert REQUIRED_CONTEXT not in names1, (
        "page 1 already contains the required context; the pagination test is vacuous"
    )
    assert page1.get("__next__"), "page 1 declares no next page"
    assert len(names1) == 100, f"page 1 should be a full page, got {len(names1)}"


# ---------------------------------------------------------------------------
# Grace window and ceiling
# ---------------------------------------------------------------------------


def test_grace_window_suppresses_a_freshly_pushed_head() -> None:
    """Right after a push, "absent" is indistinguishable from "not started"."""
    rows = run_probe("absent", "2026-07-30T19:25:00Z")
    assert criticals(rows) == [], "alarmed inside the grace window:\n" + "\n".join(rows)


def test_grace_window_is_configurable_and_bites() -> None:
    rows = run_probe(
        "absent",
        "2026-07-30T19:25:00Z",
        extra_env={"OMNINODE_CI_ABSENT_GRACE_MINUTES": "5"},
    )
    assert len(criticals(rows)) == 1, (
        "lowering the grace window did not surface the absence:\n" + "\n".join(rows)
    )


def test_ceiling_lifts_in_flight_suppression() -> None:
    """In-flight suppression must not be able to hide an absence forever.

    Without a ceiling, one permanently-hung run holds the repo in a silent
    blind spot -- re-creating the invisibility this probe exists to remove.
    """
    rows = run_probe(
        "late_start",
        "2026-07-30T23:40:00Z",
        extra_env={"OMNINODE_CI_ABSENT_CEILING_MINUTES": "25"},
    )
    crit = criticals(rows)
    assert len(crit) == 1, (
        "in-flight suppression was never lifted by the ceiling:\n" + "\n".join(rows)
    )
    assert f"#{LATE_START_PR}" in crit[0], crit[0]


# ---------------------------------------------------------------------------
# Zero-job startup-failure signature
# ---------------------------------------------------------------------------


def test_zero_job_run_alarms() -> None:
    """The upstream cause: a run that concluded having created no jobs."""
    rows = run_probe("zerojob", "2026-07-30T21:35:00Z", args=())
    crit = criticals(rows)
    assert len(crit) == 1, "\n".join(rows)
    assert ZEROJOB_RUN in crit[0], crit[0]
    assert "0 jobs" in crit[0], crit[0]


def test_healthy_run_with_jobs_does_not_alarm() -> None:
    """Contrast case: same workflow file, 54 jobs, 11 minutes."""
    rows = run_probe("zerojob_healthy", "2026-07-30T22:25:00Z", args=())
    assert criticals(rows) == [], "\n".join(rows)


def test_zero_job_scan_is_bounded_to_a_recent_window() -> None:
    """An old startup failure must not page forever after it stopped mattering."""
    rows = run_probe("zerojob", "2026-07-31T06:00:00Z", args=())
    assert criticals(rows) == [], (
        "a startup failure from hours earlier is still alarming:\n" + "\n".join(rows)
    )


# ---------------------------------------------------------------------------
# Fail-visible, not fail-silent
# ---------------------------------------------------------------------------


def test_api_error_is_a_visible_warning_not_silence(tmp_path: Path) -> None:
    """ "Could not look" must never render as "nothing is wrong"."""
    scenario = tmp_path / "broken"
    scenario.mkdir()
    (
        scenario / "repos_OmniNode_ai_omnibase_infra_pulls_state_open_per_page_100.json"
    ).write_text(json.dumps({"__error__": "HTTP 503 on /pulls"}))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "OMNINODE_CI_PROBE_FIXTURE_DIR": str(scenario),
            "OMNINODE_CI_PROBE_REPOS": "omnibase_infra",
            "OMNINODE_CI_PROBE_NOW": "2026-07-30T20:00:00Z",
        }
    )
    proc = subprocess.run(
        ["python3", str(PROBE), "--skip-zero-job-scan"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [r for r in proc.stdout.splitlines() if r.strip()]
    warn = warnings(rows)
    assert warn, "an API failure produced no WARNING row:\n" + "\n".join(rows)
    assert "state unknown" in warn[0], warn[0]
    # Deliberately not CRITICAL: an unreachable API is not evidence that PRs are
    # stranded, and paging on every blip is how a channel gets muted.
    assert criticals(rows) == [], (
        "a transient API error escalated to CRITICAL:\n" + "\n".join(rows)
    )


def test_missing_credentials_reports_that_it_did_not_run() -> None:
    """Unauthenticated reads cannot see branch protection -- say so, loudly."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("GH_PAT", None)
    env.pop("GITHUB_TOKEN", None)
    env.pop("OMNINODE_CI_PROBE_FIXTURE_DIR", None)
    proc = subprocess.run(
        ["python3", str(PROBE)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [r for r in proc.stdout.splitlines() if r.strip()]
    assert warnings(rows), "\n".join(rows)
    assert "did NOT run" in rows[0], rows[0]


def test_heartbeat_proves_the_probe_actually_scanned() -> None:
    """A clean board must be distinguishable from a probe that looked at nothing.

    This is the ``reference_detection_shelf_structurally_blind`` guard: 16/16
    sweeps were once green because none of them asserted ``scanned_count > 0``.
    """
    rows = run_probe("healed", "2026-07-30T22:00:00Z")
    hb = heartbeat(rows)
    assert "scanned 1/1 repos" in hb, hb
    assert "1 open non-draft PRs" in hb, hb
    assert "1 required contexts" in hb, hb


def test_zero_repos_scanned_is_a_warning_not_a_clean_board(tmp_path: Path) -> None:
    scenario = tmp_path / "empty"
    scenario.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "OMNINODE_CI_PROBE_FIXTURE_DIR": str(scenario),
            "OMNINODE_CI_PROBE_REPOS": "omnibase_infra",
            "OMNINODE_CI_PROBE_NOW": "2026-07-30T20:00:00Z",
        }
    )
    proc = subprocess.run(
        ["python3", str(PROBE), "--skip-zero-job-scan"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    rows = [r for r in proc.stdout.splitlines() if r.strip()]
    hb = heartbeat(rows)
    assert hb.split("|")[1] == "WARNING", f"scanning nothing reported OK: {hb}"
    assert "scanned 0" in hb, hb


# ---------------------------------------------------------------------------
# The fold-in: rows must actually reach the alert path
# ---------------------------------------------------------------------------


def test_reporter_row_shape_reaches_the_alert_selectors() -> None:
    """A CRITICAL ``ci`` row must raise ``critical_count`` and populate ``issues``.

    This is the whole point of AC6 and it is not obvious: the snapshot carries
    two row shapes and ``row_status()`` reads a different column for each. A
    detector emitting the wrong shape is silently non-alarming -- which is the
    exact defect class this ticket exists to close, reproduced one layer down.
    """
    script = REPORTER.read_text()
    match = re.search(r"^row_status='(.+)'$", script, re.MULTILINE)
    assert match, "row_status helper not found in the reporter"
    awk_fn = match.group(1)

    snapshot = "\n".join(
        [
            "timestamp|2026-07-31T00:00:00Z",
            "host|omninode-pc",
            "disk|OK|/|10/100|90G free|10%",
            "OK|runtime-dev-8085|200|healthy",
            "ci|CRITICAL|absent/omnibase_infra#2575/CI Summary|never reported",
            "ci|OK|required-contexts|scanned 1/1 repos",
        ]
    )
    proc = subprocess.run(
        [
            "awk",
            "-F|",
            awk_fn + '{ s=row_status() } s=="CRITICAL" {c++} END {print c+0}',
        ],
        input=snapshot,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "1", (
        f"a CRITICAL ci row did not register in critical_count (got "
        f"{proc.stdout.strip()!r}) -- the probe would page nobody"
    )


def test_reporter_dedup_key_is_the_ci_row_key() -> None:
    """``row_key()`` must key ``ci`` rows on their stable identity column."""
    script = REPORTER.read_text()
    row_status = re.search(r"^row_status='(.+)'$", script, re.MULTILINE)
    row_key = re.search(r"^row_key='(.+)'$", script, re.MULTILINE)
    assert row_status and row_key
    snapshot = "ci|CRITICAL|absent/omnibase_infra#2575/CI Summary|never reported 44m"
    proc = subprocess.run(
        [
            "awk",
            "-F|",
            row_status.group(1)
            + row_key.group(1)
            + '{ s=row_status() } s=="CRITICAL" {print row_key()}',
        ],
        input=snapshot,
        capture_output=True,
        text=True,
        check=True,
    )
    key = proc.stdout.strip()
    assert key == "ci|absent/omnibase_infra#2575/CI Summary", key
    assert "44m" not in key, f"volatile age leaked into the dedup key: {key}"


def test_reporter_invokes_the_probe_from_collect() -> None:
    """The fold-in must be wired, not merely present in the tree."""
    script = REPORTER.read_text()
    assert "check_ci_required_contexts" in script
    body = script.split("collect() {", 1)[1]
    assert "check_ci_required_contexts" in body, (
        "probe is defined but never called from collect(); its rows would "
        "never enter the snapshot"
    )


def test_reporter_degrades_visibly_when_the_probe_is_missing(tmp_path: Path) -> None:
    """A missing probe file must produce a WARNING row, not silence.

    The function body is extracted in Python and written to a plain file that
    bash then ``source``s directly (OMN-15788). Apple's system ``/bin/bash``
    3.2 does not reliably populate the invoking shell with functions defined
    via ``source <(...)`` process substitution -- the sourced function silently
    fails to land even though the extracted text is byte-identical, so
    ``check_ci_required_contexts`` ends up undefined ("command not found").
    Sourcing a real file instead of a process-substitution fd sidesteps that
    bash-version dependency entirely and behaves identically on bash 3.2 and
    modern bash.
    """
    script = REPORTER.read_text()
    match = re.search(
        r"^check_ci_required_contexts\(\).*?^\}", script, re.MULTILINE | re.DOTALL
    )
    assert match, "check_ci_required_contexts() not found in reporter script"
    func_file = tmp_path / "check_ci_required_contexts.sh"
    func_file.write_text(match.group(0) + "\n")
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"source {func_file}; "
            f"OMNINODE_CI_PROBE_SCRIPT={tmp_path}/nope.py check_ci_required_contexts",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "ci|WARNING|required-contexts" in proc.stdout, (
        f"missing probe produced no WARNING row: {proc.stdout!r} {proc.stderr[-500:]}"
    )
    assert "missing or unreadable" in proc.stdout, proc.stdout


def test_probe_is_governed_by_the_host_sync_manifest() -> None:
    """An un-synced probe is a silently blind detector (the OMN-15525 shape)."""
    manifest = SYNC.read_text()
    assert "omninode-ci-required-context-probe.py" in manifest, (
        "the probe is absent from the host maintenance sync MANIFEST, so nothing "
        "installs it or detects drift -- it would be merged and never deployed"
    )
    assert (
        "scripts/omninode-ci-required-context-probe.py"
        "|/data/maintenance/bin/omninode-ci-required-context-probe.py|0755" in manifest
    ), "probe manifest entry is malformed or not executable-mode"


def test_probe_has_no_third_party_imports() -> None:
    """The host runs this on bare python3 with no venv; stdlib only."""
    source = PROBE.read_text()
    imports = re.findall(r"^(?:from|import)\s+([a-zA-Z_][\w.]*)", source, re.MULTILINE)
    allowed = {
        "__future__",
        "argparse",
        "json",
        "os",
        "sys",
        "urllib",
        "collections",
        "datetime",
        "typing",
        "pathlib",
    }
    unexpected = {i.split(".")[0] for i in imports} - allowed
    assert not unexpected, (
        f"non-stdlib or unvetted imports would break the host run: {unexpected}"
    )
