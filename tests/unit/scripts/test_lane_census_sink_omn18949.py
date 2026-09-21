# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18949 — the lane census reaches a person.

Nothing refused on census drift and nothing reported it. The hourly timer on
the lab host writes a snapshot; the committed snapshot carried five findings at
severity warning; the only surface that read either rendered the prose string
``N drift item(s) — see census`` into a doctrine file, where the next refresh
normalised it away.

The CI gate added by the same ticket answers a different question: whether the
two COMMITTED files agree. CI cannot see the lab host's docker daemon at all,
so the LIVE topology having moved is only visible from the host. These cases
drive the reporter's census probe over a sandboxed snapshot and assert the row
it emits for each outcome.

Every "could not look" outcome owes a row. An unreadable census reported as an
empty green is the failure this whole family exists to remove.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "deploy" / "maintenance" / "omninode-system-slack-report.sh"
CHECKER = REPO_ROOT / "scripts" / "check_lane_census_drift.py"
MANIFEST = REPO_ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"

pytestmark = pytest.mark.unit


def _stub_bin(tmp_path: Path) -> Path:
    """Minimal stub PATH.

    Two binaries, and BOTH are macOS gaps rather than anything about the probe.
    The lab host this script actually runs on is Linux and ships both.

      flock   -- macOS does not ship it at all (memory
                 reference_macos_no_flock_use_fcntl_shim). Without a stub every
                 run here exits early on "another system report is already
                 running" and no report is produced.
      timeout -- likewise absent (it is `gtimeout` from coreutils here), so the
                 probe's bounded checker call exits 127 and every census row
                 reads as "could not evaluate". The stub drops the duration and
                 execs, which is what `timeout` does when nothing times out.

    The richer docker/df/curl stubs the sibling suite builds are not needed:
    these cases assert on the census section, and the other probes are off.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    flock = bin_dir / "flock"
    flock.write_text("#!/usr/bin/env bash\nexit 0\n")
    flock.chmod(0o755)
    timeout_stub = bin_dir / "timeout"
    timeout_stub.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
    timeout_stub.chmod(0o755)
    return bin_dir


def _stage(tmp_path: Path) -> Path:
    """Copy the reporter, inject the stub PATH, repoint its /data paths."""
    staged = tmp_path / "staged-report.sh"
    lines = SCRIPT.read_text().splitlines(keepends=True)
    bin_dir = _stub_bin(tmp_path)
    for index, line in enumerate(lines):
        if line.startswith("PATH="):
            lines.insert(index + 1, f'PATH="{bin_dir}:$PATH"\n')
            break
    else:  # pragma: no cover - the script pins PATH; a miss is a real defect
        raise AssertionError("no PATH assignment found in the reporter")
    text = "".join(lines)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    text = text.replace("/data/maintenance", str(sandbox))
    text = text.replace(
        "/run/omninode-system-slack-report.lock", str(tmp_path / "lock")
    )
    staged.write_text(text)
    return staged


def _snapshot(
    tmp_path: Path,
    *,
    emitted: str,
    findings: list[dict],
    lanes: list[str] | None = None,
) -> Path:
    path = tmp_path / "live-census.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "emitted_at": emitted,
                "lanes_checked": lanes if lanes is not None else ["dev"],
                "drift_count": len(findings),
                "findings": findings,
            }
        )
    )
    return path


def _fresh() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).isoformat()


def _run(tmp_path: Path, **extra: str) -> str:
    env = dict(os.environ)
    env.update(
        {
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
            "OMNINODE_ALERT_STATE_DIR": str(tmp_path / "sandbox" / "state"),
            "OMNINODE_ALERT_LOG_DIR": str(tmp_path / "sandbox" / "logs"),
            "OMNINODE_ALERT_LOCK_FILE": str(tmp_path / "lock"),
            "SLACK_BOT_TOKEN": "test-only-never-real",
            "SLACK_CHANNEL_ID": "C-TEST",
            # Every OTHER probe off: this file is about the census rows, and a
            # probe that reaches GitHub would make these cases network-bound.
            "OMNINODE_CI_PROBE_ENABLED": "0",
            "OMNINODE_RUNNER_TREE_CHECK_ENABLED": "0",
            "OMNINODE_FLEET_PROBE_ENABLED": "0",
            # The census probe is the subject, so it is ON.
            "OMNINODE_CENSUS_PROBE_ENABLED": "1",
            "OMNINODE_CENSUS_MANIFEST": str(MANIFEST),
            "OMNINODE_CENSUS_CHECKER": str(CHECKER),
            # The checker parses YAML, so it needs an interpreter with PyYAML.
            # The host's bare python3 may not have one, which is exactly why the
            # probe makes this overridable and reports the import error in its
            # row rather than a bare exit code.
            "OMNINODE_CENSUS_PYTHON": sys.executable,
        }
    )
    env.update(extra)
    staged = _stage(tmp_path)
    proc = subprocess.run(
        ["bash", str(staged), "--mode", "dry-run"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    out = proc.stdout
    if not out.strip():
        logs = sorted((tmp_path / "sandbox" / "logs").glob("*.log"))
        if logs:
            out = logs[-1].read_text()
    assert out.strip(), f"no report: rc={proc.returncode} stderr={proc.stderr[-2000:]}"
    return out


def _census_section(report: str) -> str:
    assert "*Lane census*" in report, "the digest must carry a lane census section"
    return report.split("*Lane census*", 1)[1].split("*Active issues*", 1)[0]


# ---------------------------------------------------------------------------
# The healthy case, which is what makes every failing case meaningful.
# ---------------------------------------------------------------------------


def test_a_fresh_agreeing_census_reports_ok_and_raises_no_issue(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, emitted=_fresh(), findings=[])
    report = _run(tmp_path, OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap))
    section = _census_section(report)
    assert "(OK)" in section
    assert "no lane drift" in section
    assert "census" not in report.split("*Active issues*", 1)[1]


# ---------------------------------------------------------------------------
# Live drift. The condition that had no reader.
# ---------------------------------------------------------------------------


def test_live_drift_is_reported_and_reaches_the_active_issues_list(
    tmp_path: Path,
) -> None:
    """A row that does not reach the issues list is a row nobody reads.

    Since OMN-18949 armed AC-1 this is CRITICAL rather than WARNING: a
    non-zero drift count refuses, and the reporter inherits that verdict from
    the same checker CI runs instead of grading it a second way. The
    containers named here are deliberately ones the committed manifest does
    NOT declare, so this is plain live drift and not a stale finding.
    """
    snap = _snapshot(
        tmp_path,
        emitted=_fresh(),
        findings=[
            {
                "lane": "dev",
                "kind": "unexpected_container",
                "container": "some-undeclared-container",
            },
            {"lane": "dev", "kind": "unexpected_container", "container": "some-other"},
        ],
    )
    report = _run(tmp_path, OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap))
    section = _census_section(report)
    assert "2 drift item(s)" in section and "dev" in section
    assert "declare it in the lane manifest" in section
    issues = report.split("*Active issues*", 1)[1]
    assert "census" in issues and "CRITICAL" in issues


def test_live_drift_and_a_manifest_contradiction_are_different_rows(
    tmp_path: Path,
) -> None:
    """The two defects have different remedies, so they may not share a key.

    Collapsing them would report "the census contradicts the manifest" for a
    lane that is merely running something undeclared, and the de-duplication
    state for one would suppress the other.
    """
    snap = _snapshot(
        tmp_path,
        emitted=_fresh(),
        findings=[
            {
                "lane": "dev",
                "kind": "unexpected_container",
                "container": "some-undeclared-container",
            }
        ],
    )
    drift_only = _census_section(
        _run(tmp_path, OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap))
    )
    assert "`drift`" in drift_only
    assert "contradicts the committed manifest" not in drift_only


def test_a_live_census_contradicting_the_manifest_is_CRITICAL(tmp_path: Path) -> None:
    """Different from drift: the census and the manifest disagree about what is DECLARED.

    Drift means the host is running something the manifest does not declare,
    which a mutable lane is allowed to do. A contradiction means the two
    descriptions cannot both be true, and that is not a lane state at all.
    """
    manifest = yaml.safe_load(MANIFEST.read_text())
    declared = next(iter(manifest["lanes"]["dev"]["services"]))["name"]
    snap = _snapshot(
        tmp_path,
        emitted=_fresh(),
        findings=[
            {"lane": "dev", "kind": "unexpected_container", "container": declared}
        ],
    )
    report = _run(tmp_path, OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap))
    section = _census_section(report)
    assert "CRITICAL" in section
    assert "contradicts the committed manifest" in section
    assert "stale_unexpected_container" in section


# ---------------------------------------------------------------------------
# Every "could not look" outcome owes a row.
# ---------------------------------------------------------------------------


def test_a_missing_live_snapshot_is_a_row_not_a_silence(tmp_path: Path) -> None:
    """A stopped hourly timer leaves no snapshot, and no row would read as healthy."""
    report = _run(tmp_path, OMNINODE_CENSUS_LIVE_SNAPSHOT=str(tmp_path / "absent.json"))
    section = _census_section(report)
    assert "no live census" in section and "WARNING" in section
    assert "timer may have stopped" in section


def test_a_stale_live_snapshot_is_a_row(tmp_path: Path) -> None:
    """A snapshot that stopped being written keeps reporting the last topology it saw."""
    snap = _snapshot(tmp_path, emitted="2026-01-01T00:00:00+00:00", findings=[])
    report = _run(tmp_path, OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap))
    section = _census_section(report)
    assert "WARNING" in section and "hourly timer has stopped writing" in section


def test_a_snapshot_with_no_readable_timestamp_is_a_row(tmp_path: Path) -> None:
    bad = tmp_path / "live-census.json"
    bad.write_text(json.dumps({"findings": [], "lanes_checked": ["dev"]}))
    report = _run(tmp_path, OMNINODE_CENSUS_LIVE_SNAPSHOT=str(bad))
    section = _census_section(report)
    assert "no readable emitted_at" in section and "WARNING" in section


def test_an_unreadable_manifest_is_a_row(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, emitted=_fresh(), findings=[])
    report = _run(
        tmp_path,
        OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap),
        OMNINODE_CENSUS_MANIFEST=str(tmp_path / "absent.yaml"),
    )
    section = _census_section(report)
    assert "manifest unreadable" in section and "WARNING" in section


def test_a_missing_checker_is_a_row(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, emitted=_fresh(), findings=[])
    report = _run(
        tmp_path,
        OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap),
        OMNINODE_CENSUS_CHECKER=str(tmp_path / "absent.py"),
    )
    section = _census_section(report)
    assert "drift checker missing or unreadable" in section and "WARNING" in section


def test_the_section_says_so_when_the_probe_did_not_run(tmp_path: Path) -> None:
    """A section printing nothing is indistinguishable from one with nothing wrong."""
    report = _run(tmp_path, OMNINODE_CENSUS_PROBE_ENABLED="0")
    section = _census_section(report)
    assert "No lane census rows this tick" in section


# ---------------------------------------------------------------------------
# The "more than one tick" requirement, and where it is NOT implemented.
# ---------------------------------------------------------------------------


def test_the_probe_implements_no_private_tick_counter() -> None:
    """The brief asks for drift held over more than one tick.

    The alert state machine already does exactly that: CONFIRM_TICKS requires a
    key to hold the same status for two consecutive ticks before it pages, and
    CLEAR_TICKS absorbs the flap on the way back. A second counter inside the
    probe would be a second answer to a question already answered, and the two
    would disagree the first time either default changed.
    """
    text = SCRIPT.read_text()
    probe = text.split("check_census_drift() {", 1)[1].split("\n}", 1)[0]
    for forbidden in ("CONFIRM_TICKS", "CLEAR_TICKS", "RENOTIFY_SECONDS"):
        assert forbidden not in probe, (
            f"the census probe must not read or reimplement {forbidden}; the "
            "shared state machine owns the hold-for-N-ticks behaviour"
        )
    assert "CONFIRM_TICKS=${OMNINODE_ALERT_CONFIRM_TICKS:-2}" in text, (
        "the shared hysteresis must still default to two ticks, which is what "
        "makes 'more than one tick' true for the census rows without any "
        "census-specific code"
    )


def test_the_census_rows_are_keyed_for_the_shared_dedup_machinery() -> None:
    """row_key() builds `census|<key>`, so each key pages once and re-arms alone."""
    text = SCRIPT.read_text()
    probe = text.split("check_census_drift() {", 1)[1].split("\n}", 1)[0]
    keys = {"snapshot", "manifest", "drift"}
    for key in keys:
        assert "'census|" in probe
        assert f"|{key}|" in probe, f"the probe must emit a row keyed '{key}'"


def test_a_checker_failure_row_names_the_REASON_not_just_an_exit_code(
    tmp_path: Path,
) -> None:
    """Rule 16: a suppressed error returns nothing and reads like a clean census.

    Found while building this: the checker parses YAML, and pointed at an
    interpreter without PyYAML it exited 1 with an empty result. With stderr
    discarded the row said "exit 1", which is true and useless. The row now
    carries the interpreter's own last line, so the reader is told the cause.
    """
    snap = _snapshot(tmp_path, emitted=_fresh(), findings=[])
    stub = tmp_path / "bin" / "python-no-yaml"
    stub.parent.mkdir(parents=True, exist_ok=True)
    # The probe calls this interpreter twice: once as `python - <snapshot>` for
    # the age read, which needs only the standard library and must still work,
    # and once as `python <checker> --manifest ...`, which needs PyYAML. The
    # stub answers the first and fails the second, which is exactly the shape
    # of a host whose bare python3 lacks the dependency.
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-" ]; then cat >/dev/null; echo 0; exit 0; fi\n'
        "echo \"ModuleNotFoundError: No module named 'yaml'\" >&2\n"
        "exit 1\n"
    )
    stub.chmod(0o755)
    report = _run(
        tmp_path,
        OMNINODE_CENSUS_LIVE_SNAPSHOT=str(snap),
        OMNINODE_CENSUS_PYTHON=str(stub),
    )
    section = _census_section(report)
    assert "could not evaluate" in section and "WARNING" in section
    assert "No module named 'yaml'" in section, (
        "the row must carry the interpreter's own error; 'exit 1' alone is not "
        "actionable and is what a discarded stderr produces"
    )


def test_the_probe_does_not_discard_stderr() -> None:
    """The source-level ratchet for the case above."""
    text = SCRIPT.read_text()
    probe = text.split("check_census_drift() {", 1)[1].split("\n}", 1)[0]
    checker_call = probe.split('--manifest "$manifest" --snapshot "$live" --json', 1)[
        1
    ].split("\n", 1)[0]
    assert "2>/dev/null" not in checker_call, (
        "the drift checker's stderr must be captured, not discarded -- a "
        "suppressed error returns an empty result indistinguishable from a "
        "clean census (CLAUDE.md rule 16)"
    )
