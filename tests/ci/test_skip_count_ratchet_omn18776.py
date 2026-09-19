# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18776 — one test per acceptance-criterion falsifier for the skip-count ratchet.

Epic OMN-18775 measured five per-repo skip counts that were byte-identical across
three consecutive CI runs each. A count that never moves is a fixed set of tests
that never runs, and nothing on the fleet notices when that set grows — which is
how 91 PostgreSQL-16 cases were collected and skipped on every run for 49 days.

Each test below is named for the falsifier it refuses. Deleting one deletes the
proof of that acceptance criterion.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "skip_count_ratchet.py"
BASELINE = REPO_ROOT / "config" / "skip_count_baseline.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "skip_count_ratchet"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_SUMMARY_GATE = REPO_ROOT / "scripts" / "ci" / "ci_summary_gate.py"

SUITE = "omnibase_infra/test-parallel"
JOB_DISPLAY_NAME = "Skip Count Ratchet (OMN-18776)"

pytestmark = pytest.mark.unit


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def _junit(cases: list[tuple[str, bool]], collected: int) -> str:
    """Render a JUnit report. `cases` is (node_id, skipped) pairs."""
    body = []
    for node_id, skipped in cases:
        classname, _, name = node_id.rpartition("::")
        inner = "<skipped message='synthetic'/>" if skipped else ""
        body.append(
            f'<testcase classname="{classname}" name="{name}">{inner}</testcase>'
        )
    joined = "".join(body)
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest" errors="0" failures="0" '
        f'skipped="{sum(1 for _, s in cases if s)}" tests="{collected}" time="1.0">'
        f"{joined}</testsuite></testsuites>"
    )


def _baseline_entry() -> dict[str, object]:
    loaded = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    entry = loaded["suites"][SUITE]
    assert isinstance(entry, dict)
    return entry


def _baseline_ids() -> list[str]:
    ids = _baseline_entry()["node_ids"]
    assert isinstance(ids, list)
    return [str(i) for i in ids]


# ---------------------------------------------------------------- AC1


def test_ac1_shipped_baseline_records_provenance_for_every_entry() -> None:
    """Falsifier: a baseline entry with no recorded provenance."""
    loaded = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    suites = loaded["suites"]
    assert suites, "positive control: the shipped baseline declares at least one suite"
    for key, entry in suites.items():
        provenance = entry.get("provenance")
        assert provenance, f"{key}: no provenance block"
        for field in ("measured_at", "measurement_command", "source_runs"):
            assert provenance.get(field), (
                f"{key}: provenance.{field} is missing or empty"
            )


def test_ac1_entry_missing_provenance_is_refused(tmp_path: Path) -> None:
    """Falsifier control: the validator must actually reject a bare entry."""
    bare = tmp_path / "baseline.yaml"
    bare.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "suites": {
                    SUITE: {
                        "repo": "omnibase_infra",
                        "job": "Tests",
                        "mode": "count",
                        "max_skips": 1,
                        "baseline_collected": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    junit = tmp_path / "j.xml"
    junit.write_text(_junit([("a.b::test_c", True)], 1), encoding="utf-8")
    result = _run("--baseline", str(bare), "--suite", SUITE, "--junit", str(junit))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "provenance" in (result.stdout + result.stderr)


# ---------------------------------------------------------------- AC2


def test_ac2_added_environmental_skip_fails_naming_repo_job_and_delta(
    tmp_path: Path,
) -> None:
    """Falsifier: a deliberately added environmentally-skipped test produces a green run."""
    entry = _baseline_entry()
    cases: list[tuple[str, bool]] = [(i, True) for i in _baseline_ids()]
    cases.append(("tests.ci.test_synthetic_omn18776::test_needs_absent_service", True))
    junit = tmp_path / "j.xml"
    junit.write_text(_junit(cases, int(entry["baseline_collected"])), encoding="utf-8")

    result = _run("--baseline", str(BASELINE), "--suite", SUITE, "--junit", str(junit))
    out = result.stdout + result.stderr
    assert result.returncode == 1, out
    assert "omnibase_infra" in out
    assert str(entry["job"]) in out
    assert "test_needs_absent_service" in out
    assert "+1" in out, "the failure must name the delta"


def test_ac2_positive_control_baseline_set_alone_passes(tmp_path: Path) -> None:
    """Positive control for the test above: without the added skip the run is green."""
    entry = _baseline_entry()
    cases = [(i, True) for i in _baseline_ids()]
    junit = tmp_path / "j.xml"
    junit.write_text(_junit(cases, int(entry["baseline_collected"])), encoding="utf-8")
    result = _run("--baseline", str(BASELINE), "--suite", SUITE, "--junit", str(junit))
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------- AC3


def test_ac3_removed_skip_passes_and_prints_a_ratchet_candidate(
    tmp_path: Path,
) -> None:
    """Falsifier: removing a skipped test turns the run red."""
    entry = _baseline_entry()
    ids = _baseline_ids()
    cases = [(i, True) for i in ids[:-1]]
    junit = tmp_path / "j.xml"
    junit.write_text(_junit(cases, int(entry["baseline_collected"])), encoding="utf-8")

    result = _run("--baseline", str(BASELINE), "--suite", SUITE, "--junit", str(junit))
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "RATCHET CANDIDATE" in out
    assert str(len(ids) - 1) in out, "the new lower number must be printed"


# ---------------------------------------------------------------- AC4


def test_ac4_ratchet_job_carries_no_continue_on_error() -> None:
    """Falsifier: `continue-on-error` appears anywhere in the new job."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert JOB_DISPLAY_NAME in text, "positive control: the job is declared in ci.yml"
    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["skip-count-ratchet"]
    assert "continue-on-error" not in job
    for step in job["steps"]:
        assert "continue-on-error" not in step, step.get("name")

    # The falsifier is a grep, so the raw block must not even mention the
    # setting in prose -- a comment naming it reads as a hit to the same probe
    # that is supposed to prove its absence.
    block = text.split("\n  skip-count-ratchet:\n", 1)[1].split("\n  # ====", 1)[0]
    assert "skip_count_ratchet.py" in block, "positive control: the block was located"
    assert "continue-on-error" not in block


def test_ac4_ratchet_job_is_registered_under_the_ci_summary_umbrella() -> None:
    """Falsifier: the context appears in neither enforcement surface."""
    gate = CI_SUMMARY_GATE.read_text(encoding="utf-8")
    assert f'"{JOB_DISPLAY_NAME}"' in gate, (
        "omnibase_infra requires exactly one context (CI Summary); an unregistered "
        "job that is skipped or absent yields SUCCESS"
    )
    strict_block = gate.split("STRICT_GATE_JOBS: tuple[str, ...] = (", 1)[1].split(
        "\n)", 1
    )[0]
    assert f'"{JOB_DISPLAY_NAME}"' in strict_block


def test_ac4_ratchet_job_is_unconditional() -> None:
    """A registered job that can legitimately skip wedges the umbrella; ours uses always()."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["skip-count-ratchet"]
    assert str(job.get("if", "")).strip() == "always()"


# ---------------------------------------------------------------- AC5


@pytest.mark.parametrize(
    "fixture_name",
    ["narrowed-run-35404351173.xml", "narrowed-run-35400977169.xml"],
)
def test_ac5_selector_narrowed_real_runs_do_not_false_fail(fixture_name: str) -> None:
    """Falsifier: a run fails on a count the impacted-test selector explains.

    Both fixtures are the real skipped-case sets of two live selector-narrowed
    omnibase_infra runs (45 and 95 unique skips against a 547 baseline).
    """
    result = _run(
        "--baseline",
        str(BASELINE),
        "--suite",
        SUITE,
        "--junit",
        str(FIXTURES / fixture_name),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "narrowed selection" in out.lower()
    assert "RATCHET CANDIDATE" not in out, (
        "a narrowed run's lower count is the selector, not a ratchet opportunity"
    )


def test_ac5_positive_control_one_new_id_in_a_narrowed_run_still_fails(
    tmp_path: Path,
) -> None:
    """The zero above is not a broken probe: one unknown id in the same shape is red."""
    original = (FIXTURES / "narrowed-run-35404351173.xml").read_text(encoding="utf-8")
    injected = original.replace(
        "</testsuite>",
        '<testcase classname="tests.ci.test_control_omn18776" '
        'name="test_positive_control"><skipped message="synthetic"/></testcase>'
        "</testsuite>",
    )
    junit = tmp_path / "j.xml"
    junit.write_text(injected, encoding="utf-8")
    result = _run("--baseline", str(BASELINE), "--suite", SUITE, "--junit", str(junit))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "test_positive_control" in result.stdout + result.stderr


# ---------------------------------------------------------------- AC6


def test_ac6_parser_declares_no_override_option() -> None:
    """Falsifier: the argument parser declares any skip, force or baseline-override option.

    Lowering a baseline is an edit to the baseline file, reviewed like any other
    diff. A command-line lever would make it a decision one lane takes alone.
    """
    result = _run("--help")
    assert result.returncode == 0, result.stdout + result.stderr
    help_text = result.stdout.lower()
    forbidden = (
        "--force",
        "--skip",
        "--allow",
        "--ignore",
        "--override",
        "--no-fail",
        "--update-baseline",
        "--set-baseline",
        "--write-baseline",
        "--max-skips",
        "--tolerance",
        "--threshold",
        "--warn-only",
        "--advisory",
    )
    for option in forbidden:
        assert option not in help_text, f"{option} would make the ratchet optional"
    assert "--junit" in help_text, "positive control: the parser's help was read"


def test_ac6_no_environment_variable_lowers_the_verdict(tmp_path: Path) -> None:
    """An env var escape hatch is the same hole wearing a different hat."""
    entry = _baseline_entry()
    cases = [(i, True) for i in _baseline_ids()]
    cases.append(("tests.ci.test_synthetic_omn18776::test_env_escape", True))
    junit = tmp_path / "j.xml"
    junit.write_text(_junit(cases, int(entry["baseline_collected"])), encoding="utf-8")

    env_names = (
        "SKIP_COUNT_RATCHET",
        "SKIP_COUNT_RATCHET_FORCE",
        "SKIP_COUNT_RATCHET_ADVISORY",
        "ENABLE_SKIP_COUNT_RATCHET",
        "SKIP_RATCHET_OVERRIDE",
    )
    source = SCRIPT.read_text(encoding="utf-8")
    assert "os.environ" not in source and "getenv" not in source, (
        "the gate must read no environment variable at all"
    )
    for name in env_names:
        assert name not in source


# ---------------------------------------------------------------- fail-closed


def test_missing_junit_input_fails_closed(tmp_path: Path) -> None:
    result = _run(
        "--baseline",
        str(BASELINE),
        "--suite",
        SUITE,
        "--junit",
        str(tmp_path / "nope.xml"),
    )
    assert result.returncode == 2, result.stdout + result.stderr


def test_unparsable_junit_fails_closed(tmp_path: Path) -> None:
    junit = tmp_path / "j.xml"
    junit.write_text("this is not xml", encoding="utf-8")
    result = _run("--baseline", str(BASELINE), "--suite", SUITE, "--junit", str(junit))
    assert result.returncode == 2, result.stdout + result.stderr


def test_unknown_suite_fails_closed(tmp_path: Path) -> None:
    junit = tmp_path / "j.xml"
    junit.write_text(_junit([("a.b::test_c", True)], 1), encoding="utf-8")
    result = _run(
        "--baseline", str(BASELINE), "--suite", "nope/nope", "--junit", str(junit)
    )
    assert result.returncode == 2, result.stdout + result.stderr


def test_selftest_mode_passes_and_is_what_the_pre_commit_hook_runs() -> None:
    result = _run("--selftest")
    assert result.returncode == 0, result.stdout + result.stderr
    hook_config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "skip_count_ratchet.py --selftest" in hook_config
