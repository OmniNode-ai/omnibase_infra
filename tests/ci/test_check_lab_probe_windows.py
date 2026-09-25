# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The lab probe-window file and its drift check (OMN-19412, seam L0.3).

``config/lab_probe_windows.yaml`` is the one machine-readable list of the
scheduled probes that read a lab lane. The reconcile dispatcher (OMN-19420) and
the ``runtime-train`` skill (OMN-19424) read it to keep a lane mutation out of
the gap before a probe. It is only safe to read if it cannot drift from the
workflow files it describes, so ``scripts/ci/check_lab_probe_windows.py``
re-derives every cron, lane and maximum duration from those files and refuses
any disagreement, and refuses a scheduled probe that has no entry.

Every test here is offline. The synthetic tests build a fake repo root in
``tmp_path`` (the probe-workflow side of the seam); the committed-file tests
read this repository's real workflows, which is the half of the check a laptop
can run without the other two repositories.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.ci import check_lab_probe_windows as plw

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOW_FILE = REPO_ROOT / "config" / "lab_probe_windows.yaml"

# The ten probes the lab release-sync plan's section 4 E names (T0.3 AC1).
PLAN_SECTION_4E: dict[str, tuple[str, str]] = {
    "C15": ("omnibase_infra", ".github/workflows/chain-canary.yml"),
    "C16": (
        "omnibase_infra",
        ".github/workflows/chain-canary-c16-receipt-identity.yml",
    ),
    "C11": ("omnibase_infra", ".github/workflows/chain-canary-c11-negative-paths.yml"),
    "C12": (
        "omnibase_infra",
        ".github/workflows/chain-canary-c12-provider-catalogue.yml",
    ),
    "C13": ("omninode_infra", ".github/workflows/c13-customer-local-delegation.yml"),
    "C14": ("omninode_infra", ".github/workflows/c14-customer-local-routing.yml"),
    "C29": ("omninode_infra", ".github/workflows/c29-customer-byo-key-delegation.yml"),
    "C17": ("omninode_infra", ".github/workflows/m4-c17-customer-surface-verdict.yml"),
    "D11": ("omnimarket", ".github/workflows/delegation-regression-nightly.yml"),
    "release-train": ("omnibase_infra", ".github/workflows/release-train-nightly.yml"),
}

C16_WORKFLOW = textwrap.dedent(
    """\
    name: Chain Canary C16 receipt identity
    on:
      schedule:
        - cron: '29 3,9,15,21 * * *'
      workflow_dispatch:
    jobs:
      c16:
        name: C16 receipt identity (dev lane)
        runs-on: [self-hosted, omnibase-verify, host-201]
        timeout-minutes: 15
        steps:
          - run: echo probe
    """
)

D11_WORKFLOW = textwrap.dedent(
    """\
    name: Delegation Regression (nightly)
    run-name: Delegation Regression (nightly) lane=${{ github.event.inputs.lane || 'stability-test' }}
    on:
      schedule:
        - cron: '30 7 * * *'
      workflow_dispatch:
        inputs:
          lane:
            default: stability-test
    jobs:
      regress:
        runs-on: [self-hosted, omnimarket-ci]
        timeout-minutes: 40
        steps:
          - run: echo probe
    """
)

C13_WORKFLOW = textwrap.dedent(
    """\
    name: C13 customer-local delegation
    on:
      schedule:
        - cron: '29 4,16 * * *'
    jobs:
      c13:
        name: C13 customer-local delegation (omnipc2)
        runs-on: [omnipc2-customer]
        timeout-minutes: 45
        steps:
          - run: echo probe
    """
)

TRAIN_WORKFLOW = textwrap.dedent(
    """\
    name: Release Train (nightly)
    on:
      schedule:
        - cron: "11 7 * * *"
    jobs:
      decide:
        runs-on: ubuntu-latest
        timeout-minutes: 30
        steps:
          - run: echo decide
      cut:
        needs: decide
        runs-on: ubuntu-latest
        timeout-minutes: 20
        steps:
          - run: echo cut
    """
)

NOT_A_PROBE = textwrap.dedent(
    """\
    name: branch protection audit
    on:
      schedule:
        - cron: "23 */4 * * *"
    jobs:
      audit:
        runs-on: ubuntu-latest
        timeout-minutes: 5
        steps:
          - run: echo audit
    """
)

WINDOWS_YAML = textwrap.dedent(
    """\
    schema_version: 1
    probes:
      - id: C16
        name: receipt identity
        repo: omnibase_infra
        workflow: .github/workflows/c16.yml
        cron: ['29 3,9,15,21 * * *']
        lane: dev
        lane_source: job_name
        max_duration_minutes: 15
      - id: release-train
        name: release train
        repo: omnibase_infra
        workflow: .github/workflows/train.yml
        cron: ['11 7 * * *']
        lane: none
        lane_source: none
        reads: the compose-dev lab-pass receipts named by config/release_train_policy.yaml
        max_duration_minutes: 50
      - id: C13
        name: customer local
        repo: omninode_infra
        workflow: .github/workflows/c13.yml
        cron: ['29 4,16 * * *']
        lane: omnipc2-customer
        lane_source: runs_on
        max_duration_minutes: 45
      - id: D11
        name: delegation regression
        repo: omnimarket
        workflow: .github/workflows/d11.yml
        cron: ['30 7 * * *']
        lane: stability-test
        lane_source: run_name
        max_duration_minutes: 40
    """
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, Path]:
    """Three fake repo roots whose workflows agree with ``WINDOWS_YAML``."""
    infra = tmp_path / "omnibase_infra"
    node = tmp_path / "omninode_infra"
    market = tmp_path / "omnimarket"
    _write(infra / ".github/workflows/c16.yml", C16_WORKFLOW)
    _write(infra / ".github/workflows/train.yml", TRAIN_WORKFLOW)
    _write(infra / ".github/workflows/audit.yml", NOT_A_PROBE)
    _write(node / ".github/workflows/c13.yml", C13_WORKFLOW)
    _write(market / ".github/workflows/d11.yml", D11_WORKFLOW)
    return {"omnibase_infra": infra, "omninode_infra": node, "omnimarket": market}


@pytest.fixture
def window_file(tmp_path: Path) -> Path:
    return _write(tmp_path / "lab_probe_windows.yaml", WINDOWS_YAML)


def _errors(window_file: Path, roots: dict[str, Path]) -> list[str]:
    return plw.check(plw.load_windows(window_file), roots)


# --------------------------------------------------------------------------- #
# The RED test T0.3 names: a mutated C16 cron is refused, naming both values.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_mutated_cron_is_refused_naming_both_values(
    window_file: Path, roots: dict[str, Path]
) -> None:
    _write(
        roots["omnibase_infra"] / ".github/workflows/c16.yml",
        C16_WORKFLOW.replace("'29 3,9,15,21 * * *'", "'30 3,9,15,21 * * *'"),
    )
    errors = _errors(window_file, roots)
    assert len(errors) == 1, errors
    assert "C16" in errors[0]
    assert "29 3,9,15,21 * * *" in errors[0]
    assert "30 3,9,15,21 * * *" in errors[0]


@pytest.mark.unit
def test_mutated_cron_in_the_window_file_is_refused(
    tmp_path: Path, roots: dict[str, Path]
) -> None:
    mutated = _write(
        tmp_path / "w.yaml",
        WINDOWS_YAML.replace("'29 3,9,15,21 * * *'", "'29 3,9,15 * * *'"),
    )
    errors = _errors(mutated, roots)
    assert any(
        "C16" in e and "29 3,9,15 * * *" in e and "29 3,9,15,21 * * *" in e
        for e in errors
    ), errors


@pytest.mark.unit
def test_agreeing_file_and_workflows_pass(
    window_file: Path, roots: dict[str, Path]
) -> None:
    assert _errors(window_file, roots) == []


@pytest.mark.unit
def test_cli_exits_zero_when_clean_and_one_on_drift(
    window_file: Path, roots: dict[str, Path]
) -> None:
    argv = ["--windows", str(window_file)] + [
        f"--root={name}={path}" for name, path in roots.items()
    ]
    assert plw.main(argv) == 0
    _write(
        roots["omnibase_infra"] / ".github/workflows/c16.yml",
        C16_WORKFLOW.replace("29 3", "28 3"),
    )
    assert plw.main(argv) == 1


@pytest.mark.unit
def test_cli_refuses_a_missing_root(window_file: Path, roots: dict[str, Path]) -> None:
    argv = [
        "--windows",
        str(window_file),
        f"--root=omnibase_infra={roots['omnibase_infra']}",
    ]
    assert plw.main(argv) == 2


# --------------------------------------------------------------------------- #
# Unlisted probes (T0.3 AC2): a scheduled probe with no entry is refused.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_unlisted_probe_cron_is_refused(
    window_file: Path, roots: dict[str, Path]
) -> None:
    _write(
        roots["omnibase_infra"] / ".github/workflows/new-probe.yml",
        C16_WORKFLOW.replace("C16 receipt identity (dev lane)", "New probe (dev lane)"),
    )
    errors = _errors(window_file, roots)
    assert len(errors) == 1, errors
    assert "unlisted" in errors[0]
    assert "omnibase_infra:.github/workflows/new-probe.yml" in errors[0]


@pytest.mark.unit
def test_unlisted_run_name_lane_probe_is_refused(
    window_file: Path, roots: dict[str, Path]
) -> None:
    _write(
        roots["omnimarket"] / ".github/workflows/d12.yml",
        D11_WORKFLOW.replace("30 7", "45 8"),
    )
    errors = _errors(window_file, roots)
    assert any(
        "unlisted" in e and "omnimarket:.github/workflows/d12.yml" in e for e in errors
    ), errors


@pytest.mark.unit
def test_unlisted_customer_machine_probe_is_refused(
    window_file: Path, roots: dict[str, Path]
) -> None:
    _write(
        roots["omninode_infra"] / ".github/workflows/c30.yml",
        C13_WORKFLOW.replace("29 4", "11 4"),
    )
    errors = _errors(window_file, roots)
    assert any(
        "unlisted" in e and "omninode_infra:.github/workflows/c30.yml" in e
        for e in errors
    ), errors


@pytest.mark.unit
def test_a_probe_cron_added_to_a_listed_workflow_is_refused(
    window_file: Path, roots: dict[str, Path]
) -> None:
    _write(
        roots["omnibase_infra"] / ".github/workflows/c16.yml",
        C16_WORKFLOW.replace(
            "    - cron: '29 3,9,15,21 * * *'\n",
            "    - cron: '29 3,9,15,21 * * *'\n    - cron: '5 12 * * *'\n",
        ),
    )
    errors = _errors(window_file, roots)
    assert any("C16" in e and "5 12 * * *" in e for e in errors), errors


@pytest.mark.unit
def test_an_unscheduled_lane_workflow_is_not_a_probe(
    window_file: Path, roots: dict[str, Path]
) -> None:
    text = C16_WORKFLOW.replace("  schedule:\n    - cron: '29 3,9,15,21 * * *'\n", "")
    _write(roots["omnibase_infra"] / ".github/workflows/manual.yml", text)
    assert _errors(window_file, roots) == []


# --------------------------------------------------------------------------- #
# Lane and duration are re-derived, not trusted.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_lane_mismatch_is_refused(window_file: Path, roots: dict[str, Path]) -> None:
    _write(
        roots["omnimarket"] / ".github/workflows/d11.yml",
        D11_WORKFLOW.replace("|| 'stability-test'", "|| 'dev'"),
    )
    errors = _errors(window_file, roots)
    assert any(
        "D11" in e and "lane" in e and "stability-test" in e and "'dev'" in e
        for e in errors
    ), errors


@pytest.mark.unit
def test_job_name_lane_mismatch_is_refused(
    window_file: Path, roots: dict[str, Path]
) -> None:
    _write(
        roots["omnibase_infra"] / ".github/workflows/c16.yml",
        C16_WORKFLOW.replace("(dev lane)", "(stability-test lane)"),
    )
    errors = _errors(window_file, roots)
    assert any("C16" in e and "stability-test" in e for e in errors), errors


@pytest.mark.unit
def test_none_lane_refused_when_the_workflow_names_a_lane(
    window_file: Path, roots: dict[str, Path]
) -> None:
    _write(
        roots["omnibase_infra"] / ".github/workflows/train.yml",
        TRAIN_WORKFLOW.replace(
            "  decide:\n", "  decide:\n    name: Decide (dev lane)\n"
        ),
    )
    errors = _errors(window_file, roots)
    assert any("release-train" in e and "dev" in e for e in errors), errors


@pytest.mark.unit
def test_duration_is_the_longest_needs_chain(
    window_file: Path, roots: dict[str, Path]
) -> None:
    # decide (30) -> cut (20) is 50, which the fixture declares.
    assert _errors(window_file, roots) == []
    _write(
        roots["omnibase_infra"] / ".github/workflows/train.yml",
        TRAIN_WORKFLOW.replace("timeout-minutes: 20", "timeout-minutes: 25"),
    )
    errors = _errors(window_file, roots)
    assert any("release-train" in e and "50" in e and "55" in e for e in errors), errors


@pytest.mark.unit
def test_missing_timeout_counts_as_the_github_default(tmp_path: Path) -> None:
    wf = _write(
        tmp_path / "w.yml", C16_WORKFLOW.replace("    timeout-minutes: 15\n", "")
    )
    assert plw.max_duration_minutes(plw.read_workflow(wf)) == 360


@pytest.mark.unit
def test_missing_workflow_is_refused(window_file: Path, roots: dict[str, Path]) -> None:
    (roots["omnimarket"] / ".github/workflows/d11.yml").unlink()
    errors = _errors(window_file, roots)
    assert any("D11" in e and "not found" in e for e in errors), errors


# --------------------------------------------------------------------------- #
# The file's own shape.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        (("lane_source: job_name", "lane_source: guess"), "lane_source"),
        (
            ("max_duration_minutes: 15", "max_duration_minutes: 0"),
            "max_duration_minutes",
        ),
        (("    lane: dev\n", ""), "lane"),
        (("schema_version: 1", "schema_version: 2"), "schema_version"),
        (("repo: omnimarket", "repo: omniweb"), "repo"),
        (("cron: ['29 4,16 * * *']", "cron: ['29 4,16 1 * *']"), "cron"),
    ],
)
def test_malformed_window_file_is_refused(
    tmp_path: Path, mutation: tuple[str, str], needle: str
) -> None:
    old, new = mutation
    assert old in WINDOWS_YAML
    bad = _write(tmp_path / "w.yaml", WINDOWS_YAML.replace(old, new, 1))
    with pytest.raises(plw.WindowFileError, match=needle):
        plw.load_windows(bad)


@pytest.mark.unit
def test_duplicate_workflow_entry_is_refused(tmp_path: Path) -> None:
    dup = WINDOWS_YAML.replace(
        "workflow: .github/workflows/train.yml", "workflow: .github/workflows/c16.yml"
    )
    with pytest.raises(plw.WindowFileError, match="more than one entry"):
        plw.load_windows(_write(tmp_path / "w.yaml", dup))


@pytest.mark.unit
def test_none_lane_requires_a_reads_statement(tmp_path: Path) -> None:
    bad = WINDOWS_YAML.replace(
        "    reads: the compose-dev lab-pass receipts named by config/release_train_policy.yaml\n",
        "",
    )
    with pytest.raises(plw.WindowFileError, match="reads"):
        plw.load_windows(_write(tmp_path / "w.yaml", bad))


# --------------------------------------------------------------------------- #
# The consumer side of the seam (the dispatcher and runtime-train read this).
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_occurrences_expand_the_cron_in_utc(window_file: Path) -> None:
    windows = {w.id: w for w in plw.load_windows(window_file)}
    day = datetime(2026, 9, 24, tzinfo=UTC)
    starts = windows["C16"].occurrences(day, day + timedelta(days=1))
    assert [s.strftime("%H:%M") for s in starts] == ["03:29", "09:29", "15:29", "21:29"]


@pytest.mark.unit
def test_busy_intervals_cover_start_to_start_plus_max_duration(
    window_file: Path,
) -> None:
    windows = plw.load_windows(window_file)
    start = datetime(2026, 9, 24, 4, 0, tzinfo=UTC)
    busy = plw.busy_intervals(windows, start, start + timedelta(hours=2))
    assert (
        "C13",
        datetime(2026, 9, 24, 4, 29, tzinfo=UTC),
        datetime(2026, 9, 24, 5, 14, tzinfo=UTC),
    ) in busy


# --------------------------------------------------------------------------- #
# The committed file, against this repository's real workflows.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_committed_file_has_exactly_one_entry_per_plan_probe() -> None:
    windows = plw.load_windows(WINDOW_FILE)
    by_id = {w.id: w for w in windows}
    for probe_id, (repo, workflow) in PLAN_SECTION_4E.items():
        assert probe_id in by_id, probe_id
        assert (by_id[probe_id].repo, by_id[probe_id].workflow) == (repo, workflow)
    keys = [(w.repo, w.workflow) for w in windows]
    assert len(keys) == len(set(keys))


@pytest.mark.unit
def test_committed_file_agrees_with_this_repos_workflows() -> None:
    """The in-repo half of the CI check: every omnibase_infra entry, and no
    unlisted omnibase_infra probe. The other two repositories are read by the
    CI job at their default branch, not here."""
    windows = [w for w in plw.load_windows(WINDOW_FILE) if w.repo == "omnibase_infra"]
    assert windows
    assert plw.check(windows, {"omnibase_infra": REPO_ROOT}) == []
