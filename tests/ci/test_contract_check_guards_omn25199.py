# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25199: the dod checks of the OMN-24976 fix fail when their own guard fails.

The review of the fix found guards that could not fail: commands joined with
``;`` under a shell with no ``-e`` (only the last command decided), ``! grep``
(a grep error, exit 2, negated to success), and a pytest check that passed when
pytest never ran. Each test here runs the real check command from the contract
in a place where it must fail (an empty directory, a missing file, a fake
``uv`` that prints no summary) and in the checkout where it must pass.
"""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from _pytest.mark.expression import Expression

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.lab_pass_receipt import EnumLabLane
from scripts.ci.lab_pass_receipt import main as receipt_main

CONTRACTS = ("OMN-24976", "OMN-25101", "OMN-25184", "OMN-25199", "OMN-25205")
BASE_SHA = "75969ce9e"


def _contract(ticket: str) -> dict[str, Any]:
    loaded = yaml.safe_load((REPO_ROOT / "contracts" / f"{ticket}.yaml").read_text())
    assert isinstance(loaded, dict)
    return loaded


def _checks(ticket: str) -> dict[str, list[dict[str, Any]]]:
    return {entry["id"]: entry["checks"] for entry in _contract(ticket)["dod_evidence"]}


def _command(ticket: str, evidence_id: str, index: int) -> str:
    value = _checks(ticket)[evidence_id][index]["check_value"]
    assert isinstance(value, str)
    return value


def _sh(
    command: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(env if env is not None else os.environ),
        timeout=120,
        check=False,
    )


# The cheap, deterministic guard checks of the review contracts: every one of
# them must pass in the checkout and fail in a directory that is not. The two
# tests-wide greps are not here: they read the whole of tests/, so a later test
# outside tests/ci naming check_lane_sync would redden this file in CI (OMN-25205
# finding 10). They are pinned against a synthetic tree further down instead.
GUARDS = {
    "25184-ac2-live-contact-not-deselected": (
        "OMN-25184",
        "dod-omn25184-ac2-unskipped-not-deselected",
        1,
    ),
    "25184-ac3-no-omni-home-cwd": (
        "OMN-25184",
        "dod-omn25184-ac3-cwd-dot-and-nine-failed",
        0,
    ),
    "25184-ac3-cwd-dot-count": (
        "OMN-25184",
        "dod-omn25184-ac3-cwd-dot-and-nine-failed",
        1,
    ),
    "25184-ac3-nine-failed-pin": (
        "OMN-25184",
        "dod-omn25184-ac3-cwd-dot-and-nine-failed",
        2,
    ),
    "25199-ac2-no-semicolon-chain": (
        "OMN-25199",
        "dod-omn25199-ac2-guards-joined",
        0,
    ),
    "25199-ac3-no-negated-grep": (
        "OMN-25199",
        "dod-omn25199-ac3-negated-greps-and-markers",
        0,
    ),
}

# The two tests-wide greps, with the contract that holds each.
TESTS_WIDE = {
    "25184-ac4": ("OMN-25184", "dod-omn25184-ac4-tests-wide-grep", 0),
    "25101-ac3": ("OMN-25101", "dod-omn25101-ac3-no-other-pin-and-full-suite", 0),
}


@pytest.mark.parametrize("name", sorted(GUARDS))
def test_guard_passes_in_the_checkout_and_fails_in_a_wrong_cwd(
    name: str, tmp_path: Path
) -> None:
    ticket, evidence_id, index = GUARDS[name]
    command = _command(ticket, evidence_id, index)
    good = _sh(command, REPO_ROOT)
    assert good.returncode == 0, (name, good.stdout, good.stderr)
    broken = _sh(command, tmp_path)
    assert broken.returncode != 0, (name, broken.stdout, broken.stderr)


@pytest.mark.parametrize("name", sorted(TESTS_WIDE))
def test_tests_wide_grep_reads_the_tree_it_runs_in(name: str, tmp_path: Path) -> None:
    """Check the PR tree, then keep the same evidence after dev advances."""
    ticket, evidence_id, index = TESTS_WIDE[name]
    command = _command(ticket, evidence_id, index)
    repo, _ = _synthetic_repo(tmp_path, [])
    assert _sh(command, repo).returncode != 0
    inside = repo / "tests" / "ci" / "test_a.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("check_lane_sync\n")
    landed = _introduce_ticket_contract(repo, "reviewed fix")
    assert _sh(command, repo).returncode == 0

    outside = repo / "tests" / "unit" / "test_b.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("check_lane_sync\n")
    _git(repo, "add", "tests")
    _git(repo, "commit", "-qm", "follow-up test")
    assert _sh(command, repo).returncode != 0

    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    result = _sh(command, repo)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["tests/ci/test_a.py"]
    assert _git(repo, "rev-parse", "HEAD") != landed
    no_tests_dir = tmp_path / "no_tests_dir"
    no_tests_dir.mkdir()
    assert _sh(command, no_tests_dir).returncode != 0


def test_old_semicolon_and_negated_grep_forms_pass_in_a_wrong_cwd(
    tmp_path: Path,
) -> None:
    """Positive control: the forms the review rejected exit 0 where they must not."""
    old_tests_wide = (
        'bash -o pipefail -c \'out=$(grep -rl "check_lane_sync" tests/ | sort); '
        'printf "%s\\n" "$out"; test -n "$out"; '
        'test -z "$(printf "%s\\n" "$out" | grep -v "^tests/ci/")"\''
    )
    old_negated = (
        'bash -o pipefail -c \'! grep -rnE "not live_contact" '
        ".github/workflows tests/ci/conftest.py'"
    )
    assert _sh(old_tests_wide, tmp_path).returncode == 0
    assert _sh(old_negated, tmp_path).returncode == 0


def test_a_missing_file_fails_the_grep_free_of_match_guards(tmp_path: Path) -> None:
    """Exit 2 (a file that is not there) must not read as 'no match'."""
    command = _command(
        "OMN-25184", "dod-omn25184-ac2-unskipped-not-deselected", 1
    ).replace("tests/ci/conftest.py", "tests/ci/conftest_that_is_missing.py")
    assert _sh(command, REPO_ROOT).returncode != 0
    for path in ("tests/conftest.py", "tests/ci/conftest.py", "conftest.py"):
        assert (REPO_ROOT / path).is_file(), path


# One pattern for the scan and for its control: a guard joined with ``;`` and a
# negated grep or printf pipeline (``! grep``, ``! printf ... | grep``).
OFFENDER = re.compile(r";[ \t]*(test|!)|(?<![\w!=])![ \t]+(grep|printf|echo)\b")


def _offending_lines(value: str) -> list[str]:
    return [line for line in value.splitlines() if OFFENDER.search(line)]


def test_no_check_chains_a_guard_with_a_semicolon_or_negates_a_grep() -> None:
    offenders: list[str] = []
    for ticket in CONTRACTS:
        for evidence_id, checks in _checks(ticket).items():
            for check in checks:
                for line in _offending_lines(check["check_value"]):
                    offenders.append(f"{ticket} {evidence_id}: {line[:80]}")
    assert not offenders, offenders


@pytest.mark.parametrize(
    "old",
    [
        'out=$(a); printf x; test -n "$out"; test -z "$(b)"',
        "! grep -rnE x y",
        '! printf "%s\\n" "$out" | grep -Eqi "skipped|deselected"',
        "bash -c '! printf x | grep y'",
        "a && ! grep x f",
    ],
)
def test_the_offender_scan_finds_the_old_shapes(old: str) -> None:
    """Control: the same constant the scan above uses reads each old shape."""
    assert _offending_lines(old), old


@pytest.mark.parametrize(
    "fine",
    [
        'grep -n "x" f\ntest $? -eq 1',
        'out=$(a) && printf "%s" "$out" | grep -Eq "^23 passed in "',
        'test "$(printf x | grep -c y)" = 0',
        'grep -n "! gre[p]" f',
    ],
)
def test_the_offender_scan_passes_the_accepted_shapes(fine: str) -> None:
    assert not _offending_lines(fine), fine


def _cwd_matches_check_count(text: str) -> bool:
    cwd = len(re.findall(r"^ +cwd: \.$", text, flags=re.M))
    checks = len(re.findall(r"^ +- check_type:", text, flags=re.M))
    return cwd > 0 and cwd == checks


@pytest.mark.parametrize("ticket", CONTRACTS)
def test_every_check_has_cwd_dot(ticket: str) -> None:
    text = (REPO_ROOT / "contracts" / f"{ticket}.yaml").read_text()
    assert _cwd_matches_check_count(text), ticket
    assert not re.search(r"^ +cwd:.*OMNI_HOME", text, flags=re.M)


def test_the_cwd_count_helper_sees_a_check_with_no_cwd() -> None:
    good = "    checks:\n      - check_type: command\n        cwd: .\n"
    bad = good + "      - check_type: command\n"
    assert _cwd_matches_check_count(good)
    assert not _cwd_matches_check_count(bad)


def _fake_uv(directory: Path, output: str) -> dict[str, str]:
    directory.mkdir(exist_ok=True)
    shim = directory / "uv"
    shim.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{output}\nEOF\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{directory}:{env['PATH']}"
    return env


SUMMARY_OUTPUTS = [
    ("....\n23 passed in 1.20s", True),
    ("22 passed in 1.20s", False),
    ("22 passed, 1 skipped in 1.20s", False),
    ("23 passed, 1 deselected in 1.20s", False),
    ("SKIPPED [1] x\n23 passed in 1s", False),
    ("23 passed in 1s\nSKIPPED [1] x", False),
    ("error: no pyproject.toml found", False),
    ("", False),
]

# Every check that reads the pytest summary of the 23-test file.
SUMMARY_CHECKS = {
    "25101-ac1": ("OMN-25101", "dod-omn25101-ac1-fixture-exit-and-unskipped", 1),
    "25184-ac1": ("OMN-25184", "dod-omn25184-ac1-guard-file-23-passed", 0),
    "25199-ac4": ("OMN-25199", "dod-omn25199-ac4-guard-files-23-passed", 0),
}


@pytest.mark.parametrize(("output", "passes"), SUMMARY_OUTPUTS)
def test_ac2_check_needs_the_pytest_summary_and_no_skip(
    output: str, passes: bool, tmp_path: Path
) -> None:
    command = _command("OMN-25184", "dod-omn25184-ac2-unskipped-not-deselected", 0)
    env = _fake_uv(tmp_path / "shim", output)
    assert (_sh(command, REPO_ROOT, env).returncode == 0) is passes, output


def test_the_skip_count_is_what_decides_the_skipped_case(tmp_path: Path) -> None:
    """Control: without its skip clause the check passes the skipped output."""
    command = _command("OMN-25184", "dod-omn25184-ac2-unskipped-not-deselected", 0)
    without_skip_clause = command.replace(
        '&&\ntest "$(printf "%s\\n" "$out" | grep -Eci "skipped|deselected")" = 0', ""
    )
    assert without_skip_clause != command
    env = _fake_uv(tmp_path / "shim", "SKIPPED [1] x\n23 passed in 1s")
    assert _sh(without_skip_clause, REPO_ROOT, env).returncode == 0
    assert _sh(command, REPO_ROOT, env).returncode != 0


@pytest.mark.parametrize("name", sorted(SUMMARY_CHECKS))
@pytest.mark.parametrize(
    ("output", "passes"), SUMMARY_OUTPUTS[:2] + SUMMARY_OUTPUTS[-2:]
)
def test_the_23_passed_checks_need_the_summary(
    name: str, output: str, passes: bool, tmp_path: Path
) -> None:
    ticket, evidence_id, index = SUMMARY_CHECKS[name]
    command = _command(ticket, evidence_id, index)
    env = _fake_uv(tmp_path / "shim", output)
    assert (_sh(command, REPO_ROOT, env).returncode == 0) is passes, (name, output)


@pytest.mark.parametrize("name", sorted(SUMMARY_CHECKS))
def test_the_23_passed_checks_fail_when_pytest_exits_non_zero(
    name: str, tmp_path: Path
) -> None:
    """The ``&&`` join: a pytest that exits 1 cannot be rescued by its summary line."""
    ticket, evidence_id, index = SUMMARY_CHECKS[name]
    command = _command(ticket, evidence_id, index)
    directory = tmp_path / "shim"
    directory.mkdir()
    shim = directory / "uv"
    shim.write_text("#!/usr/bin/env bash\necho '23 passed in 1s'\nexit 1\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{directory}:{env['PATH']}"
    assert _sh(command, REPO_ROOT, env).returncode != 0


def test_old_ac2_form_passed_when_pytest_never_ran(tmp_path: Path) -> None:
    old = (
        "bash -o pipefail -c 'out=$(uv run pytest 2>&1); "
        'printf "%s\\n" "$out" | tail -3; '
        '! printf "%s\\n" "$out" | grep -Eqi "skipped|deselected"\''
    )
    env = _fake_uv(tmp_path / "shim", "error: no pyproject.toml found")
    assert _sh(old, REPO_ROOT, env).returncode == 0


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return done.stdout.strip()


def _synthetic_repo(tmp_path: Path, messages: list[str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    for message in messages:
        _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return repo, base


def _introduce_ticket_contract(repo: Path, message: str) -> str:
    contract = repo / "contracts" / "OMN-24976.yaml"
    contract.parent.mkdir()
    contract.write_text("ticket_id: OMN-24976\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


CLOSURE_CHECKS = {
    "25184-ac5": ("OMN-25184", "dod-omn25184-ac5-no-git-closure-path", 0),
    "25199-ac1": ("OMN-25199", "dod-omn25199-ac1-closure-claim", 0),
}


def _closure_command(base: str, name: str = "25184-ac5") -> str:
    ticket, evidence_id, index = CLOSURE_CHECKS[name]
    command = _command(ticket, evidence_id, index)
    assert BASE_SHA in command
    return command.replace(BASE_SHA, base)


@pytest.mark.parametrize(
    "message",
    [
        "fix(OMN-24976): a lane census drift fails lane_sync",
        "test(OMN-24976): add failing tests",
        "Refs OMN-24976",
        "fix(OMN-25199): cwd guards\n\nTicket: OMN-24976",
        "Closes OMN-25101\n\nRefs OMN-24976",
        "Fixes [OMN-25101](https://tracker.example/issue/OMN-25101)",
    ],
)
@pytest.mark.parametrize("name", sorted(CLOSURE_CHECKS))
def test_commit_messages_that_only_reference_the_ticket_pass(
    message: str, name: str, tmp_path: Path
) -> None:
    repo, base = _synthetic_repo(tmp_path, [message])
    assert _sh(_closure_command(base, name), repo).returncode == 0


@pytest.mark.parametrize(
    "message",
    [
        "fix: x\n\nCloses OMN-24976",
        "fix: x\n\ncloses: OMN-24976",
        "Fixes OMN-24976",
        "Resolves https://linear.app/omninode/issue/OMN-24976",
        "Completed OMN-24976",
        "Closes OMN-25101, OMN-24976",
        "Closes OMN-25101 and OMN-24976",
        "fix: x\n\nResolves OMN-25101, OMN-25184, OMN-24976",
        "Fixes [OMN-24976](https://tracker.example/issue/OMN-24976)",
        "Closes [OMN-25101](https://tracker.example/issue/OMN-25101), [OMN-24976]",
        "Fixes [OMN-24976]",
        "Fixes: OMN-25101,OMN-24976",
    ],
)
@pytest.mark.parametrize("name", sorted(CLOSURE_CHECKS))
def test_a_closing_keyword_before_the_ticket_fails_the_check(
    message: str, name: str, tmp_path: Path
) -> None:
    repo, base = _synthetic_repo(tmp_path, ["fine", message])
    assert _sh(_closure_command(base, name), repo).returncode != 0


@pytest.mark.parametrize("name", sorted(CLOSURE_CHECKS))
@pytest.mark.parametrize(
    ("original_message", "expected_rc"),
    [("Refs OMN-24976", 0), ("Closes OMN-24976", 1)],
)
def test_closing_keyword_evidence_ends_at_the_landed_fix(
    name: str, original_message: str, expected_rc: int, tmp_path: Path
) -> None:
    repo, base = _synthetic_repo(tmp_path, [])
    _introduce_ticket_contract(repo, original_message)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "Closes OMN-24976")
    command = _closure_command(base, name)
    assert _sh(command, repo).returncode == 1

    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    assert _sh(command, repo).returncode == expected_rc


@pytest.mark.parametrize("name", sorted(CLOSURE_CHECKS))
def test_an_unreadable_range_fails_the_closure_check(name: str, tmp_path: Path) -> None:
    repo, _ = _synthetic_repo(tmp_path, ["fine"])
    assert _sh(_closure_command("0" * 12, name), repo).returncode != 0
    assert _sh(_closure_command("HEAD", name), repo).returncode != 0
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    assert _sh(_closure_command("HEAD", name), not_a_repo).returncode != 0


MARKER_OPTION = re.compile(r"""-m[ \t]+(?:"([^"]+)"|'([^']+)')""")


def _marker_expressions_in(text: str) -> list[str]:
    """Every pytest ``-m`` expression of a workflow, in either quote form.

    Only lines that run pytest count (``git commit -m "..."`` is not a marker
    expression), and an expression built from a shell or workflow variable is
    skipped because it cannot be read statically.
    """
    found: list[str] = []
    for line in re.sub(r"\\\n[ \t]*", " ", text).splitlines():
        if "pytest" not in line:
            continue
        for double, single in MARKER_OPTION.findall(line):
            expression = double or single
            if "$" not in expression:
                found.append(expression)
    return found


def _ci_marker_expressions() -> list[str]:
    expressions: list[str] = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.y*ml")):
        expressions.extend(_marker_expressions_in(workflow.read_text()))
    return expressions


def _drops_live_contact(expression: str) -> bool:
    """True when the expression runs a plain unit case but not a live_contact one.

    An expression that selects another tier (``integration and bastion``) runs no
    plain unit case and is not judged; one that runs unit cases must run the same
    cases when they also carry ``live_contact``.
    """
    compiled = Expression.compile(expression)
    runs_unit = compiled.evaluate(lambda name, **_: name == "unit")
    runs_live = compiled.evaluate(lambda name, **_: name in {"unit", "live_contact"})
    return runs_unit and not runs_live


def test_no_ci_marker_expression_deselects_a_live_contact_case() -> None:
    expressions = _ci_marker_expressions()
    assert len(expressions) >= 3, expressions
    for expression in expressions:
        assert not _drops_live_contact(expression), expression


def test_the_marker_scan_reads_both_quote_forms_and_skips_other_commands() -> None:
    text = (
        'run: uv run pytest -m "not live_contact and not (slow)" tests/\n'
        "run: uv run pytest -m 'not live_contact or kafka' tests/\n"
        'run: uv run pytest -m "$GATE" tests/\n'
        'run: git commit -m "not live_contact"\n'
    )
    assert _marker_expressions_in(text) == [
        "not live_contact and not (slow)",
        "not live_contact or kafka",
    ]
    for expression in _marker_expressions_in(text):
        assert _drops_live_contact(expression), expression


def test_the_marker_evaluator_deselects_what_it_should() -> None:
    """Control: the evaluator used above does report a deselection."""
    compiled = Expression.compile("not (live_contact or slow)")
    assert not compiled.evaluate(lambda name, **_: name == "live_contact")
    assert _drops_live_contact("not (live_contact or slow)")
    assert not _drops_live_contact("integration and bastion")
    assert not _drops_live_contact("not slow and not kafka")


def test_live_contact_is_registered_and_selected_by_its_own_workflow() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert '"live_contact(artifact):' in pyproject
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-live-contact.yml").read_text()
    assert "'-m', 'live_contact'" in workflow


def test_the_census_helper_callers_pass_an_empty_string_only_on_purpose() -> None:
    """OMN-25199 finding 8: only the named-but-empty test sends ``""``."""
    source = (
        REPO_ROOT
        / "tests/unit/scripts/test_lane_census_unpublishable_drift_omn20798.py"
    ).read_text()
    tree = ast.parse(source)
    senders: list[str] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for call in ast.walk(function):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_run"
                and any(
                    isinstance(k.value, ast.Constant) and k.value.value == ""
                    for k in call.keywords
                )
            ):
                senders.append(function.name)
    assert senders == [
        "test_lane_census_empty_broker_variable_is_unset_not_indeterminate"
    ]
    script = (REPO_ROOT / "scripts" / "lane-census-check.sh").read_text()
    assert '-z "${LANE_MEMORY_BROKER_CONTAINER:-}"' in script


# OMN-25205 finding 3: what a FAIL receipt does to the job that emits it.


@pytest.mark.parametrize("outcome", ["fail", "indeterminate"])
def test_a_receipt_with_a_non_passing_lane_sync_ends_the_emit_step_red(
    outcome: str, tmp_path: Path
) -> None:
    """A lane_sync that is not ok exits the emit step 1, failed or indeterminate.

    So the change from indeterminate to failed (census exit 8) does not move the
    emitting job's exit: it was already 1 at the merge base.
    """
    out = tmp_path / "receipt.json"
    argv = [
        "emit",
        "--sha",
        "78bbba2f9b28cf11ef8dccf45e07a251381700d4",
        "--lane",
        EnumLabLane.COMPOSE_DEV.value,
        "--started-at",
        "2026-11-16T07:00:00Z",
        "--finished-at",
        "2026-11-16T07:05:00Z",
        "--check",
        "ready_main:ok:GET /ready -> 200",
        "--check",
        f"lane_sync:{outcome}:census exit=8 findings=4",
        "--out",
        str(out),
    ]
    assert receipt_main(argv) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["result"] == "FAIL"


def test_the_emitting_job_is_a_post_merge_job_outside_ci_summary() -> None:
    """The red lands on a post-merge job that no required context reads.

    The workflow runs on a closed pull request, so it is not a check of the PR
    under review, and the one required context of dev, CI Summary, is a job of
    ci.yml with no ``needs`` that reads only its own run's jobs.
    """
    workflows = REPO_ROOT / ".github" / "workflows"
    rebuild = yaml.safe_load((workflows / "runtime-rebuild-trigger.yml").read_text())
    trigger = rebuild.get("on", rebuild.get(True))
    assert trigger == {
        "pull_request": {"types": ["closed"], "branches": ["main", "dev"]}
    }
    assert "verify-lane-converged" in rebuild["jobs"]
    ci = yaml.safe_load((workflows / "ci.yml").read_text())
    summary = ci["jobs"]["ci-summary"]
    assert summary["name"] == "CI Summary"
    assert "needs" not in summary
    assert not [
        name
        for name, job in ci["jobs"].items()
        if "lab_pass_receipt.py emit" in json.dumps(job)
    ]
