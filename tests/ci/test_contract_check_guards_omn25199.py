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
import os
import re
import stat
import subprocess
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
CONTRACTS = ("OMN-24976", "OMN-25101", "OMN-25184", "OMN-25199")
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


# The cheap, deterministic guard checks of the three review contracts: every
# one of them must pass in the checkout and fail in a directory that is not.
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
    "25184-ac4-tests-wide-grep": (
        "OMN-25184",
        "dod-omn25184-ac4-tests-wide-grep",
        0,
    ),
    "25101-ac3-tests-wide-grep": (
        "OMN-25101",
        "dod-omn25101-ac3-no-other-pin-and-full-suite",
        0,
    ),
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


def test_no_check_chains_a_guard_with_a_semicolon_or_negates_a_grep() -> None:
    offenders: list[str] = []
    for ticket in CONTRACTS:
        for evidence_id, checks in _checks(ticket).items():
            for check in checks:
                value = check["check_value"]
                for line in value.splitlines():
                    if re.search(r";[ \t]*(test|!)", line) or "! grep" in line:
                        offenders.append(f"{ticket} {evidence_id}: {line[:80]}")
    assert not offenders, offenders


def test_the_ac1_comparison_is_red_for_the_old_text() -> None:
    """The offender scan above finds the old shape (control for it)."""
    old = 'out=$(a); printf x; test -n "$out"; test -z "$(b)"'
    assert re.search(r";[ \t]*(test|!)", old)


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


@pytest.mark.parametrize(
    ("output", "passes"),
    [
        ("....\n23 passed in 1.20s", True),
        ("22 passed in 1.20s", False),
        ("22 passed, 1 skipped in 1.20s", False),
        ("23 passed, 1 deselected in 1.20s", False),
        ("error: no pyproject.toml found", False),
        ("", False),
    ],
)
def test_ac2_check_needs_the_pytest_summary_and_no_skip(
    output: str, passes: bool, tmp_path: Path
) -> None:
    command = _command("OMN-25184", "dod-omn25184-ac2-unskipped-not-deselected", 0)
    env = _fake_uv(tmp_path / "shim", output)
    assert (_sh(command, REPO_ROOT, env).returncode == 0) is passes, output


def test_old_ac2_form_passed_when_pytest_never_ran(tmp_path: Path) -> None:
    old = (
        "bash -o pipefail -c 'out=$(uv run pytest 2>&1); "
        'printf "%s\\n" "$out" | tail -3; '
        '! printf "%s\\n" "$out" | grep -Eqi "skipped|deselected"\''
    )
    env = _fake_uv(tmp_path / "shim", "error: no pyproject.toml found")
    assert _sh(old, REPO_ROOT, env).returncode == 0


def _synthetic_repo(tmp_path: Path, messages: list[str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
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

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    for message in messages:
        git("commit", "-q", "--allow-empty", "-m", message)
    return repo, base


def _closure_command(base: str) -> str:
    command = _command("OMN-25184", "dod-omn25184-ac5-no-git-closure-path", 0)
    assert BASE_SHA in command
    return command.replace(BASE_SHA, base)


@pytest.mark.parametrize(
    "message",
    [
        "fix(OMN-24976): a lane census drift fails lane_sync",
        "test(OMN-24976): add failing tests",
        "Refs OMN-24976",
        "fix(OMN-25199): cwd guards\n\nTicket: OMN-24976",
    ],
)
def test_commit_messages_that_only_reference_the_ticket_pass(
    message: str, tmp_path: Path
) -> None:
    repo, base = _synthetic_repo(tmp_path, [message])
    assert _sh(_closure_command(base), repo).returncode == 0


@pytest.mark.parametrize(
    "message",
    [
        "fix: x\n\nCloses OMN-24976",
        "fix: x\n\ncloses: OMN-24976",
        "Fixes OMN-24976",
        "Resolves https://linear.app/omninode/issue/OMN-24976",
        "Completed OMN-24976",
    ],
)
def test_a_closing_keyword_before_the_ticket_fails_the_check(
    message: str, tmp_path: Path
) -> None:
    repo, base = _synthetic_repo(tmp_path, ["fine", message])
    assert _sh(_closure_command(base), repo).returncode != 0


def test_an_unreadable_range_fails_the_closure_check(tmp_path: Path) -> None:
    repo, _ = _synthetic_repo(tmp_path, ["fine"])
    assert _sh(_closure_command("0" * 12), repo).returncode != 0
    assert _sh(_closure_command("HEAD"), repo).returncode != 0


def _ci_marker_expressions() -> list[str]:
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    found = re.findall(r'-m\s+"([^"]+)"', text)
    return [e for e in found if re.fullmatch(r"[\w\s()]+", e)]


def test_no_ci_marker_expression_deselects_a_live_contact_case() -> None:
    expressions = _ci_marker_expressions()
    assert len(expressions) >= 3, expressions
    for expression in expressions:
        compiled = Expression.compile(expression)
        runs = compiled.evaluate(lambda name, **_: name in {"unit", "live_contact"})
        assert runs, expression


def test_the_marker_evaluator_deselects_what_it_should() -> None:
    """Control: the evaluator used above does report a deselection."""
    compiled = Expression.compile("not (live_contact or slow)")
    assert not compiled.evaluate(lambda name, **_: name == "live_contact")


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
