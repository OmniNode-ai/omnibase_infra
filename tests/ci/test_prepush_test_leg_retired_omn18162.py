# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-push TEST leg stays retired in this repo (OMN-18162).

OMN-18162 removed the governed impacted-test selector (``prepush-smart-tests``,
entry ``bash scripts/hooks/prepush_smart_tests.sh``) from the pre-push stage of
``.pre-commit-config.yaml``. It was the only hook here that invoked a test
runner, so pre-push is now policy, type and lint only and finishes in seconds.
Hosted CI is the enforced merge gate and the test surface.

Plan of record: ``knowledge-base-internal#328``,
``beta/plans/2026-09-09-ci-runner-placement-and-pre-push-retirement-plan.md``,
phase 1. Operator ruling 2026-09-10: retire the test leg one repo at a time.

Per repo rule 5 (enforcement, not detection) this module is the mechanism, not
a note. Detection that is not wired as a gate gets ignored, and a retirement
that lives only in a comment is re-added by the next lane that wants faster
local feedback.

Four properties are asserted, and each one fails a different way of undoing the
change:

1. **No pre-push hook invokes a test runner.** Stated as a property of the
   stage rather than a denylist of one hook id, so re-adding the same behaviour
   under a new id or a new wrapper script is caught too.
2. **The other three pre-push hooks survive, by id and in order.** Plan AC1
   forbids removing, reordering or changing any other pre-push hook; a
   retirement that also quietly drops the deploy-scope gate is not this change.
3. **The selector script is NOT deleted, and says it is manual-only.** Plan AC6.
   ``docker/docker-compose.gate-runner.yml`` and
   ``config/runner_routing_policy.yaml`` reference it and roughly ten test
   modules pin its content, so deleting it breaks surfaces the retirement never
   intended to touch.
4. **No bypass surface was introduced.** Plan AC2 / repo rule 10. The retirement
   must not ship an env knob that turns the leg back on, because an ambient
   variable is inherited by every descendant process and leaves no receipt.

This is a hermetic static scan. It reads files off disk, runs no subprocess and
touches no network.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
SELECTOR_SCRIPT = REPO_ROOT / "scripts" / "hooks" / "prepush_smart_tests.sh"

PRE_PUSH_STAGE = "pre-push"

# The pre-push stage after OMN-18162, by hook id, in file order. The plan's AC1a
# records the post-change count for this repo as 3.
EXPECTED_PRE_PUSH_HOOK_IDS: tuple[str, ...] = (
    "mypy-type-check",
    "onex-validate-architecture-layers",
    "prepush-deploy-scope-dod",
)

# Tokens that mean "this hook runs a test runner". Matched against a hook's
# resolved command surface (entry plus the script it names, when that script
# lives in this repo).
_TEST_RUNNER_RE = re.compile(r"\b(pytest|py\.test|unittest)\b")

# Env names that would restore the retired leg if a later change reintroduced
# one as a live knob. Their appearance inside the retained selector script is
# expected and is not a bypass -- the script is manual-only. Their appearance in
# the pre-commit config would be.
_RESTORE_KNOB_RE = re.compile(r"\b(PREPUSH_[A-Z0-9_]+|ENABLE_SMART_TESTS)\b")


def _load_config() -> dict[str, Any]:
    raw = yaml.safe_load(PRECOMMIT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"{PRECOMMIT_CONFIG} did not parse to a mapping"
    return raw


def _pre_push_hooks() -> list[dict[str, Any]]:
    """Hooks that DECLARE the pre-push stage in this config, in file order.

    This is the narrow set the plan counts. Its AC1a records the post-change
    figure for omnibase_infra as 3.
    """
    hooks: list[dict[str, Any]] = []
    for repo in _load_config().get("repos") or []:
        for hook in repo.get("hooks") or []:
            if PRE_PUSH_STAGE in (hook.get("stages") or []):
                hooks.append(hook)
    return hooks


def _hooks_reaching_pre_push() -> list[dict[str, Any]]:
    """Hooks that can actually execute on `git push`, which is a wider set.

    Measured, not assumed. This config sets ``default_stages: [pre-commit]``,
    so a hook with no ``stages:`` key looks pinned away from pre-push -- but
    ``pre-commit run --hook-stage pre-push --all-files`` in this repo runs six
    hooks, not the three that declare the stage. ``default_stages`` yields to a
    stage declaration in the upstream hook's own manifest, which this config
    cannot see and this test cannot read for a remote repo.

    So the retirement property is asserted against every hook that is not
    positively pinned to some other stage. That over-approximates rather than
    under-approximates: a hook wrongly included merely has to not run tests,
    while a hook wrongly excluded is a test leg this gate would miss. Given the
    whole point is that no test runner executes on push, the over-approximation
    is the correct direction to be wrong in.
    """
    hooks: list[dict[str, Any]] = []
    for repo in _load_config().get("repos") or []:
        for hook in repo.get("hooks") or []:
            stages = hook.get("stages")
            if not stages or PRE_PUSH_STAGE in stages:
                hooks.append(hook)
    return hooks


def _strip_comments(text: str, marker: str = "#") -> str:
    """Drop whole-line comments so prose about a token is not read as the token.

    Repo rule 15 in the other direction: a comment naming a retired knob is
    documentation, and a gate that fires on documentation about itself is the
    failure mode that broke three separate pull requests in one window.
    """
    kept = [line for line in text.splitlines() if not line.lstrip().startswith(marker)]
    return "\n".join(kept)


def _strip_python_prose(source: str) -> str:
    """Remove comments and docstrings from Python source, keeping other strings.

    Docstrings are the reason this exists. ``scripts/validate.py`` documents a
    command a reader could run by hand -- ``pytest
    tests/ci/test_architecture_compliance.py`` -- inside the architecture-layer
    validator's docstring, while the function itself shells out to
    ``check_architecture.sh``. Scanning raw text reads that sentence as an
    invocation and fails the retirement check on a hook that runs no tests. The
    OMN-18162 plan found the same string by the same means and recorded it as
    the one false positive in its own inventory.

    Only docstrings and comments go. Ordinary string literals stay, because
    ``subprocess.run(["pytest", ...])`` is a real invocation that lives entirely
    inside one -- dropping every ``STRING`` token would make this scan fail open
    on the most direct way to run a test suite from Python. The positive control
    in this module caught exactly that, which is why it is here.

    A file that will not parse is returned unchanged, so a syntax error degrades
    to the stricter reading rather than to a silent pass.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _strip_comments(source)

    prose_lines: set[int] = set()
    for node in ast.walk(tree):
        # A bare string expression statement is a docstring or a block comment
        # written as one. Either way it does not execute.
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.end_lineno is not None
        ):
            prose_lines.update(range(node.lineno, node.end_lineno + 1))

    kept = [
        line
        for number, line in enumerate(source.splitlines(), start=1)
        if number not in prose_lines
    ]
    return _strip_comments("\n".join(kept))


def _executable_text(path: Path) -> str:
    """A file's content with its non-executing prose removed."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:  # pragma: no cover - binary entry
        return ""
    if path.suffix == ".py":
        return _strip_python_prose(source)
    return _strip_comments(source)


def _command_surface(hook: dict[str, Any]) -> str:
    """The hook's entry, plus the executable body of any in-repo script it names.

    Resolving one level past the entry is the point. The OMN-18162 plan's own
    round-1 inventory missed a test invocation precisely because it searched
    pre-commit configurations for the name of a test runner, and the hook it
    missed named a script whose imported module launched the runner.
    """
    entry = str(hook.get("entry", ""))
    surface = [entry]
    for token in entry.split():
        candidate = REPO_ROOT / token
        if candidate.is_file():
            surface.append(_executable_text(candidate))
    return "\n".join(surface)


@pytest.mark.unit
def test_no_pre_push_hook_invokes_a_test_runner() -> None:
    """The retired leg stays retired, stated as a property of the stage.

    Asserted against the resolved command surface rather than a denylist of one
    hook id, so the same behaviour under a different id or a different wrapper
    fails here too.
    """
    offenders: list[str] = []
    for hook in _hooks_reaching_pre_push():
        surface = _strip_comments(_command_surface(hook))
        match = _TEST_RUNNER_RE.search(surface)
        if match is not None:
            offenders.append(f"{hook.get('id')!r} (matched {match.group(0)!r})")

    assert not offenders, (
        "A pre-push hook invokes a test runner, which OMN-18162 retired in this "
        f"repo: {', '.join(offenders)}. Hosted CI is the enforced merge gate and "
        "the test surface. Do not re-add a test-running hook at the pre-push "
        "stage without a ruling that supersedes the operator ruling of "
        "2026-09-10 recorded in .pre-commit-config.yaml; if you have one, change "
        "this test in the same commit and cite the ruling here."
    )


@pytest.mark.unit
def test_scan_detects_a_real_test_invocation(tmp_path: Path) -> None:
    """Positive control for the zero above (repo rule 16).

    An empty offender list is only evidence when the same scan is known to
    return rows against an input that should produce one. Without this, a
    ``_strip_python_prose`` that swallowed the whole file, or a regex that
    matched nothing, would read exactly like a clean retirement.

    Two inputs, because the two halves fail differently: a shell entry that
    calls the runner directly, and a Python entry whose *code* calls it while
    its docstring does not -- the inverse of the ``scripts/validate.py`` case
    that made this stripping necessary.
    """
    shell_script = tmp_path / "runs_tests.sh"
    shell_script.write_text("#!/usr/bin/env bash\nuv run pytest tests/unit/\n")
    assert _TEST_RUNNER_RE.search(_executable_text(shell_script)) is not None, (
        "The scan failed to see a plain shell pytest invocation, so a zero from "
        "it is not evidence of a retired leg."
    )

    py_script = tmp_path / "runs_tests.py"
    py_script.write_text(
        '"""A docstring that names no runner."""\n'
        "import subprocess\n"
        'subprocess.run(["pytest", "tests/unit/"], check=True)\n'
    )
    assert _TEST_RUNNER_RE.search(_executable_text(py_script)) is not None, (
        "The scan failed to see a Python pytest invocation outside a docstring, "
        "so tokenising is stripping executable code and the zero is false."
    )

    prose_only = tmp_path / "documents_a_runner.py"
    prose_only.write_text('"""For comprehensive analysis, use: pytest tests/ci/."""\n')
    assert _TEST_RUNNER_RE.search(_executable_text(prose_only)) is None, (
        "The scan read a docstring as an invocation. This is the false positive "
        "that scripts/validate.py produces and that the stripping exists to fix."
    )


@pytest.mark.unit
def test_prepush_smart_tests_hook_is_unwired() -> None:
    """The specific retired hook id appears at no stage in the config.

    Narrower than the property above and worth keeping separate: this one names
    what was removed, so a failure reads as "the retirement was reverted" rather
    than as a generic policy breach.
    """
    all_ids = [
        hook.get("id")
        for repo in (_load_config().get("repos") or [])
        for hook in (repo.get("hooks") or [])
    ]
    assert "prepush-smart-tests" not in all_ids, (
        "The 'prepush-smart-tests' hook is wired again. OMN-18162 retired the "
        "pre-push test leg in this repo; see the retirement block in "
        ".pre-commit-config.yaml for the plan of record and the ruling."
    )


@pytest.mark.unit
def test_other_pre_push_hooks_are_untouched() -> None:
    """The three surviving pre-push hooks, by id and in order (plan AC1/AC1a).

    The retirement removes exactly one hook. A change that also drops or
    reorders the deploy-scope gate or the architecture-layer validator is a
    different change and does not ride along on this one.
    """
    actual = tuple(str(hook.get("id")) for hook in _pre_push_hooks())
    assert actual == EXPECTED_PRE_PUSH_HOOK_IDS, (
        "The pre-push stage changed beyond the OMN-18162 retirement.\n"
        f"  expected: {EXPECTED_PRE_PUSH_HOOK_IDS}\n"
        f"  actual:   {actual}\n"
        "Pre-push is policy, type and lint only. Adding a non-test hook here is "
        "allowed, but update this tuple in the same commit so the stage's "
        "contents stay asserted rather than assumed."
    )


@pytest.mark.unit
def test_selector_script_is_retained_and_marked_manual_only() -> None:
    """The script survives the retirement and says so in its header (plan AC6).

    It is referenced by the gate-runner compose file and the runner routing
    policy, and its content is pinned by roughly ten test modules, so deleting
    it would break surfaces this change never meant to touch.
    """
    assert SELECTOR_SCRIPT.is_file(), (
        f"{SELECTOR_SCRIPT.relative_to(REPO_ROOT)} was deleted. OMN-18162 "
        "unwires it from pre-push and keeps it as a manually-invocable library; "
        "docker/docker-compose.gate-runner.yml and "
        "config/runner_routing_policy.yaml both reference it."
    )

    header = "\n".join(SELECTOR_SCRIPT.read_text(encoding="utf-8").splitlines()[:60])
    assert "MANUAL INVOCATION ONLY" in header, (
        "The selector script's header must state that it is manual-invocation "
        "only, so a reader who finds it does not assume it still runs on push."
    )
    assert "OMN-18162" in header, (
        "The selector script's header must cite OMN-18162, so the reason it is "
        "unwired is resolvable from the file itself."
    )


@pytest.mark.unit
def test_retirement_introduced_no_bypass_knob() -> None:
    """No env knob in the pre-commit config restores the retired leg (AC2).

    Repo rule 10 and the selector's own OMN-16480 finding: an ambient
    environment variable is inherited by every descendant process, is bound to
    no repo or commit, never expires and leaves no receipt. The retirement must
    not ship one, and it does not need one -- the leg is simply gone.
    """
    executable_config = _strip_comments(PRECOMMIT_CONFIG.read_text(encoding="utf-8"))
    found = sorted(set(_RESTORE_KNOB_RE.findall(executable_config)))
    assert not found, (
        "The pre-commit config names pre-push test-leg env knobs outside a "
        f"comment: {found}. OMN-18162 retires the leg outright; a knob that "
        "turns it back on is the bypass surface the retirement removes."
    )


@pytest.mark.unit
def test_deselected_unit_tests_keep_an_execution_path() -> None:
    """The tests pull-request CI deselects still run somewhere (OMN-18162).

    This is the half of the retirement that is easy to get wrong. Pre-push ran
    ``-m "not integration"``; ci.yml runs ``-m "not slow and not chaos and not
    kafka and not performance and not live_github_api"``. Collecting
    ``tests/unit`` under each filter and diffing gave 39 tests -- 25
    ``performance``, 14 ``slow`` -- whose only execution path was the leg this
    change removes. Losing them silently was the one outcome the OMN-18162
    review named as the bad option, ahead of both keeping the leg and deleting
    the tests.

    ``microbenchmarks-nightly.yml`` is that path. Asserting its existence and
    its selection here means deleting it fails a test rather than quietly
    restoring the gap.
    """
    workflow = REPO_ROOT / ".github" / "workflows" / "microbenchmarks-nightly.yml"
    assert workflow.is_file(), (
        "microbenchmarks-nightly.yml is gone. It is the only execution path for "
        "the 39 tests/unit tests that pull-request CI deselects by marker. "
        "Deleting it re-opens the coverage gap OMN-18162 closed; if these tests "
        "should not run at all, delete the tests too and update this module."
    )

    parsed = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)

    # `on` is parsed by PyYAML 1.1 rules as the boolean True, not the string.
    triggers = parsed.get("on", parsed.get(True))
    assert isinstance(triggers, dict) and "schedule" in triggers, (
        "The deselected-test workflow must stay scheduled. A manual-only "
        "workflow is not an execution path, it is a button nobody presses."
    )

    body = workflow.read_text(encoding="utf-8")
    for marker in ("performance", "slow"):
        assert marker in body, (
            f"The nightly no longer selects the {marker!r} marker, which "
            "pull-request CI deselects. Those tests would then run nowhere."
        )
    assert "ONEX_RUN_MICROBENCHMARKS" in body, (
        "The nightly must set ONEX_RUN_MICROBENCHMARKS. Four microbenchmarks in "
        "tests/unit/runtime/test_policy_registry_performance.py skip without "
        "it, so scheduling alone would run zero of them and report green."
    )


@pytest.mark.unit
def test_microbenchmark_skip_is_not_keyed_on_the_generic_ci_variable() -> None:
    """The microbenchmark skip may not key off ``CI`` again (OMN-18162).

    ``CI`` is set by every GitHub Actions runner, so a ``skipif`` on it means
    "never runs in CI, anywhere, including a job written specifically to run
    it". Combined with ci.yml's marker deselection that left four tests with no
    execution path at all, discoverable only by reading two filters against
    each other. The switch is an explicit opt-in now, and must stay one.
    """
    module = (
        REPO_ROOT / "tests" / "unit" / "runtime" / "test_policy_registry_performance.py"
    )
    executable = _strip_python_prose(module.read_text(encoding="utf-8"))
    assert 'os.environ.get("CI"' not in executable, (
        "test_policy_registry_performance.py keys a skip off the generic CI "
        "environment variable again. Every Actions runner sets it, so the "
        "tests it guards can never run in any job. Gate them on "
        "ONEX_RUN_MICROBENCHMARKS, which only microbenchmarks-nightly.yml sets."
    )
    assert "ONEX_RUN_MICROBENCHMARKS" in executable, (
        "test_policy_registry_performance.py no longer reads "
        "ONEX_RUN_MICROBENCHMARKS, so microbenchmarks-nightly.yml cannot lift "
        "its skips and the nightly would report green having run nothing."
    )
