# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin whole-tree CI counterparts for OMN-19612's staged-file hooks."""

from __future__ import annotations

import fnmatch
import re
import shlex
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import SOFT_ALLOWLIST, STRICT_GATE_JOBS
from scripts.ci.detect_test_paths import (
    PRE_COMMIT_CONFIG_PATH,
    compute_selection,
)
from scripts.ci.run_validators_in_process import parse_specs

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
REQUIRED_CHECKS = REPO_ROOT / ".github" / "required-checks.yaml"
ADJ = REPO_ROOT / "scripts" / "ci" / "test_selection_adjacency.yaml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
THIS_TEST = "tests/ci/test_staged_precommit_backstops_omn19612.py"

# These hooks were already staged-file scoped before OMN-19612. Keeping the
# baseline explicit makes a newly staged-scoped hook fail this test until its
# whole-tree counterpart is added below; the baseline may shrink, not grow.
PRE_EXISTING_STAGED_HOOKS = frozenset(
    {
        "check-lockfile-registry-allowlist",
        "check-pin-reachability",
        "handler-routing-schema-gate",
        "lint-pricing-manifest",
        "onex-imperative-orchestrator-ratchet",
        "onex-orchestration-monolith-ratchet",
        "onex-orchestrator-reducer-state-invariant",
        "onex-validate-backend-secret-discipline",
        "onex-validate-declarative-nodes",
        "operation-match-requires-operation",
        "tenant-scoped-ingress-gate",
        # These were already pass_filenames: true before commit c30aa2504dd9
        # ("fix: scope local validators to staged files (OMN-19612)") -- not
        # moved to staged scope by this PR, so a whole-tree backstop for them
        # is out of scope here. Discovered once _staged_scoped_hook_ids()
        # stopped requiring an explicit files: regex (a types_or: filter is
        # just as much a staged-scope signal as files: is).
        "check-ai-slop",
        "exposed-identifier-gate",
        "no-env-fallbacks",
        "onex-validate-markdown-links",
        "reject-deploy-gate-skip-token",
        "shell-hygiene",
    }
)


@dataclass(frozen=True)
class ShellBackstop:
    """A backstop implemented by a standalone shell command step."""

    job_id: str
    logical_lines: tuple[str, ...]
    allow_echo_banners: bool = False


@dataclass(frozen=True)
class ValidatorBackstop:
    """A backstop implemented as one entry in the validator-runner heredoc."""

    job_id: str
    module: str
    args: tuple[str, ...]


Backstop = ShellBackstop | ValidatorBackstop

BACKSTOPS: dict[str, Backstop] = {
    "check-no-credential-in-log": ShellBackstop(
        job_id="onex-validation",
        logical_lines=(
            "uv run python scripts/ci/check_no_credential_in_log.py "
            "--root src/omnibase_infra",
        ),
        allow_echo_banners=True,
    ),
    "handler-any-signature": ValidatorBackstop(
        job_id="lint",
        module="omnibase_infra.validators.handler_any_signature",
        args=("--max-violations", "0", "src/omnibase_infra"),
    ),
    "envelope-tenant-dimension": ValidatorBackstop(
        job_id="lint",
        module="omnibase_infra.validators.envelope_tenant_dimension",
        args=("src/omnibase_infra",),
    ),
    "kafka-no-hardcoded-fallback": ShellBackstop(
        job_id="lint",
        logical_lines=("bash scripts/validation/check_kafka_no_hardcoded_fallback.sh",),
    ),
    "no-infra-inmemory-import": ShellBackstop(
        job_id="lint",
        logical_lines=("bash scripts/validation/check_no_infra_inmemory_import.sh",),
    ),
    "validate-spdx-headers": ShellBackstop(
        job_id="onex-validation",
        logical_lines=("uv run --frozen onex spdx validate src tests scripts",),
    ),
}

VALIDATOR_RUNNER_LINE = (
    "PYTHONPATH=src uv run python scripts/ci/run_validators_in_process.py "
    "<<'VALIDATORS'"
)
PYTEST_STEP_NAMES = (
    "Run pytest (smart selection)",
    "Run pytest (full suite)",
)
_HEREDOC_PATTERN = re.compile(
    r"<<(?P<strip_tabs>-)?\s*(?P<quote>['\"]?)"
    r"(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)
_GITHUB_EXPRESSION = re.compile(r"\$\{\{.*?\}\}")


@dataclass(frozen=True)
class Heredoc:
    """A heredoc body kept out of the shell command-line stream."""

    delimiter: str
    body: str
    terminated: bool


@dataclass(frozen=True)
class ShellRun:
    """Logical executable lines and separately captured heredoc bodies."""

    logical_lines: tuple[str, ...]
    heredocs: tuple[Heredoc, ...]


# Steps in a backstop's job that are legitimately advisory (report-only
# ratchets unrelated to any staged-scoped hook) and so are allowed to carry
# continue-on-error: true without failing the fail-closed check below. Only
# the step(s) that actually implement a BACKSTOPS whole-tree run are checked.
ADVISORY_STEP_NAMES = frozenset(
    {
        "Run imperative-orchestrator ratchet report (ARCH-004)",
        "Run orchestration-monolith ratchet report (ARCH-004 Signal B)",
    }
)
ALLOWED_BACKSTOP_STEP_CONDITIONS = frozenset({"always()", "success()", "!cancelled()"})


def _load_yaml(path: Path) -> dict[object, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path.name} did not parse to a mapping"
    return loaded


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _shell_run(run: str) -> ShellRun:
    """Parse a workflow ``run`` block into executable lines and heredocs."""
    physical_lines = run.splitlines()
    logical_lines: list[str] = []
    heredocs: list[Heredoc] = []
    index = 0

    while index < len(physical_lines):
        line = physical_lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue

        while line.endswith("\\") and index < len(physical_lines):
            continuation = physical_lines[index].strip()
            index += 1
            line = f"{line[:-1].rstrip()} {continuation}".strip()

        logical_lines.append(line)
        match = _HEREDOC_PATTERN.search(line)
        if match is None:
            continue

        delimiter = match.group("delimiter")
        strip_tabs = match.group("strip_tabs") is not None
        body_lines: list[str] = []
        terminated = False
        while index < len(physical_lines):
            body_line = physical_lines[index]
            index += 1
            shell_body_line = body_line.lstrip("\t") if strip_tabs else body_line
            if shell_body_line == delimiter:
                terminated = True
                break
            body_lines.append(shell_body_line)
        heredocs.append(
            Heredoc(
                delimiter=delimiter,
                body="\n".join(body_lines),
                terminated=terminated,
            )
        )

    return ShellRun(tuple(logical_lines), tuple(heredocs))


def _normalise_whitespace(value: str) -> str:
    return " ".join(value.split())


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list), "workflow job has no steps list"
    assert all(isinstance(step, dict) for step in steps), (
        "workflow job contains a non-mapping step"
    )
    return steps


def _is_safe_echo_banner(line: str) -> bool:
    forbidden = (";", "&&", "||", "|", "$(", "`")
    return line.startswith("echo ") and not any(token in line for token in forbidden)


def _step_lines(step: dict[str, Any], *, allow_echo_banners: bool) -> tuple[str, ...]:
    parsed = _shell_run(str(step.get("run", "")))
    lines = tuple(_normalise_whitespace(line) for line in parsed.logical_lines)
    if allow_echo_banners:
        return tuple(line for line in lines if not _is_safe_echo_banner(line))
    return lines


def _matching_backstop_steps(
    job: dict[str, Any], backstop: Backstop
) -> list[dict[str, Any]]:
    if isinstance(backstop, ShellBackstop):
        expected = tuple(_normalise_whitespace(line) for line in backstop.logical_lines)
        return [
            step
            for step in _steps(job)
            if _step_lines(step, allow_echo_banners=backstop.allow_echo_banners)
            == expected
        ]

    return [
        step
        for step in _steps(job)
        if _step_lines(step, allow_echo_banners=False) == (VALIDATOR_RUNNER_LINE,)
    ]


def _normalise_condition(value: object) -> str:
    condition = str(value).strip()
    return condition.removeprefix("${{").removesuffix("}}").strip()


def _is_true(value: object) -> bool:
    return value is True or _normalise_condition(value).lower() == "true"


def _assert_no_default_run_shell(scope: Mapping[Any, Any], scope_name: str) -> None:
    defaults = scope.get("defaults")
    if defaults is None:
        return
    assert isinstance(defaults, dict), f"{scope_name} defaults is not a mapping"
    run_defaults = defaults.get("run")
    if run_defaults is None:
        return
    assert isinstance(run_defaults, dict), f"{scope_name} defaults.run is not a mapping"
    assert "shell" not in run_defaults, (
        f"{scope_name} sets defaults.run.shell, which can change backstop semantics"
    )


def _assert_backstop_step_shape(hook_id: str, step: dict[str, Any]) -> None:
    step_name = str(step.get("name", "<unnamed>"))
    for key in ("shell", "working-directory", "env"):
        assert key not in step, (
            f"{hook_id} whole-tree step {step_name!r} sets forbidden {key!r}"
        )
    if "if" in step:
        condition = _normalise_condition(step["if"])
        assert condition in ALLOWED_BACKSTOP_STEP_CONDITIONS, (
            f"{hook_id} whole-tree step {step_name!r} has conditional `if`: "
            f"{step['if']!r}; allowed unconditional forms are "
            f"{sorted(ALLOWED_BACKSTOP_STEP_CONDITIONS)!r}"
        )
    assert not _is_true(step.get("continue-on-error", False)), (
        f"{hook_id} whole-tree step {step_name!r} is continue-on-error"
    )


def _option_values(tokens: list[str], names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in names:
            assert index + 1 < len(tokens), f"pytest option {token!r} has no value"
            values.append(tokens[index + 1])
            index += 2
            continue
        matching_name = next(
            (name for name in names if token.startswith(f"{name}=")), None
        )
        if matching_name is not None:
            values.append(token.removeprefix(f"{matching_name}="))
        index += 1
    return values


def _has_option(tokens: list[str], names: tuple[str, ...]) -> bool:
    return any(
        token in names or any(token.startswith(f"{name}=") for name in names)
        for token in tokens
    )


def _ignore_covers_this_test(value: str) -> bool:
    normalised = value
    while normalised.startswith("./"):
        normalised = normalised[2:]
    path_prefix = normalised.rstrip("/")
    covers_path = path_prefix == THIS_TEST or THIS_TEST.startswith(f"{path_prefix}/")
    return covers_path or fnmatch.fnmatch(THIS_TEST, normalised)


def _pytest_tokens(step: dict[str, Any]) -> list[str]:
    pytest_lines = [
        _normalise_whitespace(line)
        for line in _shell_run(str(step.get("run", ""))).logical_lines
        if _normalise_whitespace(line).startswith("uv run pytest ")
        or _normalise_whitespace(line) == "uv run pytest"
    ]
    assert len(pytest_lines) == 1, (
        f"pytest step {step.get('name', '<unnamed>')!r} must have exactly one "
        "logical `uv run pytest` command"
    )
    command = _GITHUB_EXPRESSION.sub("GITHUB_EXPRESSION", pytest_lines[0])
    return shlex.split(command)


def _pyproject_addopts_tokens() -> list[str]:
    pytest_options = _load_toml(PYPROJECT)["tool"]["pytest"]["ini_options"]
    addopts = pytest_options.get("addopts", [])
    if isinstance(addopts, str):
        return shlex.split(addopts)
    assert isinstance(addopts, list) and all(
        isinstance(option, str) for option in addopts
    ), "pytest addopts must be a string or list of strings"
    return [token for option in addopts for token in shlex.split(option)]


def _staged_scoped_hook_ids() -> set[str]:
    """Hooks pre-commit hands only the staged diff at commit time.

    pass_filenames: true is the actual staged-scope signal -- it is what
    limits the hook's file arguments to the staged diff. A files: regex or
    a types_or: filter narrows WHICH staged files qualify; neither is
    required for the hook to be staged-scoped, and a hook that filters by
    types_or alone (no files:) is exactly as staged-scoped as one that uses
    files:.
    """
    config = _load_yaml(PRECOMMIT_CONFIG)
    return {
        str(hook["id"])
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("pass_filenames") is True
        and "pre-commit" in hook.get("stages", ["pre-commit"])
    }


def _workflow() -> dict[object, Any]:
    return _load_yaml(CI_WORKFLOW)


def _job(job_id: str) -> dict[str, Any]:
    job = _workflow()["jobs"][job_id]
    assert isinstance(job, dict)
    return job


def test_every_new_staged_hook_has_a_declared_whole_tree_backstop() -> None:
    staged = _staged_scoped_hook_ids()
    assert staged >= PRE_EXISTING_STAGED_HOOKS
    assert staged - PRE_EXISTING_STAGED_HOOKS == set(BACKSTOPS)


@pytest.mark.parametrize("hook_id", BACKSTOPS)
def test_whole_tree_counterpart_is_in_its_required_job(hook_id: str) -> None:
    backstop = BACKSTOPS[hook_id]
    job = _job(backstop.job_id)
    backstop_steps = _matching_backstop_steps(job, backstop)
    assert backstop_steps, (
        f"{hook_id} has no exact whole-tree execution step in {backstop.job_id}"
    )
    for step in backstop_steps:
        _assert_backstop_step_shape(hook_id, step)
        if isinstance(backstop, ValidatorBackstop):
            parsed = _shell_run(str(step["run"]))
            assert len(parsed.heredocs) == 1, (
                f"{hook_id} validator runner must have exactly one heredoc"
            )
            heredoc = parsed.heredocs[0]
            assert heredoc.delimiter == "VALIDATORS" and heredoc.terminated, (
                f"{hook_id} validator runner has no complete VALIDATORS heredoc"
            )
            specs = parse_specs(heredoc.body)
            assert any(
                spec.module == backstop.module and spec.args == backstop.args
                for spec in specs
            ), (
                f"{hook_id} has no validator spec with module {backstop.module!r} "
                f"and exact args {backstop.args!r}"
            )

    assert "if" not in job, f"{backstop.job_id} acquired a job-level condition"
    assert not _is_true(job.get("continue-on-error", False))
    checked_steps = [
        step
        for step in _steps(job)
        if str(step.get("name", "")) not in ADVISORY_STEP_NAMES
    ]
    assert checked_steps, f"{backstop.job_id} has no non-advisory steps to check"
    assert all(
        not _is_true(step.get("continue-on-error", False)) for step in checked_steps
    ), (
        f"a non-advisory step of {backstop.job_id} is continue-on-error, so "
        f"{hook_id}'s whole-tree run can fail silently"
    )
    workflow = _workflow()
    _assert_no_default_run_shell(workflow, "workflow")
    _assert_no_default_run_shell(job, f"job {backstop.job_id}")

    job_name = str(job["name"])
    assert job_name in STRICT_GATE_JOBS
    assert job_name not in SOFT_ALLOWLIST


def test_this_pin_is_selected_for_changes_to_its_inputs() -> None:
    for changed_files in (
        [PRE_COMMIT_CONFIG_PATH],
        [".github/workflows/ci.yml"],
    ):
        selection = compute_selection(
            changed_files=changed_files,
            adjacency_path=ADJ,
            ref_name="dev",
            event_name="pull_request",
            feature_flag_enabled=True,
        )
        assert selection.is_full_suite or any(
            THIS_TEST.startswith(selected_path)
            for selected_path in selection.selected_paths
        ), (
            f"{changed_files!r} does not select {THIS_TEST}: "
            f"{selection.selected_paths!r}"
        )


def test_pytest_job_does_not_path_gate_this_pin() -> None:
    condition = str(_job("test-parallel").get("if", "")).lower()
    for path_signal in ("path", ".pre-commit-config", "ci.yml"):
        assert path_signal not in condition, (
            "test-parallel path-gates the OMN-19612 pin via its job-level "
            f"condition: {condition!r}"
        )


@pytest.mark.parametrize("step_name", PYTEST_STEP_NAMES)
def test_pytest_invocation_cannot_deselect_this_pin(step_name: str) -> None:
    matching_steps = [
        step for step in _steps(_job("test-parallel")) if step.get("name") == step_name
    ]
    assert len(matching_steps) == 1, (
        f"test-parallel must have exactly one {step_name!r} step"
    )
    tokens = _pytest_tokens(matching_steps[0])

    assert not _has_option(tokens, ("-k", "--keyword")), (
        f"{step_name} can keyword-deselect {THIS_TEST}: {tokens!r}"
    )
    assert not _has_option(tokens, ("--deselect",)), (
        f"{step_name} can explicitly deselect {THIS_TEST}: {tokens!r}"
    )
    for ignored in _option_values(tokens, ("--ignore", "--ignore-glob")):
        assert not _ignore_covers_this_test(ignored), (
            f"{step_name} ignores {THIS_TEST} via {ignored!r}"
        )
    for marker_expression in _option_values(tokens, ("-m", "--markers")):
        collapsed = _normalise_whitespace(marker_expression).lower()
        assert "not unit" not in collapsed, (
            f"{step_name} deselects this test's unit marker: {marker_expression!r}"
        )

    addopts = _pyproject_addopts_tokens()
    assert not _has_option(addopts, ("-k", "--keyword", "--deselect")), (
        f"pytest addopts can deselect {THIS_TEST}: {addopts!r}"
    )
    for ignored in _option_values(addopts, ("--ignore", "--ignore-glob")):
        assert not _ignore_covers_this_test(ignored), (
            f"pytest addopts ignores {THIS_TEST} via {ignored!r}"
        )

    pytest_options = _load_toml(PYPROJECT)["tool"]["pytest"]["ini_options"]
    testpaths = pytest_options.get("testpaths")
    if testpaths is not None:
        assert isinstance(testpaths, list) and "tests" in testpaths, (
            "pytest testpaths no longer includes the tests root"
        )


def test_required_summary_and_workflow_trigger_cannot_drop_the_backstops() -> None:
    workflow = _workflow()
    triggers = workflow[True] if True in workflow else workflow["on"]
    pull_request = triggers.get("pull_request") or {}
    assert "paths" not in pull_request
    assert "paths-ignore" not in pull_request

    summary = _job("ci-summary")
    assert summary["name"] == "CI Summary"
    assert "needs" not in summary
    required = _load_yaml(REQUIRED_CHECKS)
    required_names = {
        gate["name"] for gate in required["gates"] if gate.get("mode") == "REQUIRED"
    }
    assert "CI Summary" in required_names
