# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Lint job's one-process validator runner and its diff-scoped Plugin* guard (OMN-19613).

Two changes to ``.github/workflows/ci.yml``'s ``Lint`` job, which every test
shard waits for:

* fourteen ``omnibase_infra.validators`` steps became one step that runs them
  all in one interpreter (``scripts/ci/run_validators_in_process.py``);
* the ``no-plugin-daemon-classes`` pre-commit step scans only the PR's files on
  ``pull_request`` (``scripts/ci/precommit_diff_scope.py``), and the whole tree
  otherwise.

What these tests pin: the runner keeps each validator's exit semantics, never
stops early and reports each failure separately; the workflow cannot grow a
validator step outside the runner; and the scope falls back to every file in
each case the ticket names.
"""

from __future__ import annotations

import io
import subprocess
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from scripts.ci import precommit_diff_scope as scope_mod
from scripts.ci import run_validators_in_process as runner

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RUNNER_SCRIPT = "scripts/ci/run_validators_in_process.py"
SCOPE_SCRIPT = "scripts/ci/precommit_diff_scope.py"
EXPECTED_VALIDATOR_COUNT = 14


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _module(main: Callable[[list[str]], object] | None) -> types.ModuleType:
    mod = types.ModuleType("fake")
    if main is not None:
        mod.main = main  # type: ignore[attr-defined]
    return mod


def _importer(table: dict[str, types.ModuleType]) -> Callable[[str], object]:
    def _import(name: str) -> object:
        if name not in table:
            raise ModuleNotFoundError(name)
        return table[name]

    return _import


def _spec(name: str, *args: str, title: str | None = None) -> runner.ValidatorSpec:
    return runner.ValidatorSpec(
        title=title or name, module=f"omnibase_infra.validators.{name}", args=args
    )


def _raise(exc: BaseException) -> Callable[[list[str]], object]:
    def _main(argv: list[str]) -> object:
        raise exc

    return _main


def _lint_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["lint"]["steps"]
    assert isinstance(steps, list)
    return steps


def _runner_step_heredoc() -> str:
    steps = [s for s in _lint_steps() if RUNNER_SCRIPT in str(s.get("run", ""))]
    assert len(steps) == 1, "exactly one Lint step must invoke the validator runner"
    script = str(steps[0]["run"])
    _head, sep, rest = script.partition("<<'VALIDATORS'\n")
    assert sep, "the runner step must feed its list through a quoted VALIDATORS heredoc"
    body, sep, _ = rest.partition("\nVALIDATORS")
    assert sep, "the VALIDATORS heredoc is not terminated"
    return body


# --------------------------------------------------------------------------- #
# AC1: one interpreter, original arguments, exit semantics, never stops early
# --------------------------------------------------------------------------- #


def test_parse_specs_reads_title_module_and_args() -> None:
    specs = runner.parse_specs(
        "# comment\n\n"
        "Gate A (OMN-1) | omnibase_infra.validators.a --max-violations 74 src/x\n"
        "Gate B | omnibase_infra.validators.b .\n"
    )
    assert specs == [
        runner.ValidatorSpec(
            "Gate A (OMN-1)",
            "omnibase_infra.validators.a",
            ("--max-violations", "74", "src/x"),
        ),
        runner.ValidatorSpec("Gate B", "omnibase_infra.validators.b", (".",)),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "no pipe omnibase_infra.validators.a",
        " | omnibase_infra.validators.a",
        "Title |   ",
        "Title | omnibase_core.validators.x src",
        "Title | os.system rm",
        "# only a comment\n\n",
    ],
)
def test_parse_specs_refuses_a_malformed_list(text: str) -> None:
    with pytest.raises(runner.SpecError):
        runner.parse_specs(text)


def test_run_all_runs_every_validator_after_a_failure_and_passes_args() -> None:
    seen: list[tuple[str, list[str]]] = []

    def recorder(name: str, code: int) -> types.ModuleType:
        def _main(argv: list[str]) -> int:
            seen.append((name, argv))
            return code

        return _module(_main)

    table = {
        "omnibase_infra.validators.first": recorder("first", 1),
        "omnibase_infra.validators.second": recorder("second", 0),
        "omnibase_infra.validators.third": recorder("third", 2),
    }
    specs = [
        _spec("first", "src/omnibase_infra"),
        _spec("second", "--max-violations", "0", "src"),
        _spec("third"),
    ]
    results = runner.run_all(
        specs, importer=_importer(table), stdout=io.StringIO(), stderr=io.StringIO()
    )
    assert seen == [
        ("first", ["src/omnibase_infra"]),
        ("second", ["--max-violations", "0", "src"]),
        ("third", []),
    ]
    assert [r.exit_code for r in results] == [1, 0, 2]


@pytest.mark.parametrize(
    ("main", "expected"),
    [
        (lambda argv: 0, 0),
        (lambda argv: 3, 3),
        (lambda argv: None, 0),
        (_raise(SystemExit(None)), 0),
        (_raise(SystemExit(0)), 0),
        (_raise(SystemExit(2)), 2),
        (_raise(SystemExit("fatal: message")), 1),
        (_raise(RuntimeError("boom")), 1),
        (lambda argv: "not an int", 1),
        (lambda argv: True, 1),
    ],
)
def test_run_one_matches_python_dash_m_exit_semantics(
    main: Callable[[list[str]], object], expected: int
) -> None:
    err = io.StringIO()
    code = runner.run_one(
        _spec("v"),
        importer=_importer({"omnibase_infra.validators.v": _module(main)}),
        stderr=err,
    )
    assert code == expected


def test_run_one_fails_an_unimportable_module_and_a_module_without_main() -> None:
    err = io.StringIO()
    assert runner.run_one(_spec("missing"), importer=_importer({}), stderr=err) == 1
    assert "ModuleNotFoundError" in err.getvalue()
    no_main = _importer({"omnibase_infra.validators.nomain": _module(None)})
    assert runner.run_one(_spec("nomain"), importer=no_main, stderr=err) == 1


def test_run_one_restores_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["outer"])
    captured: list[list[str]] = []

    def _main(argv: list[str]) -> int:
        captured.append(list(sys.argv))
        raise SystemExit(4)

    code = runner.run_one(
        _spec("v", "a"),
        importer=_importer({"omnibase_infra.validators.v": _module(_main)}),
        stderr=io.StringIO(),
    )
    assert code == 4
    assert captured == [["omnibase_infra.validators.v", "a"]]
    assert sys.argv == ["outer"]


def test_real_validator_gives_the_same_exit_code_in_process_as_its_main(
    tmp_path: Path,
) -> None:
    """Positive control against a real module: the budget gate fails on a tree
    over budget and passes under it, identically through the runner."""
    from omnibase_infra.validators import type_ignore_budget

    (tmp_path / "m.py").write_text(
        "x = foo(bar)  # type: ignore[arg-type]\ny = obj.attr  # type: ignore[union-attr]\n",
        encoding="utf-8",
    )
    for budget, expected in (("1", 1), ("5", 0)):
        args = ["--max-violations", budget, str(tmp_path)]
        assert type_ignore_budget.main(args) == expected
        spec = runner.ValidatorSpec(
            "budget", "omnibase_infra.validators.type_ignore_budget", tuple(args)
        )
        assert runner.run_one(spec, stderr=io.StringIO()) == expected


# --------------------------------------------------------------------------- #
# AC2: each failure reported separately, by name, as its own annotation
# --------------------------------------------------------------------------- #


def test_annotation_per_failure_titled_with_the_step_name() -> None:
    table = {
        "omnibase_infra.validators.ok": _module(lambda argv: 0),
        "omnibase_infra.validators.bad1": _module(lambda argv: 1),
        "omnibase_infra.validators.bad2": _module(_raise(SystemExit(2))),
    }
    out = io.StringIO()
    runner.run_all(
        [
            _spec("ok", title="Passing gate"),
            _spec("bad1", "src", title="Check handler Any signature gate (OMN-10820)"),
            _spec("bad2", title="Gate: with, specials"),
        ],
        importer=_importer(table),
        stdout=out,
        stderr=io.StringIO(),
    )
    errors = [
        line for line in out.getvalue().splitlines() if line.startswith("::error")
    ]
    assert errors == [
        "::error title=Check handler Any signature gate (OMN-10820)::"
        "omnibase_infra.validators.bad1 exited with code 1 (args: src)",
        "::error title=Gate%3A with%2C specials::omnibase_infra.validators.bad2 exited with code 2 (args: none)",
    ]


def test_annotation_run_keeps_validator_output_inside_its_log_group() -> None:
    def _main(argv: list[str]) -> int:
        print("VIOLATION in src/x.py:3")
        return 1

    out = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(out):
        runner.run_all(
            [_spec("v", title="Gate V")],
            importer=_importer({"omnibase_infra.validators.v": _module(_main)}),
            stdout=out,
            stderr=io.StringIO(),
        )
    lines = out.getvalue().splitlines()
    assert (
        lines.index("::group::Gate V")
        < lines.index("VIOLATION in src/x.py:3")
        < lines.index("::endgroup::")
    )


def test_annotation_main_exit_code_and_step_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    fail_table = {
        "omnibase_infra.validators.a": _module(lambda argv: 0),
        "omnibase_infra.validators.b": _module(lambda argv: 1),
    }
    monkeypatch.setattr("importlib.import_module", _importer(fail_table))
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "A | omnibase_infra.validators.a\nB | omnibase_infra.validators.b\n"
        ),
    )
    assert runner.main([]) == 1
    text = summary.read_text(encoding="utf-8")
    assert "| B | `omnibase_infra.validators.b` | **1** |" in text
    assert "2 validators, 1 failed" in text

    monkeypatch.setattr("sys.stdin", io.StringIO("A | omnibase_infra.validators.a\n"))
    assert runner.main([]) == 0


def test_annotation_on_a_malformed_list_fails_the_step(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Title | os.system\n"))
    assert runner.main([]) == 1
    assert "::error title=Validator list::" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# AC3: the workflow and the runner cannot drift apart
# --------------------------------------------------------------------------- #


def test_ci_yml_runner_list_has_the_fourteen_validators_and_each_is_real() -> None:
    specs = runner.parse_specs(_runner_step_heredoc())
    assert len(specs) == EXPECTED_VALIDATOR_COUNT
    assert len({s.module for s in specs}) == EXPECTED_VALIDATOR_COUNT
    for spec in specs:
        rel = Path("src", *spec.module.split(".")).with_suffix(".py")
        assert (REPO_ROOT / rel).is_file(), f"{spec.module} does not exist at {rel}"
        assert "def main(" in (REPO_ROOT / rel).read_text(encoding="utf-8"), (
            f"{spec.module} has no main()"
        )


def test_ci_yml_has_no_validator_step_outside_the_runner() -> None:
    """A new ``python -m omnibase_infra.validators.X`` step anywhere in ci.yml
    re-introduces the per-step interpreter cost; it belongs in the heredoc."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []) if isinstance(job, dict) else []:
            run = str(step.get("run", ""))
            if "-m omnibase_infra.validators." in run:
                offenders.append(f"{job_name}: {step.get('name')}")
    assert offenders == []


def test_ci_yml_runner_step_sets_pythonpath_src() -> None:
    assert _runner_step_heredoc()  # the step exists
    step = next(s for s in _lint_steps() if RUNNER_SCRIPT in str(s.get("run", "")))
    assert (
        str(step["run"])
        .lstrip()
        .startswith("PYTHONPATH=src uv run python " + RUNNER_SCRIPT)
    )


# --------------------------------------------------------------------------- #
# AC4: no-plugin-daemon-classes is diff-scoped on PRs, full otherwise
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(),
    ).stdout


def _lock(core_version: str, other_version: str = "1.0.0") -> str:
    return (
        "version = 1\n\n"
        f'[[package]]\nname = "omnibase-core"\nversion = "{core_version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n\n'
        f'[[package]]\nname = "pydantic"\nversion = "{other_version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )


def _repo_with_pr(
    tmp_path: Path, pr_files: dict[str, str]
) -> tuple[Path, Callable[[Sequence[str]], str]]:
    """A repo whose HEAD is a two-parent merge commit, like GitHub's PR checkout."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "uv.lock").write_text(_lock("0.47.23"), encoding="utf-8")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "pr")
    for name, text in pr_files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pr")
    _git(repo, "checkout", "-q", "dev")
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base moved")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge pr", "pr")

    def git(args: Sequence[str]) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            env=scrub_git_location_env(),
        )
        if proc.returncode != 0:
            raise scope_mod.GitError(proc.stderr)
        return proc.stdout

    return repo, git


@pytest.mark.parametrize(
    "event", ["push", "merge_group", "workflow_dispatch", "schedule"]
)
def test_plugin_daemon_scope_is_full_tree_off_pull_request(
    tmp_path: Path, event: str
) -> None:
    _, git = _repo_with_pr(tmp_path, {"src/c.py": "z = 3\n"})
    assert scope_mod.decide(event, git).args == ("--all-files",)


def test_plugin_daemon_scope_is_the_pr_diff_on_pull_request(tmp_path: Path) -> None:
    repo, git = _repo_with_pr(tmp_path, {"src/c.py": "z = 3\n"})
    scope = scope_mod.decide("pull_request", git)
    base = _git(repo, "rev-parse", "HEAD^1").strip()
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert scope.args == ("--from-ref", base, "--to-ref", head)
    # The range is the PR's own change: the file the base moved on is not in it.
    assert _git(repo, "diff", "--name-only", f"{base}...{head}").split() == ["src/c.py"]


@pytest.mark.parametrize(
    "path", [".pre-commit-config.yaml", ".github/workflows/ci.yml"]
)
def test_plugin_daemon_scope_is_full_tree_when_the_hook_config_changes(
    tmp_path: Path, path: str
) -> None:
    _, git = _repo_with_pr(tmp_path, {path: "changed: true\n", "src/c.py": "z = 3\n"})
    assert scope_mod.decide("pull_request", git).args == ("--all-files",)


def test_plugin_daemon_scope_is_full_tree_when_the_core_pin_moves(
    tmp_path: Path,
) -> None:
    _, git = _repo_with_pr(tmp_path, {"uv.lock": _lock("0.47.24")})
    assert scope_mod.decide("pull_request", git).args == ("--all-files",)


def test_plugin_daemon_scope_stays_scoped_when_another_lock_entry_moves(
    tmp_path: Path,
) -> None:
    _, git = _repo_with_pr(
        tmp_path, {"uv.lock": _lock("0.47.23", other_version="2.0.0")}
    )
    assert scope_mod.decide("pull_request", git).args[0] == "--from-ref"


def test_plugin_daemon_scope_fails_closed_when_head_is_not_a_merge_commit(
    tmp_path: Path,
) -> None:
    repo, git = _repo_with_pr(tmp_path, {"src/c.py": "z = 3\n"})
    _git(
        repo, "checkout", "-q", "pr"
    )  # a single-parent head, as a non-merge checkout would be
    assert scope_mod.decide("pull_request", git).args == ("--all-files",)


def test_plugin_daemon_scope_fails_closed_when_git_cannot_answer() -> None:
    def broken(args: Sequence[str]) -> str:
        raise scope_mod.GitError("fatal: not a git repository")

    scope = scope_mod.decide("pull_request", broken)
    assert scope.args == ("--all-files",)
    assert "failing closed" in scope.reason


def test_plugin_daemon_scope_fails_closed_on_an_unreadable_lockfile(
    tmp_path: Path,
) -> None:
    _, git = _repo_with_pr(tmp_path, {"uv.lock": "this is [ not toml\n"})
    assert scope_mod.decide("pull_request", git).args == ("--all-files",)


def test_plugin_daemon_ci_yml_step_uses_the_scope_and_a_depth_two_checkout() -> None:
    steps = _lint_steps()
    checkout = steps[0]
    assert str(checkout.get("uses", "")).startswith("actions/checkout@")
    assert checkout.get("with", {}).get("fetch-depth") == 2
    guard = [s for s in steps if "no-plugin-daemon-classes" in str(s.get("run", ""))]
    assert len(guard) == 1, (
        "the Plugin* guard step must exist exactly once and always run"
    )
    run = str(guard[0]["run"])
    assert SCOPE_SCRIPT in run
    assert "--all-files" not in run, (
        "the full-tree fallback belongs to the scope script"
    )
    assert "if" not in guard[0], (
        "the step must run on every event; only its file list narrows"
    )
    assert guard[0].get("env", {}).get("EVENT_NAME") == "${{ github.event_name }}"
