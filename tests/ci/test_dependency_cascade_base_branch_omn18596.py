# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate for the OMN-18596 base-branch resolution in the cascade job.

Why this file exists
--------------------
``dependency-cascade.yml`` checks the downstream repository out with no
``ref:``, so the bump branch is cut from that repository's own default branch.
The job then opened the pull request with a hardcoded ``--base main``.

Those two are not the same ref and on this fleet they are very far apart.
``main`` is release-synced: it is the last published release, advanced only by
an automated fast-forward, and it is never a pull-request target. On
2026-09-17 the mismatch opened ``omniclaude#2215`` against ``main``, which
rendered a three-line lockfile bump as a 16,201-addition diff; a person had to
retarget it by hand at 10:36:38Z before it read as anything but a catastrophe.

This is the same defect class OMN-18588 corrected in the sweep job the same
day: a branch name baked into automation that fans out across repositories,
correct only for as long as every repository agrees.

So this gate pins two things, and the second matters as much as the first:

* the base is READ from the checkout rather than written into the workflow, so
  base and head cannot diverge -- the PR targets the exact ref its commit sits
  on top of;
* a ref that CANNOT be resolved refuses. A cascade that guesses its own base
  is how a wrong branch becomes a wrong diff.

Pinned by EXECUTION, not by string match, the way the OMN-18588 gate beside
this one is: the workflow delimits its resolver, this module extracts that
exact program out of the shipped YAML and runs it against real git
repositories. A resolver asserted by text is one silent edit away from
re-hardcoding the branch it was written to stop hardcoding.

Ticket: OMN-18596
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "dependency-cascade.yml"

_OPEN = "# >>> OMN-18596 base-branch resolver >>>"
_CLOSE = "# <<< OMN-18596 base-branch resolver <<<"


def _resolver_program() -> str:
    """Lift the delimited resolver out of the shipped workflow, dedented.

    The ``${{ matrix.repo }}`` expression inside the refusal message is a
    GitHub template, not shell. It is substituted here with a literal so the
    extracted program is runnable; nothing else is rewritten, because the point
    of the exercise is to run what ships.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")
    start = body.index(_OPEN)
    end = body.index(_CLOSE, start)
    block = body[start:end]
    lines = [
        line[10:] if line.startswith(" " * 10) else line for line in block.splitlines()
    ]
    program = "\n".join(lines)
    return re.sub(r"\$\{\{[^}]*\}\}", "a-downstream-repo", program)


def _run(cwd: Path, output_file: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", _resolver_program()],
        cwd=cwd,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "GITHUB_OUTPUT": str(output_file)},
        capture_output=True,
        text=True,
        check=False,
    )


def _git(cwd: Path, *args: str) -> None:
    """Run git against ``cwd`` and ONLY ``cwd``.

    ``env=scrub_git_location_env(...)`` is not optional (OMN-14891): git exports
    GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into every hook environment and
    those OVERRIDE both ``cwd=`` and ``git -C``, so an unscrubbed call from a
    test running under a hook mutates the real invoking worktree rather than
    the fixture.
    """
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


@pytest.fixture
def repo_on_branch(tmp_path: Path) -> Path:
    """A repository whose default branch is deliberately NOT ``dev`` or ``main``."""
    root = tmp_path / "downstream"
    root.mkdir()
    _git(root, "init", "--initial-branch", "trunk")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    _git(root, "add", "pyproject.toml")
    _git(root, "commit", "-m", "init")
    return root


class TestTheBaseIsReadFromTheCheckout:
    def test_the_workflow_no_longer_hardcodes_a_base_branch(self) -> None:
        """Asserted on the EFFECTIVE lines, never on the prose beside them.

        A gate that fires on a comment describing the gate is the failure mode
        omni_home operating rule 15 records three separate times in one window.
        Comment lines are stripped before the match, so this file's own
        explanation of the defect cannot be mistaken for the defect.
        """
        body = "\n".join(
            line
            for line in _WORKFLOW.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "--base main" not in body, (
            "the dependency cascade hardcodes --base main; main is release-synced "
            "on this fleet and is never a pull-request target, and the bump "
            "branch is cut from the downstream repo's default branch instead"
        )
        assert "--base dev" not in body, (
            "replacing one hardcoded branch with another repeats the defect; the "
            "base must be resolved from the checkout at run time"
        )
        assert "steps.vars.outputs.base" in body

    def test_a_repo_whose_default_branch_is_not_dev_is_targeted_at_its_own(
        self, repo_on_branch: Path, tmp_path: Path
    ) -> None:
        """The whole bug in one assertion: no literal can produce ``trunk``."""
        out = tmp_path / "gh_output"
        out.touch()
        result = _run(repo_on_branch, out)
        assert result.returncode == 0, result.stderr
        assert "base=trunk" in out.read_text(encoding="utf-8")

    def test_the_resolved_base_is_the_ref_the_bump_branch_is_cut_from(
        self, repo_on_branch: Path, tmp_path: Path
    ) -> None:
        """Base and head cannot diverge, because the base is head's own parent.

        Proven by resolving first and THEN cutting the bump branch, which is the
        shipped order, and checking the recorded base is an ancestor of the new
        branch's tip.
        """
        out = tmp_path / "gh_output"
        out.touch()
        assert _run(repo_on_branch, out).returncode == 0
        base = next(
            line.split("=", 1)[1]
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.startswith("base=")
        )
        _git(repo_on_branch, "checkout", "-b", "automation/bump-omnibase-infra-0.38.29")
        (repo_on_branch / "uv.lock").write_text("locked\n", encoding="utf-8")
        _git(repo_on_branch, "add", "uv.lock")
        _git(repo_on_branch, "commit", "-m", "chore(deps): bump")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, "HEAD"],
            cwd=repo_on_branch,
            check=True,
            capture_output=True,
            env=scrub_git_location_env(os.environ),
        )


class TestAnUnresolvableRefRefuses:
    def test_a_detached_head_refuses_and_writes_no_base(
        self, repo_on_branch: Path, tmp_path: Path
    ) -> None:
        """An invented default is how a wrong branch becomes a wrong diff."""
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_on_branch,
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.strip()
        _git(repo_on_branch, "checkout", "--detach", sha)
        out = tmp_path / "gh_output"
        out.touch()
        result = _run(repo_on_branch, out)
        assert result.returncode == 1
        assert "refusing to open a dependency-bump PR" in result.stdout
        assert "base=" not in out.read_text(encoding="utf-8")

    def test_the_positive_control_proves_the_refusal_is_not_vacuous(
        self, repo_on_branch: Path, tmp_path: Path
    ) -> None:
        """The same program, same fixture, attached: exit 0 and a base written."""
        out = tmp_path / "gh_output"
        out.touch()
        result = _run(repo_on_branch, out)
        assert result.returncode == 0
        assert "base=trunk" in out.read_text(encoding="utf-8")
