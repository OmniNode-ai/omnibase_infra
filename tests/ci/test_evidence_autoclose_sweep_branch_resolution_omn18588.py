# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate for the OMN-18588 cwd-branch resolution in the sweep job.

Why this file exists
--------------------
OMN-16902 taught the sweep to materialise every repository an OCC contract's
``cwd`` names. It cloned each one at a hardcoded ``--branch dev``. That is a
fact about most of the registry and not a fact about repositories, and the
difference was invisible for as long as every declared cwd happened to name a
repository whose default branch is ``dev``.

``knowledge-base-internal`` has no ``dev`` branch at all. The clone therefore
failed, and it failed into the catch-all arm whose status reads
``UNAVAILABLE_REPO_NOT_READABLE_BY_TOKEN``. That string names a credential
problem. The credential was never the problem: the App the sweep mints its
token from is installed on ALL repositories in the organisation. Three
separate readings of the sweep log concluded the token was too narrow, and the
remediation those readings produced was to move the probe to a different
repository -- nine days spent relocating evidence to work around a branch
name.

So this gate pins two things, and the second matters as much as the first:

* a repository whose default branch is ``main`` is cloned at ``main``;
* a repository whose branch CANNOT be resolved is reported under its own
  status and is never silently given ``dev``. An invented default is how a
  wrong branch becomes a wrong behaviour verdict, and a verdict grounded in a
  branch nobody chose is worse than no verdict.

Pinned by EXECUTION, not by string match, for the reason the OMN-16902 gate
beside this one already states: the workflow delimits its resolver, and this
module extracts that exact program out of the shipped YAML and runs it. A
resolver asserted by text is one silent edit away from re-hardcoding the
branch it was written to stop hardcoding.

Ticket: OMN-18588
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

_BEGIN_MARKER = "# ---8<--- OMN-18588 BEGIN cwd-branch-resolution"
_END_MARKER = "# ---8<--- OMN-18588 END cwd-branch-resolution"

_MATERIALISE_STEP = "Derive and materialise the cwd repo set the OCC contracts name"

# The status a row whose branch could not be resolved must carry. Asserted as a
# literal because it is the operator-visible string: the whole cost of this
# ticket was a status string that named the wrong cause.
_UNRESOLVED_STATUS = "UNRESOLVED_DEFAULT_BRANCH"


@pytest.fixture(scope="module")
def steps() -> list[dict[str, object]]:
    loaded = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    jobs = loaded["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["evidence-autoclose-sweep"]
    assert isinstance(job, dict)
    declared = job["steps"]
    assert isinstance(declared, list)
    return declared


def _step_by_name(steps: list[dict[str, object]], fragment: str) -> dict[str, object]:
    matches = [s for s in steps if fragment in str(s.get("name", ""))]
    assert matches, (
        f"no step in evidence-autoclose-sweep.yml has a name containing "
        f"{fragment!r}; declared steps: {[s.get('name') for s in steps]}"
    )
    assert len(matches) == 1, f"{fragment!r} matched {len(matches)} steps"
    return matches[0]


@pytest.fixture(scope="module")
def materialise_run(steps: list[dict[str, object]]) -> str:
    return str(_step_by_name(steps, _MATERIALISE_STEP)["run"])


@pytest.fixture(scope="module")
def resolution_program(materialise_run: str) -> str:
    """The branch resolver the shipped workflow actually runs, verbatim."""
    assert _BEGIN_MARKER in materialise_run and _END_MARKER in materialise_run, (
        "the materialise step must delimit its cwd-branch resolver with "
        f"{_BEGIN_MARKER!r} / {_END_MARKER!r} so this gate can execute the "
        "exact program the runner executes rather than paraphrasing it."
    )
    body = materialise_run.split(_BEGIN_MARKER, 1)[1].split(_END_MARKER, 1)[0]
    assert not body.lstrip("\n").startswith(" "), (
        "the resolver must sit at column 0 inside the run script (a nested "
        "heredoc would not terminate); found leading indentation."
    )
    return body


def _resolve(
    program: str,
    tmp_path: Path,
    dests: list[tuple[str, str]],
    default_branches: dict[str, str],
) -> tuple[list[tuple[str, str, str]], str]:
    """Run the extracted resolver; return its parsed stdout rows and stderr."""
    prog = tmp_path / "resolve_cwd_branches.py"
    prog.write_text(program, encoding="utf-8")

    dests_tsv = tmp_path / "cwd-dests.tsv"
    dests_tsv.write_text(
        "".join(f"{dest}\t{repo}\n" for dest, repo in dests), encoding="utf-8"
    )
    branch_map = tmp_path / "default-branches.json"
    branch_map.write_text(json.dumps(default_branches), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(prog), str(dests_tsv), str(branch_map)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"the workflow's cwd-branch resolver exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    rows: list[tuple[str, str, str]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        assert len(parts) == 3, (
            f"every emitted row must be <dest>\\t<repo>\\t<branch>; got {line!r}"
        )
        rows.append((parts[0], parts[1], parts[2]))
    return rows, proc.stderr


def test_resolution_block_is_extractable(resolution_program: str) -> None:
    assert resolution_program.strip(), "the marked resolver block is empty"


def test_a_repo_whose_default_is_main_resolves_main(
    resolution_program: str, tmp_path: Path
) -> None:
    """The defect, stated as a test.

    This is the exact row that has been failing on every scheduled run: a cwd
    in a repository whose only branch is ``main``.
    """
    rows, stderr = _resolve(
        resolution_program,
        tmp_path,
        dests=[("knowledge-base-internal", "knowledge-base-internal")],
        default_branches={"knowledge-base-internal": "main"},
    )
    assert rows == [("knowledge-base-internal", "knowledge-base-internal", "main")], (
        "a repository whose default branch is `main` must be cloned at `main`; "
        f"resolver emitted {rows!r} (stderr: {stderr!r})"
    )


def test_a_repo_whose_default_is_dev_still_resolves_dev(
    resolution_program: str, tmp_path: Path
) -> None:
    """No regression for the registry majority, which really is on ``dev``."""
    rows, _ = _resolve(
        resolution_program,
        tmp_path,
        dests=[("omnibase_infra", "omnibase_infra")],
        default_branches={"omnibase_infra": "dev"},
    )
    assert rows == [("omnibase_infra", "omnibase_infra", "dev")]


def test_mixed_corpus_resolves_each_repo_independently(
    resolution_program: str, tmp_path: Path
) -> None:
    """One run, two default branches. A single global branch cannot pass this."""
    rows, _ = _resolve(
        resolution_program,
        tmp_path,
        dests=[
            ("knowledge-base-internal", "knowledge-base-internal"),
            ("omnibase_infra", "omnibase_infra"),
            ("omni_worktrees/OMN-16764/omniintelligence", "omniintelligence"),
        ],
        default_branches={
            "knowledge-base-internal": "main",
            "omnibase_infra": "dev",
            "omniintelligence": "dev",
        },
    )
    assert {dest: branch for dest, _repo, branch in rows} == {
        "knowledge-base-internal": "main",
        "omnibase_infra": "dev",
        "omni_worktrees/OMN-16764/omniintelligence": "dev",
    }


def test_an_unresolvable_repo_is_reported_and_never_defaulted(
    resolution_program: str, tmp_path: Path
) -> None:
    """The releasing direction is the one that must not happen.

    A repository the branch lookup could not answer for is REPORTED. It does
    not silently acquire ``dev``: a clone at a branch nobody chose produces a
    behaviour verdict about code nobody chose, which is worse than the honest
    refusal of no verdict at all.
    """
    rows, stderr = _resolve(
        resolution_program,
        tmp_path,
        dests=[("omniweb", "omniweb"), ("omnibase_infra", "omnibase_infra")],
        default_branches={"omnibase_infra": "dev"},
    )
    assert ("omniweb", "omniweb", "dev") not in rows, (
        "an unresolved repository was silently given `dev` -- the exact "
        "invented-default this ticket exists to remove"
    )
    assert [r[0] for r in rows] == ["omnibase_infra"], (
        f"only resolvable rows may be emitted on stdout; got {rows!r}"
    )
    assert f"{_UNRESOLVED_STATUS}\tomniweb\tomniweb" in stderr, (
        "an unresolvable repository must be named on stderr under "
        f"{_UNRESOLVED_STATUS!r}; stderr was {stderr!r}"
    )


def test_a_blank_default_branch_is_unresolved_not_empty(
    resolution_program: str, tmp_path: Path
) -> None:
    """A lookup that answered with nothing has not answered.

    `gh api ... --jq .default_branch` prints an empty line when the read fails
    in some shapes, so the empty string reaches this program as a value. An
    empty branch would render `--branch ''`, which git rejects with a message
    about refspecs that names neither the repository nor the cause.
    """
    rows, stderr = _resolve(
        resolution_program,
        tmp_path,
        dests=[("omnimemory", "omnimemory")],
        default_branches={"omnimemory": "   "},
    )
    assert rows == [], f"a blank branch must not be emitted as a row; got {rows!r}"
    assert f"{_UNRESOLVED_STATUS}\tomnimemory\tomnimemory" in stderr


def test_the_clone_no_longer_hardcodes_a_branch(materialise_run: str) -> None:
    """The regression guard on the shipped step, complementing the above.

    The execution tests prove the resolver is right. They cannot prove the
    clone USES it, and a correct resolver whose output nothing reads is the
    same outage with more code.
    """
    assert "--branch dev" not in materialise_run, (
        "the materialise step still clones at a hardcoded `--branch dev`; the "
        "resolved per-repository branch is what it must pass."
    )
    assert '--branch "${branch}"' in materialise_run, (
        "the clone must pass the resolved branch variable so a repository "
        "whose default is not `dev` is cloned at the branch it actually has."
    )
