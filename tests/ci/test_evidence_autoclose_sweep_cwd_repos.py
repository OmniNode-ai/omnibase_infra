# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate for the OMN-16902 cwd-repo materialisation in the sweep job.

Why this file exists
--------------------
The evidence-autoclose sweep runs `onex skill dod_verify` against OCC
contracts whose behaviour checks pin ``cwd: "${OMNI_HOME}/<repo>"``. Three
layers of that path have each failed in turn, and each failure rendered as a
*false FAILED* — a verdict that names the runner, not the product:

* OMN-16845 — ``OMNI_HOME`` was unset, so every cwd rendered to ``/<repo>``.
* OMN-16846 — the co-installed verifier made the gate venv impure, so the
  OMN-15620 purity gate refused every ``uv run pytest`` before it executed.
* OMN-16902 (this) — with both fixed, only ``omnibase_infra`` actually
  existed under ``OMNI_HOME``. Run 33210339910 recorded
  ``cwd does not exist: '${OMNI_HOME}/omnimarket'`` and
  ``.../omninode_infra`` as FAILED on three of four diagnosed seeds, and the
  OCC-homed ``test_passes`` check died on ``Failed to spawn: pytest`` on all
  four.

The sweep only runs on a schedule and on ``workflow_dispatch``, so nothing at
PR time would notice if this wiring rotted — the "detection that is never
enforced" shape CLAUDE.md Rule 5 forbids. These tests are that enforcement.

The derivation is pinned by EXECUTION, not by string match. The workflow
embeds its classifier between two markers; this module extracts that exact
program out of the shipped YAML and runs it against a synthetic contract
corpus. A hardcoded repo list in the workflow cannot satisfy
``test_derivation_classifies_a_repo_it_has_never_seen`` — which is the point
of OMN-16902 AC2: a list that does not derive rots silently the first time a
new repo appears.

Ticket: OMN-16902
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

_BEGIN_MARKER = "# ---8<--- OMN-16902 BEGIN cwd-repo-derivation"
_END_MARKER = "# ---8<--- OMN-16902 END cwd-repo-derivation"

_MATERIALISE_STEP = "Derive and materialise the cwd repo set the OCC contracts name"
_OCC_PYTEST_PROOF_STEP = "Prove an OCC governance worktree can spawn pytest"
_POST_SWEEP_PURITY_STEP = "Assert the gate venv is still pure after the sweep"


@pytest.fixture(scope="module")
def workflow() -> dict[str, object]:
    loaded = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def steps(workflow: dict[str, object]) -> list[dict[str, object]]:
    jobs = workflow["jobs"]
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
def derivation_program(steps: list[dict[str, object]]) -> str:
    """The classifier the shipped workflow actually runs, verbatim."""
    run = str(_step_by_name(steps, _MATERIALISE_STEP)["run"])
    assert _BEGIN_MARKER in run and _END_MARKER in run, (
        "the materialise step must delimit its cwd-repo classifier with "
        f"{_BEGIN_MARKER!r} / {_END_MARKER!r} so this gate can execute the "
        "exact program the runner executes rather than paraphrasing it."
    )
    body = run.split(_BEGIN_MARKER, 1)[1].split(_END_MARKER, 1)[0]
    # The YAML block scalar already stripped the step's base indentation, so
    # the extracted program is at column 0 exactly as `python3 -` receives it.
    assert not body.lstrip("\n").startswith(" "), (
        "the classifier must sit at column 0 inside the run script (a nested "
        "heredoc would not terminate); found leading indentation."
    )
    return body


def _run_derivation(
    program: str, contracts_dir: Path, tmp_path: Path
) -> tuple[str, str]:
    prog = tmp_path / "derive_cwd_repos.py"
    prog.write_text(program, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(prog), str(contracts_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"the workflow's cwd-repo classifier exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc.stdout, proc.stderr


def _write_contract(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(body, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A synthetic OCC contract corpus covering every cwd shape in the wild."""
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    # Plain repo root — the OMN-16790 shape (omnimarket) that ran FAILED.
    _write_contract(
        contracts,
        "OMN-1.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "${OMNI_HOME}/omnimarket"\n',
    )
    # Unquoted form.
    _write_contract(
        contracts,
        "OMN-2.yaml",
        "dod_evidence:\n  - checks:\n      - cwd: ${OMNI_HOME}/omnibase_core\n",
    )
    # A repo the sweep already checks out — still derived, so the job can
    # prove it is present rather than assume it.
    _write_contract(
        contracts,
        "OMN-3.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "${OMNI_HOME}/omnibase_infra"\n',
    )
    # Sub-directory inside a repo: the REPO is what must be materialised.
    _write_contract(
        contracts,
        "OMN-4.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "${OMNI_HOME}/omniclaude/plugins/onex"\n',
    )
    # Operator worktree path — the terminal segment names the repo.
    _write_contract(
        contracts,
        "OMN-5.yaml",
        "dod_evidence:\n  - checks:\n"
        '      - cwd: "${OMNI_HOME}/omni_worktrees/OMN-16764/omniintelligence"\n',
    )
    # Trailing comment must not defeat the match.
    _write_contract(
        contracts,
        "OMN-6.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "${OMNI_HOME}/omnimemory"  # note\n',
    )
    # Prose that merely MENTIONS a path is not a cwd declaration.
    _write_contract(
        contracts,
        "OMN-7.yaml",
        "ac_summary: |\n  executed at ${OMNI_HOME}/omniweb by hand\n",
    )
    # Unclassifiable: traversal, bare worktrees root, non-OMNI_HOME absolute.
    _write_contract(
        contracts,
        "OMN-8.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "${OMNI_HOME}/../etc"\n',
    )
    _write_contract(
        contracts,
        "OMN-9.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "${OMNI_HOME}/omni_worktrees"\n',
    )
    _write_contract(
        contracts,
        "OMN-10.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "/opt/elsewhere/entirely"\n',
    )
    return contracts


def test_derivation_block_is_extractable(derivation_program: str) -> None:
    assert derivation_program.strip(), "the marked classifier block is empty"


def test_derivation_classifies_every_cwd_shape(
    derivation_program: str, corpus: Path, tmp_path: Path
) -> None:
    stdout, stderr = _run_derivation(derivation_program, corpus, tmp_path)
    pairs = sorted(
        tuple(line.split("\t")) for line in stdout.splitlines() if line.strip()
    )
    assert pairs == sorted(
        [
            ("omni_worktrees/OMN-16764/omniintelligence", "omniintelligence"),
            ("omnibase_core", "omnibase_core"),
            ("omnibase_infra", "omnibase_infra"),
            ("omniclaude", "omniclaude"),
            ("omnimarket", "omnimarket"),
            ("omnimemory", "omnimemory"),
        ]
    ), f"unexpected derivation\nstdout:\n{stdout}\nstderr:\n{stderr}"

    # Every unsatisfiable shape is REPORTED, never silently dropped: a cwd the
    # job cannot classify is exactly the class that renders as a false FAILED.
    assert "${OMNI_HOME}/../etc" in stderr
    assert "${OMNI_HOME}/omni_worktrees" in stderr
    assert "/opt/elsewhere/entirely" in stderr


def test_derivation_classifies_a_repo_it_has_never_seen(
    derivation_program: str, tmp_path: Path
) -> None:
    """AC2: derived, not hardcoded.

    A repo that does not exist anywhere in the registry today must still be
    derived. A workflow carrying a literal repo list would fail here — which
    is the failure mode OMN-16902 was filed about, arriving by a fourth door.
    """
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    _write_contract(
        contracts,
        "OMN-99.yaml",
        'dod_evidence:\n  - checks:\n      - cwd: "${OMNI_HOME}/omni_notyetinvented"\n',
    )
    stdout, _ = _run_derivation(derivation_program, contracts, tmp_path)
    assert stdout.splitlines() == ["omni_notyetinvented\tomni_notyetinvented"]


def test_materialise_step_targets_omni_home_and_guards_the_gate_checkout(
    steps: list[dict[str, object]],
) -> None:
    run = str(_step_by_name(steps, _MATERIALISE_STEP)["run"])
    assert '"${OMNI_HOME}/${dest}"' in run, (
        "derived dests must be materialised under OMNI_HOME — that is the "
        "only root _resolve_cwd's containment check accepts."
    )
    # A destructive refresh that could ever resolve to the gate checkout would
    # delete the job out from under itself.
    assert "GITHUB_WORKSPACE" in run and "rm -rf" in run, (
        "the step must refresh each dest destructively AND refuse any dest "
        "that resolves to GITHUB_WORKSPACE or OMNI_HOME itself."
    )


def test_occ_worktree_can_spawn_pytest_is_proven_before_the_sweep(
    steps: list[dict[str, object]],
) -> None:
    """AC3/D2 as an executable precondition, not a comment.

    The OCC-homed check does NOT run in `.occ-src`: `_infer_occ_cwd` returns
    `_occ_dev_root`, an ephemeral `git worktree` of the governance ref that
    the collector materialises mid-run. No step can pre-provision that path,
    so the proof has to be "a worktree made the same way can spawn pytest".
    """
    step = _step_by_name(steps, _OCC_PYTEST_PROOF_STEP)
    run = str(step["run"])
    assert "git" in run and "worktree" in run and "origin/dev" in run, (
        "the proof must build a worktree of the governance ref the collector "
        "actually uses, not test `.occ-src` in place."
    )
    assert "pytest" in run, "the proof must spawn pytest, which is what failed"
    names = [str(s.get("name", "")) for s in steps]
    assert names.index(step["name"]) < names.index(  # type: ignore[arg-type]
        next(n for n in names if "Run evidence autoclose sweep" in n)
    ), "the D2 proof must run BEFORE the sweep so a sweep timeout cannot swallow it"


def test_uv_no_sync_is_not_forced_onto_the_check_subprocesses(
    steps: list[dict[str, object]],
) -> None:
    """D2's mechanism, pinned.

    `UV_NO_SYNC=1` on the sweep step is inherited by every behaviour-check
    subprocess. In the OCC governance worktree — created fresh inside the run
    — it makes `uv run pytest` materialise an EMPTY `.venv` and then die on
    `error: Failed to spawn: pytest` (run 33210339910, all four seeds). The
    pre-synced dests below make dropping it close to free.
    """
    for fragment in ("Run evidence autoclose sweep", "Diagnose verdict divergence"):
        step = _step_by_name(steps, fragment)
        env = step.get("env") or {}
        assert isinstance(env, dict)
        assert "UV_NO_SYNC" not in env, (
            f"{fragment!r} still exports UV_NO_SYNC, which forbids the OCC "
            "governance worktree from provisioning the interpreter its "
            "test_passes check needs."
        )


def test_gate_venv_purity_is_re_asserted_after_the_sweep(
    steps: list[dict[str, object]],
) -> None:
    """No cross-installs into the canonical venv — proven after the fact.

    Dropping UV_NO_SYNC lets `uv run` sync projects during the sweep. Exact-mode
    `uv sync` cannot make `.venv` impure, but "cannot" is a claim; this step is
    the measurement, and it runs `if: always()` so a failing sweep does not
    hide a corrupted gate venv.
    """
    step = _step_by_name(steps, _POST_SWEEP_PURITY_STEP)
    assert step.get("if") in ("always()", "${{ always() }}"), (
        "the post-sweep purity assertion must run unconditionally"
    )
    assert "find_undeclared_onex_providers" in str(step["run"]), (
        "it must call the same predicate the OMN-15620 pytest session gate "
        "calls, not a restatement of it."
    )


def test_every_pip_install_targets_the_dispatch_venv(
    steps: list[dict[str, object]],
) -> None:
    """OMN-16846 regression guard, re-armed for the steps OMN-16902 adds."""
    for step in steps:
        run = str(step.get("run", ""))
        for line in run.splitlines():
            if "uv pip install" not in line:
                continue
            assert '--python "$PYTHON_BIN"' in line, (
                f"`uv pip install` without the dispatch-venv interpreter: {line!r}. "
                "Installing an undeclared onex.nodes provider into the gate venv "
                "is exactly what OMN-16846 removed."
            )
