# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the inline-workflow import checker (OMN-18684).

WHAT THIS GUARDS. A ``run:`` heredoc that imports a repository module is a call
site no rename tool can see. ``dev-lane-liveness.yml``'s saturation monitor
imported ``load_route_policy``; OMN-18412 renamed it to
``load_contract_policy``; the monitor went red on its next scheduled run and
stayed red, producing no saturation record at all across runs 35337065964,
35336042566 and 35335179281 before anyone read a log.

THE LOAD-BEARING TEST is ``test_a_renamed_function_is_caught_in_a_fixture``:
the OMN-18684 defect reconstructed from scratch in a fixture repository, which
fails the checker, against a positive control differing only in the name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "check_workflow_inline_python_imports.py"

_spec = importlib.util.spec_from_file_location(
    "check_workflow_inline_python_imports", MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = checker
_spec.loader.exec_module(checker)


def _fixture_repo(root: Path, *, exported: str, imported: str) -> Path:
    """A miniature repository in the shape of the real defect.

    ``exported`` is the name the module defines; ``imported`` is the name the
    workflow's heredoc asks for. They are equal in the control case and differ
    in the reconstruction of OMN-18684.
    """
    (root / "scripts/ci").mkdir(parents=True)
    (root / "scripts/ci/runner_route_decision.py").write_text(
        f"def probe_fleet(token, group, api):\n"
        f"    return {{}}\n"
        f"\n"
        f"\n"
        f"def {exported}(path=None):\n"
        f"    return 'omnibase-ci'\n",
        encoding="utf-8",
    )
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/dev-lane-liveness.yml").write_text(
        "name: dev-lane-liveness\n"
        "on:\n"
        "  schedule:\n"
        "    - cron: '*/30 * * * *'\n"
        "jobs:\n"
        "  saturation-record:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Probe the org runner registry\n"
        "        run: |\n"
        "          python3 - <<'PYEOF'\n"
        "          import sys\n"
        '          sys.path.insert(0, "scripts/ci")\n'
        f"          from runner_route_decision import probe_fleet, {imported}\n"
        "          PYEOF\n",
        encoding="utf-8",
    )
    return root


def test_a_renamed_function_is_caught_in_a_fixture(tmp_path: Path) -> None:
    """AC3's falsifier, reconstructed: the module exports the renamed name and
    the workflow still asks for the old one. The checker must name the
    workflow, the job, the step, the module and the missing name -- enough to
    fix it without opening a CI log.
    """
    root = _fixture_repo(
        tmp_path / "renamed",
        exported="load_contract_policy",
        imported="load_route_policy",
    )
    violations = [
        v
        for path in checker.discover(root)
        for v in checker.check_file(path, root=root)
    ]
    assert len(violations) == 1
    rendered = violations[0].render()
    assert "dev-lane-liveness.yml" in rendered
    assert "saturation-record" in rendered
    assert "Probe the org runner registry" in rendered
    assert "load_route_policy" in rendered
    assert checker.main(["--root", str(root)]) == 1


def test_the_same_fixture_with_the_right_name_is_clean(tmp_path: Path) -> None:
    """Positive control for the test above: identical in every respect except
    that the workflow asks for the name the module actually exports. Without
    this, a checker that flagged every import would pass the test above while
    being useless.
    """
    root = _fixture_repo(
        tmp_path / "matching",
        exported="load_contract_runner_group",
        imported="load_contract_runner_group",
    )
    assert checker.main(["--root", str(root)]) == 0


def test_the_live_repository_is_clean() -> None:
    """The gate's actual verdict on this checkout. Red here means a workflow
    heredoc and its module have drifted apart, which is the defect itself and
    not a problem with the test.
    """
    violations = [
        v
        for path in checker.discover(REPO_ROOT)
        for v in checker.check_file(path, root=REPO_ROOT)
    ]
    assert violations == [], "\n".join(v.render() for v in violations)


def test_the_saturation_step_is_covered_rather_than_merely_clean() -> None:
    """A clean verdict means nothing if the step is not being read.

    Pins that the checker resolves the real ``saturation-record`` heredoc to
    the real module, so a future edit that makes the step invisible to the
    checker -- dropping the ``sys.path.insert``, moving the module -- fails
    here instead of silently un-gating it.
    """
    workflow = REPO_ROOT / ".github/workflows/dev-lane-liveness.yml"
    source = (REPO_ROOT / "scripts/ci/runner_route_decision.py").read_text(
        encoding="utf-8"
    )
    defined = checker.module_level_names(source)
    assert "load_contract_runner_group" in defined
    assert "load_route_policy" not in defined

    text = workflow.read_text(encoding="utf-8")
    names = {
        name
        for module, clause in checker._FROM_IMPORT.findall(text)
        if module == "runner_route_decision"
        for name in checker.imported_names(clause)
    }
    assert names, "the checker's import regex no longer matches the live step"
    assert names <= defined


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        (
            "probe_fleet, load_contract_runner_group",
            ("probe_fleet", "load_contract_runner_group"),
        ),
        ("probe_fleet as pf", ("probe_fleet",)),
        ("a, b, c", ("a", "b", "c")),
    ],
)
def test_import_clauses_resolve_to_the_names_that_must_exist(
    clause: str, expected: tuple[str, ...]
) -> None:
    """An ``as`` alias renames the binding locally; the name that must exist
    upstream is the one on the left, and checking the alias instead would miss
    exactly the rename this gate is for.
    """
    assert checker.imported_names(clause) == expected


def test_a_stdlib_import_is_skipped_rather_than_guessed_at(tmp_path: Path) -> None:
    """Scope, asserted: only modules resolving to a repository file are read.
    A checker that reported ``from json import loads`` would be noise, and
    noisy checkers get turned off (CLAUDE.md rule 5).
    """
    root = tmp_path / "stdlib"
    (root / "scripts/ci").mkdir(parents=True)
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/w.yml").write_text(
        "jobs:\n"
        "  j:\n"
        "    steps:\n"
        "      - name: s\n"
        "        run: |\n"
        "          import sys\n"
        '          sys.path.insert(0, "scripts/ci")\n'
        "          from json import loads, dumps\n",
        encoding="utf-8",
    )
    assert checker.main(["--root", str(root)]) == 0


def test_a_step_with_no_search_path_resolves_nothing(tmp_path: Path) -> None:
    """Search paths are read from the step, never assumed. A step that inserts
    nothing cannot import a repository module either, so reporting one would
    be a false finding.
    """
    root = tmp_path / "nopath"
    (root / "scripts/ci").mkdir(parents=True)
    (root / "scripts/ci/runner_route_decision.py").write_text(
        "def probe_fleet():\n    return {}\n", encoding="utf-8"
    )
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/w.yml").write_text(
        "jobs:\n"
        "  j:\n"
        "    steps:\n"
        "      - name: s\n"
        "        run: |\n"
        "          from runner_route_decision import gone\n",
        encoding="utf-8",
    )
    assert checker.main(["--root", str(root)]) == 0


def test_module_level_names_covers_every_importable_shape() -> None:
    """Constants and re-exported imports are importable too; a checker that
    saw only ``def`` would flag ``from m import SOME_CONSTANT`` as missing.
    """
    names = checker.module_level_names(
        "import os\n"
        "from pathlib import Path\n"
        "CONSTANT = 1\n"
        "ANNOTATED: int = 2\n"
        "def fn():\n    pass\n"
        "class Cls:\n    pass\n"
        "def _inner_only():\n    LOCAL = 3\n"
    )
    assert {"os", "Path", "CONSTANT", "ANNOTATED", "fn", "Cls"} <= names
    assert "LOCAL" not in names


def test_composite_action_steps_are_read(tmp_path: Path) -> None:
    """``.github/actions/*/action.yml`` carries ``runs.steps`` rather than
    ``jobs``, and a heredoc there drifts the same way.
    """
    root = tmp_path / "action"
    (root / "scripts/ci").mkdir(parents=True)
    (root / "scripts/ci/mod.py").write_text("def kept():\n    pass\n", encoding="utf-8")
    (root / ".github/actions/thing").mkdir(parents=True)
    (root / ".github/actions/thing/action.yml").write_text(
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - name: s\n"
        "      shell: bash\n"
        "      run: |\n"
        "        import sys\n"
        '        sys.path.insert(0, "scripts/ci")\n'
        "        from mod import gone\n",
        encoding="utf-8",
    )
    assert checker.main(["--root", str(root)]) == 1
