# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18865: the pre-merge twin of the OMN-14631 content-parity gate.

The image build already proves the installed package tree is byte-for-byte
the staged source tree plus declared force-includes. That proof runs on the
lab, after the change has merged in the package's own repository. Twice on
2026-09-19 a change green in its own repository broke it and was found only
post-merge:

  15:37Z  omnibase_core#1710 added a hatch force-include the gate could not
          resolve, so the mapped file read as an extra installed file:
          ``differing files (1 total): ['data/gitignore-baseline.yaml']``.
          Fixed in the GATE by omnibase_infra#3846 (OMN-18847).

  21:19Z  omnimarket#2670 adopted a propagated ``.gitignore`` block declaring
          a BARE ``merge-sweep/``. A bare directory pattern matches at any
          depth, so it also matched the real, git-tracked package directory
          ``src/omnimarket/adapters/codex/skills/merge-sweep/`` and hatchling
          dropped a tracked source file from the wheel:
          ``differing files (1 total):
          ['adapters/codex/skills/merge-sweep/SKILL.md']``.
          Fixed in the PACKAGES by omnibase_core#1718 and omnimarket#2694
          (OMN-18859).

Both are decidable from the package repository alone, at pull-request time.
This module pins the check that decides them.

The replays below are HERMETIC: each constructs the wheel by hand with
``zipfile`` and hands it to the check with ``--wheel``, so they need no
network, no uv and no build backend, and they still exercise the real
comparison. Two end-to-end tests that DO build a wheel are marked ``slow``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_PATH = REPO_ROOT / "scripts" / "ci" / "check_wheel_content_parity.py"
GATE_PATH = REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"


def _load(path: Path, name: str) -> ModuleType:
    """Load a script by file location, the way the repo's other script tests do."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check() -> ModuleType:
    return _load(CHECK_PATH, "check_wheel_content_parity_under_test")


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    return _load(GATE_PATH, "compute_workspace_provenance_under_test")


# ---------------------------------------------------------------------------
# Single source of truth
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_check_imports_the_gates_own_comparison_and_does_not_reimplement_it(
    check: ModuleType,
) -> None:
    """The comparison primitives must BE the image gate's, not a copy of them.

    This is the load-bearing property of OMN-18865 and the reason it is a
    test rather than a naming convention. The 2026-09-19 window found three
    separate copies of one wrong assumption about the staged tree; a fourth
    copy, living in a package repository where nobody would think to look
    when the image gate changed, is the specific outcome this check must not
    produce. Comparing module identity rather than source text means a lane
    that pastes the logic in fails here instead of at the next incident.
    """
    gate_spec = importlib.util.spec_from_file_location("gate_identity_probe", GATE_PATH)
    assert gate_spec is not None and gate_spec.loader is not None
    gate_module = importlib.util.module_from_spec(gate_spec)
    sys.modules["gate_identity_probe"] = gate_module
    gate_spec.loader.exec_module(gate_module)

    for name in (
        "_tracked_files",
        "_digest_files",
        "_diff_file_maps",
        "_force_included_files",
        "_is_excluded_part",
    ):
        imported = getattr(check, name)
        original = getattr(gate_module, name)
        # Two loads of one file produce distinct code objects, so identity is
        # the wrong test. The property that matters is the DEFINITION SITE: a
        # reimplementation would be defined in the check's own file, and a
        # divergent copy of the gate would sit at a different line.
        assert Path(imported.__code__.co_filename) == GATE_PATH, (
            f"{name} used by the pre-merge check is defined in "
            f"{imported.__code__.co_filename}, not in the image gate at "
            f"{GATE_PATH}. The comparison must have exactly one "
            f"implementation (OMN-18865): a copy drifts from the gate it is "
            f"supposed to predict, which is how a pull request goes green on "
            f"a tree the image build then refuses."
        )
        assert imported.__code__.co_firstlineno == original.__code__.co_firstlineno, (
            f"{name} resolved to a different definition of the same name in "
            f"{GATE_PATH}."
        )


# ---------------------------------------------------------------------------
# Incident replays
# ---------------------------------------------------------------------------


def _write_project(
    root: Path,
    *,
    import_name: str,
    gitignore: str = "",
    force_include: str = "",
    extra_files: dict[str, str] | None = None,
) -> None:
    """Lay down a minimal hatchling project with a source package."""
    pkg = root / "src" / import_name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
        "\n"
        "[project]\n"
        f'name = "{import_name.replace("_", "-")}"\n'
        'version = "0.1.0"\n'
        "\n"
        "[tool.hatch.build.targets.wheel]\n"
        f'packages = ["src/{import_name}"]\n' + force_include
    )
    if gitignore:
        (root / ".gitignore").write_text(gitignore)
    for rel, content in (extra_files or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _make_wheel(path: Path, import_name: str, members: dict[str, str]) -> Path:
    """Build a wheel-shaped zip by hand, with no build backend involved."""
    wheel = path / f"{import_name.replace('_', '-')}-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for rel, content in members.items():
            zf.writestr(f"{import_name}/{rel}", content)
        zf.writestr(f"{import_name}-0.1.0.dist-info/METADATA", "Name: x\n")
        zf.writestr(f"{import_name}-0.1.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
    return wheel


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECK_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_replay_omnimarket_2670_a_dropped_tracked_source_file_is_drift(
    tmp_path: Path,
) -> None:
    """The 21:19Z incident: the wheel silently lost a tracked source file.

    A bare ``merge-sweep/`` in the propagated hygiene block matched a real
    package directory, so hatchling excluded it from the wheel while the
    staged tree kept it. The check must name the dropped file, because
    "something differs" is not actionable and the operator-visible symptom on
    the lane was a single opaque path.
    """
    _write_project(
        tmp_path,
        import_name="omnimarket",
        gitignore="merge-sweep/\n",
        extra_files={
            "src/omnimarket/adapters/codex/skills/merge-sweep/SKILL.md": "skill\n"
        },
    )
    # The wheel hatchling actually produced: everything EXCEPT the excluded file.
    (tmp_path / "dist").mkdir()
    wheel = _make_wheel(tmp_path / "dist", "omnimarket", {"__init__.py": ""})

    result = _run(
        ["--repo-root", str(tmp_path), "--package", "omnimarket", "--wheel", str(wheel)]
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "adapters/codex/skills/merge-sweep/SKILL.md" in result.stderr
    assert "DROPPED" in result.stderr


@pytest.mark.unit
def test_replay_omnibase_core_1710_a_declared_force_include_is_not_drift(
    tmp_path: Path,
) -> None:
    """The 15:37Z incident: a legitimate force-include must NOT read as drift.

    This is the arm omnibase_infra#3846 added to the image gate. The check
    inherits it by importing the gate's resolver, so a package repository
    that ships a governance spec inside its wheel stays green.
    """
    _write_project(
        tmp_path,
        import_name="omnibase_core",
        force_include=(
            "\n[tool.hatch.build.targets.wheel.force-include]\n"
            '"architecture-handshakes/gitignore-baseline.yaml" = '
            '"omnibase_core/data/gitignore-baseline.yaml"\n'
        ),
        extra_files={"architecture-handshakes/gitignore-baseline.yaml": "spec: yes\n"},
    )
    (tmp_path / "dist").mkdir()
    wheel = _make_wheel(
        tmp_path / "dist",
        "omnibase_core",
        {"__init__.py": "", "data/gitignore-baseline.yaml": "spec: yes\n"},
    )

    result = _run(
        [
            "--repo-root",
            str(tmp_path),
            "--package",
            "omnibase_core",
            "--wheel",
            str(wheel),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
def test_a_force_included_file_whose_content_drifted_still_fails_closed(
    tmp_path: Path,
) -> None:
    """Resolving the mapping is a narrowing, never a disarming.

    The force-include arm must keep comparing the CONTENT of the mapped file.
    A wheel carrying a stale copy of a force-included spec is exactly the
    2026-07-14 stale-config shape the parity gate was built for.
    """
    _write_project(
        tmp_path,
        import_name="omnibase_core",
        force_include=(
            "\n[tool.hatch.build.targets.wheel.force-include]\n"
            '"architecture-handshakes/gitignore-baseline.yaml" = '
            '"omnibase_core/data/gitignore-baseline.yaml"\n'
        ),
        extra_files={"architecture-handshakes/gitignore-baseline.yaml": "spec: new\n"},
    )
    (tmp_path / "dist").mkdir()
    wheel = _make_wheel(
        tmp_path / "dist",
        "omnibase_core",
        {"__init__.py": "", "data/gitignore-baseline.yaml": "spec: STALE\n"},
    )

    result = _run(
        [
            "--repo-root",
            str(tmp_path),
            "--package",
            "omnibase_core",
            "--wheel",
            str(wheel),
        ]
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "data/gitignore-baseline.yaml" in result.stderr
    assert "content differs" in result.stderr


@pytest.mark.unit
def test_an_undeclared_extra_wheel_file_is_still_drift(tmp_path: Path) -> None:
    """A wheel file with no source behind it remains a refusal."""
    _write_project(tmp_path, import_name="omnibase_core")
    (tmp_path / "dist").mkdir()
    wheel = _make_wheel(
        tmp_path / "dist",
        "omnibase_core",
        {"__init__.py": "", "data/mystery.yaml": "x\n"},
    )

    result = _run(
        [
            "--repo-root",
            str(tmp_path),
            "--package",
            "omnibase_core",
            "--wheel",
            str(wheel),
        ]
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "data/mystery.yaml" in result.stderr
    assert "no source behind them" in result.stderr


@pytest.mark.unit
def test_a_matching_wheel_passes_and_writes_a_json_result(tmp_path: Path) -> None:
    """The positive control for every zero above: a clean tree must go green."""
    _write_project(tmp_path, import_name="omnibase_core")
    (tmp_path / "dist").mkdir()
    wheel = _make_wheel(tmp_path / "dist", "omnibase_core", {"__init__.py": ""})
    out = tmp_path / "result.json"

    result = _run(
        [
            "--repo-root",
            str(tmp_path),
            "--package",
            "omnibase_core",
            "--wheel",
            str(wheel),
            "--json",
            str(out),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(out.read_text())
    assert payload["status"] == "verified"
    assert payload["differing_files"] == []
    assert payload["source_digest"] == payload["wheel_digest"]


# ---------------------------------------------------------------------------
# The build-root guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_guard_refuses_a_build_root_an_ignore_pattern_matches(
    check: ModuleType, tmp_path: Path
) -> None:
    """hatchling disarms the whole ignore file when the build root is ignored.

    MEASURED with hatchling 1.32.3, one minimal project, one variable:

        .gitignore = ["merge-sweep/"]          built under /private/tmp -> excluded
        .gitignore = ["tmp/", "merge-sweep/"]  built under /private/tmp -> INCLUDED
        .gitignore = ["tmp/", "merge-sweep/"]  built under ~/.cache     -> excluded

    So a check run from the wrong directory returns a confident GREEN on a
    tree whose wheel the image build refuses -- a false PASS, the dangerous
    direction. The guard refuses the location instead of reporting a verdict
    it cannot support.
    """
    root = tmp_path / "scratch" / "proj"
    root.mkdir(parents=True)
    (root / ".gitignore").write_text("scratch/\n")

    errors = check.assert_build_root_is_not_vcs_ignored(root)
    assert errors, "a build root under an ignored directory must be refused"
    assert "scratch" in errors[0]
    assert "REFUSING" in errors[0]


@pytest.mark.unit
def test_the_guard_catches_a_pattern_with_no_trailing_slash(
    check: ModuleType, tmp_path: Path
) -> None:
    """A bare name with no trailing slash matches a directory in git too.

    This is a real bug this guard had until a positive control caught it.
    omnibase_compat ignores a bare ``.cache`` with NO trailing slash; judging
    that repository from a path under ``~/.cache/`` produced a confident
    GREEN on the tree whose wheel drops ``src/omnibase_compat/env/``. A guard
    that only looked at patterns ending in ``/`` saw nothing.
    """
    root = tmp_path / ".cache" / "proj"
    root.mkdir(parents=True)
    (root / ".gitignore").write_text(".cache\n")

    errors = check.assert_build_root_is_not_vcs_ignored(root)
    assert errors, "a trailing slash is optional in git and must be optional here"
    assert ".cache" in errors[0]


@pytest.mark.unit
def test_the_guard_ignores_an_anchored_pattern(
    check: ModuleType, tmp_path: Path
) -> None:
    """An anchored pattern is relative to the project root and cannot match an ancestor."""
    root = tmp_path / "build" / "proj"
    root.mkdir(parents=True)
    (root / ".gitignore").write_text("/build/\n")

    assert check.assert_build_root_is_not_vcs_ignored(root) == []


@pytest.mark.unit
def test_the_guard_fails_closed_on_an_unreadable_gitignore(
    check: ModuleType, tmp_path: Path
) -> None:
    """An unverified build root is an error, never an assumed-clean root."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".gitignore").write_bytes(b"\xff\xfe\x00invalid")

    errors = check.assert_build_root_is_not_vcs_ignored(root)
    assert errors
    assert "Failing closed" in errors[0]


@pytest.mark.unit
def test_a_missing_source_package_directory_is_unresolvable_not_a_pass(
    tmp_path: Path,
) -> None:
    """Exit 2, never exit 0: an unresolvable comparison has not passed."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'x'\nversion = '0.1.0'\n"
    )
    result = _run(["--repo-root", str(tmp_path), "--package", "omnibase_core"])
    assert result.returncode == 2, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# End-to-end: the check builds the wheel itself
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_a_bare_ignore_pattern_drops_a_real_package_directory(
    tmp_path: Path,
) -> None:
    """The omnimarket#2670 shape, with a REAL hatchling build in the loop.

    The hand-built wheels above pin the comparison; this pins the premise
    behind it, that hatchling genuinely applies the VCS ignore file as a
    build-time exclude. If a future hatchling stops doing so, this test goes
    green-by-accident-free: it fails, and the failure says the premise moved.
    """
    _write_project(
        tmp_path,
        import_name="omnimarket",
        gitignore="merge-sweep/\n",
        extra_files={
            "src/omnimarket/adapters/codex/skills/merge-sweep/SKILL.md": "skill\n"
        },
    )
    result = _run(["--repo-root", str(tmp_path), "--package", "omnimarket"])
    if result.returncode == 2:
        pytest.skip(
            f"wheel build unavailable in this environment: {result.stderr[:400]}"
        )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "adapters/codex/skills/merge-sweep/SKILL.md" in result.stderr


@pytest.mark.slow
def test_end_to_end_a_clean_tree_builds_a_matching_wheel(tmp_path: Path) -> None:
    """The positive control for the test above."""
    _write_project(
        tmp_path,
        import_name="omnimarket",
        gitignore="/build/\n",
        extra_files={
            "src/omnimarket/adapters/codex/skills/merge-sweep/SKILL.md": "skill\n"
        },
    )
    result = _run(["--repo-root", str(tmp_path), "--package", "omnimarket"])
    if result.returncode == 2:
        pytest.skip(
            f"wheel build unavailable in this environment: {result.stderr[:400]}"
        )
    assert result.returncode == 0, result.stdout + result.stderr
