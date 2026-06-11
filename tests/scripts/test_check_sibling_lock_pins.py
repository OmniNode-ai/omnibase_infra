# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/runtime_build/check_sibling_lock_pins.py (OMN-12987).

Recurrence guard for the 2026-06-11 stability bootstrap crash: a workspace-mode
rebuild vendored omnibase_infra 0.37.0 / core 0.42.0 even though omnimarket dev's
uv.lock pinned infra 0.38.1 @ e2dbdc95 / core 0.44.0 @ c97c2c9a. The preflight
must fail-fast when a vendored sibling drifts from the consuming repo's lock.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "runtime_build" / "check_sibling_lock_pins.py"


def _load_module() -> Any:
    mod_name = "check_sibling_lock_pins"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_mod = _load_module()


# The exact pins from omnimarket dev's uv.lock at the time of the 06-11 crash.
_LOCK_INFRA_REV = "e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59"
_LOCK_CORE_REV = "c97c2c9a45c5fb0def5fb7dacfd5f01278bb9f55"
# The stale SHA the crashing build actually vendored.
_STALE_INFRA_REV = "2c1d672f00000000000000000000000000000000"


def _write_lock(path: Path) -> None:
    """Write a minimal uv.lock that pins the four siblings to git revs."""
    path.write_text(
        f"""
version = 1

[[package]]
name = "omnibase-compat"
version = "0.5.1"
source = {{ git = "https://github.com/OmniNode-ai/omnibase_compat.git?rev=4d887307#4d887307aae34d9d40d389ba91070cb411ce3df5" }}

[[package]]
name = "omnibase-core"
version = "0.44.0"
source = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git?rev={_LOCK_CORE_REV}#{_LOCK_CORE_REV}" }}

[[package]]
name = "omnibase-infra"
version = "0.38.1"
source = {{ git = "https://github.com/OmniNode-ai/omnibase_infra.git?rev={_LOCK_INFRA_REV}#{_LOCK_INFRA_REV}" }}

[[package]]
name = "onex-change-control"
version = "0.5.0"
source = {{ git = "https://github.com/OmniNode-ai/onex_change_control.git?rev=4877d3c2#4877d3c223517cb0c7e1eca462ba0f4d38916314" }}
""",
        encoding="utf-8",
    )


def _make_repo(root: Path, repo_dir: str, version: str, sha: str) -> Path:
    repo = root / repo_dir
    repo.mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "{repo_dir}"\nversion = "{version}"\n', encoding="utf-8"
    )
    (repo / ".build-sha").write_text(sha + "\n", encoding="utf-8")
    return repo


# ---------------------------------------------------------------------------
# parse_lock_pins
# ---------------------------------------------------------------------------


def test_parse_lock_pins_extracts_git_revs(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    _write_lock(lock)
    pins = _mod.parse_lock_pins(lock)
    assert pins["omnibase-infra"].version == "0.38.1"
    assert pins["omnibase-infra"].git_rev == _LOCK_INFRA_REV
    assert pins["omnibase-core"].git_rev == _LOCK_CORE_REV
    # Only the four known siblings are extracted.
    assert set(pins) == {
        "omnibase-compat",
        "omnibase-core",
        "omnibase-infra",
        "onex-change-control",
    }


def test_parse_lock_pins_missing_lock_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _mod.parse_lock_pins(tmp_path / "nope.lock")


# ---------------------------------------------------------------------------
# evaluate — matched and mismatched
# ---------------------------------------------------------------------------


def test_evaluate_all_pins_match(tmp_path: Path) -> None:
    lock = tmp_path / "lock.lock"
    _write_lock(lock)
    pins = _mod.parse_lock_pins(lock)
    root = tmp_path / "omni_home"
    _make_repo(
        root, "omnibase_compat", "0.5.1", "4d887307aae34d9d40d389ba91070cb411ce3df5"
    )
    _make_repo(root, "omnibase_core", "0.44.0", _LOCK_CORE_REV)
    _make_repo(root, "omnibase_infra", "0.38.1", _LOCK_INFRA_REV)
    _make_repo(
        root, "onex_change_control", "0.5.0", "4877d3c223517cb0c7e1eca462ba0f4d38916314"
    )

    comparisons = _mod.evaluate(pins, root, "omnibase_infra")
    assert all(c.ok for c in comparisons)
    assert [c.ok for c in comparisons].count(True) == 4


def test_evaluate_stale_infra_sha_fails(tmp_path: Path) -> None:
    """The exact 06-11 failure: infra vendored at a stale SHA must mismatch."""
    lock = tmp_path / "lock.lock"
    _write_lock(lock)
    pins = _mod.parse_lock_pins(lock)
    root = tmp_path / "omni_home"
    _make_repo(
        root, "omnibase_compat", "0.5.1", "4d887307aae34d9d40d389ba91070cb411ce3df5"
    )
    _make_repo(root, "omnibase_core", "0.42.0", _STALE_INFRA_REV)
    # infra at the stale pre-OMN-12501-guard SHA + downgraded version.
    _make_repo(root, "omnibase_infra", "0.37.0", _STALE_INFRA_REV)
    _make_repo(
        root, "onex_change_control", "0.5.0", "4877d3c223517cb0c7e1eca462ba0f4d38916314"
    )

    comparisons = _mod.evaluate(pins, root, "omnibase_infra")
    by_pkg = {c.package: c for c in comparisons}
    assert not by_pkg["omnibase-infra"].ok
    assert not by_pkg["omnibase-core"].ok
    assert by_pkg["omnibase-compat"].ok
    assert by_pkg["onex-change-control"].ok


def test_evaluate_missing_repo_raises(tmp_path: Path) -> None:
    lock = tmp_path / "lock.lock"
    _write_lock(lock)
    pins = _mod.parse_lock_pins(lock)
    root = tmp_path / "omni_home"
    # Only create one repo; the others are absent.
    _make_repo(
        root, "omnibase_compat", "0.5.1", "4d887307aae34d9d40d389ba91070cb411ce3df5"
    )
    with pytest.raises(FileNotFoundError):
        _mod.evaluate(pins, root, "omnibase_infra")


# ---------------------------------------------------------------------------
# read_actual_git_rev — marker vs git fallback
# ---------------------------------------------------------------------------


def test_read_actual_git_rev_prefers_marker(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / ".build-sha").write_text("deadbeefcafebabe\n", encoding="utf-8")
    assert _mod.read_actual_git_rev(repo) == "deadbeefcafebabe"


# ---------------------------------------------------------------------------
# main — end-to-end exit codes
# ---------------------------------------------------------------------------


def _run_main(
    tmp_path: Path, versions: dict[str, tuple[str, str]]
) -> subprocess.CompletedProcess:
    root = tmp_path / "omni_home"
    consuming = root / "omnimarket"
    consuming.mkdir(parents=True)
    _write_lock(consuming / "uv.lock")
    for repo_dir, (ver, sha) in versions.items():
        _make_repo(root, repo_dir, ver, sha)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--omni-home", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_main_exit_zero_on_match(tmp_path: Path) -> None:
    res = _run_main(
        tmp_path,
        {
            "omnibase_compat": ("0.5.1", "4d887307aae34d9d40d389ba91070cb411ce3df5"),
            "omnibase_core": ("0.44.0", _LOCK_CORE_REV),
            "omnibase_infra": ("0.38.1", _LOCK_INFRA_REV),
            "onex_change_control": (
                "0.5.0",
                "4877d3c223517cb0c7e1eca462ba0f4d38916314",
            ),
        },
    )
    assert res.returncode == 0, res.stderr


def test_main_exit_one_on_stale_infra(tmp_path: Path) -> None:
    res = _run_main(
        tmp_path,
        {
            "omnibase_compat": ("0.5.1", "4d887307aae34d9d40d389ba91070cb411ce3df5"),
            "omnibase_core": ("0.42.0", _STALE_INFRA_REV),
            "omnibase_infra": ("0.37.0", _STALE_INFRA_REV),
            "onex_change_control": (
                "0.5.0",
                "4877d3c223517cb0c7e1eca462ba0f4d38916314",
            ),
        },
    )
    assert res.returncode == 1, res.stdout
    assert "do not match" in res.stderr
    assert "omnibase-infra" in res.stderr


def test_provenance_script_folds_lock_pin_comparison(tmp_path: Path) -> None:
    """compute_workspace_provenance.py must merge the staged lock-pin JSON."""
    prov_script = (
        _REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"
    )
    sibling_dir = tmp_path / "workspace" / "sibling-repos"
    for repo in ("omnibase_compat", "onex_change_control", "omnimarket"):
        (sibling_dir / repo).mkdir(parents=True)
        (sibling_dir / repo / "pyproject.toml").write_text(
            f'[project]\nname = "{repo}"\nversion = "0.1.0"\n', encoding="utf-8"
        )
    # Stage a lock-pin comparison file the provenance step should fold in.
    pin_payload = [
        {
            "package": "omnibase-infra",
            "repo": "omnibase_infra",
            "expected_version": "0.38.1",
            "actual_version": "0.38.1",
            "expected_git_rev": _LOCK_INFRA_REV,
            "actual_git_rev": _LOCK_INFRA_REV,
            "version_match": True,
            "sha_match": True,
        }
    ]
    (sibling_dir / ".sibling-lock-pins.json").write_text(
        json.dumps(pin_payload), encoding="utf-8"
    )

    manifest_path = tmp_path / "build-provenance.json"
    script_src = prov_script.read_text(encoding="utf-8")
    script_src = script_src.replace(
        'SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")',
        f'SIBLING_REPOS_DIR = Path("{sibling_dir}")',
    ).replace(
        'OUTPUT_MANIFEST = Path("/app/build-provenance.json")',
        f'OUTPUT_MANIFEST = Path("{manifest_path}")',
    )
    patched = tmp_path / "prov_folded.py"
    patched.write_text(script_src, encoding="utf-8")

    subprocess.run(
        [sys.executable, str(patched)], capture_output=True, text=True, check=False
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "lock_pin_comparison" in manifest
    assert manifest["lock_pin_comparison"] == pin_payload


def test_provenance_script_lock_pin_comparison_empty_when_absent(
    tmp_path: Path,
) -> None:
    """Missing staged lock-pin file yields an empty (well-formed) comparison."""
    prov_script = (
        _REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"
    )
    sibling_dir = tmp_path / "workspace" / "sibling-repos"
    for repo in ("omnibase_compat", "onex_change_control", "omnimarket"):
        (sibling_dir / repo).mkdir(parents=True)
        (sibling_dir / repo / "pyproject.toml").write_text(
            f'[project]\nname = "{repo}"\nversion = "0.1.0"\n', encoding="utf-8"
        )
    manifest_path = tmp_path / "build-provenance.json"
    script_src = prov_script.read_text(encoding="utf-8")
    script_src = script_src.replace(
        'SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")',
        f'SIBLING_REPOS_DIR = Path("{sibling_dir}")',
    ).replace(
        'OUTPUT_MANIFEST = Path("/app/build-provenance.json")',
        f'OUTPUT_MANIFEST = Path("{manifest_path}")',
    )
    patched = tmp_path / "prov_empty.py"
    patched.write_text(script_src, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(patched)], capture_output=True, text=True, check=False
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["lock_pin_comparison"] == []


def test_main_provenance_out_written(tmp_path: Path) -> None:
    root = tmp_path / "omni_home"
    consuming = root / "omnimarket"
    consuming.mkdir(parents=True)
    _write_lock(consuming / "uv.lock")
    _make_repo(
        root, "omnibase_compat", "0.5.1", "4d887307aae34d9d40d389ba91070cb411ce3df5"
    )
    _make_repo(root, "omnibase_core", "0.44.0", _LOCK_CORE_REV)
    _make_repo(root, "omnibase_infra", "0.38.1", _LOCK_INFRA_REV)
    _make_repo(
        root, "onex_change_control", "0.5.0", "4877d3c223517cb0c7e1eca462ba0f4d38916314"
    )
    out = tmp_path / "pins.json"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--omni-home",
            str(root),
            "--provenance-out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report) == 4
    infra = next(r for r in report if r["package"] == "omnibase-infra")
    assert infra["expected_git_rev"] == _LOCK_INFRA_REV
    assert infra["actual_git_rev"] == _LOCK_INFRA_REV
    assert infra["sha_match"] is True
