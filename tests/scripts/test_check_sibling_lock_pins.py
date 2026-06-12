# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compatibility tests for sibling lock-pin ratchet scripts (OMN-12977)."""

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
_PROVENANCE_SCRIPT = (
    _REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"
)

_LOCK_INFRA_REV = "e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59"
_LOCK_CORE_REV = "c97c2c9a45c5fb0def5fb7dacfd5f01278bb9f55"


def _load_module() -> Any:
    mod_name = "check_sibling_lock_pins_script_tests"
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_mod = _load_module()


LOCK_FIXTURE = f"""\
[[package]]
name = "omnibase-core"
version = "0.44.0"
source = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git?rev={_LOCK_CORE_REV}#{_LOCK_CORE_REV}" }}

[[package]]
name = "omnibase-infra"
version = "0.38.1"
source = {{ git = "https://github.com/OmniNode-ai/omnibase_infra.git?rev={_LOCK_INFRA_REV}#{_LOCK_INFRA_REV}" }}

[[package]]
name = "omnibase-spi"
version = "0.20.6"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "omnimarket"
version = "0.4.3"
source = {{ editable = "." }}
dependencies = [
    {{ name = "omnibase-infra", git = "https://github.com/OmniNode-ai/omnibase_infra.git?rev={_LOCK_INFRA_REV}" }},
]
"""


def _write_lock(path: Path) -> Path:
    path.write_text(LOCK_FIXTURE, encoding="utf-8")
    return path


def _make_clone(tmp_path: Path, name: str, version: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@example.test",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    return root


def test_parse_lock_pins_extracts_requested_git_and_registry_pins() -> None:
    pins = _mod.parse_lock_pins(
        LOCK_FIXTURE,
        packages=["omnibase-infra", "omnibase-core", "omnibase-spi"],
    )

    assert pins["omnibase-infra"].version == "0.38.1"
    assert pins["omnibase-infra"].git_rev == _LOCK_INFRA_REV
    assert pins["omnibase-core"].git_rev == _LOCK_CORE_REV
    assert pins["omnibase-spi"].git_rev is None


def test_parse_lock_pins_ignores_dependency_revs_for_editable_package() -> None:
    pins = _mod.parse_lock_pins(LOCK_FIXTURE, packages=["omnimarket"])

    assert pins["omnimarket"].version == "0.4.3"
    assert pins["omnimarket"].git_rev is None


def test_parse_lock_pins_missing_requested_package_raises() -> None:
    with pytest.raises(KeyError):
        _mod.parse_lock_pins(LOCK_FIXTURE, packages=["omnibase-infra", "missing"])


def test_check_pins_passes_registry_version_match(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path / "uv.lock")
    clone = _make_clone(tmp_path, "omnibase-spi", "0.20.6")
    out = tmp_path / "pins.json"

    rc = _mod.check_pins(lock, {"omnibase-spi": clone}, output_path=out)

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["drift_count"] == 0
    assert payload["comparisons"][0]["package"] == "omnibase-spi"


def test_check_pins_fails_registry_version_drift(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path / "uv.lock")
    clone = _make_clone(tmp_path, "omnibase-spi", "0.19.0")
    out = tmp_path / "pins.json"

    rc = _mod.check_pins(lock, {"omnibase-spi": clone}, output_path=out)

    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["drift_count"] == 1
    assert payload["comparisons"][0]["drift_direction"] == "backward"


def test_cli_writes_comparison_output(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path / "uv.lock")
    clone = _make_clone(tmp_path, "omnibase-spi", "0.20.6")
    out = tmp_path / "pins.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--repo",
            f"omnibase-spi={clone}",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["drift_count"] == 0


def _write_patched_provenance_script(
    tmp_path: Path,
    sibling_dir: Path,
    manifest_path: Path,
    comparison_path: Path,
) -> Path:
    script_src = _PROVENANCE_SCRIPT.read_text(encoding="utf-8")
    script_src = (
        script_src.replace(
            'SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")',
            f'SIBLING_REPOS_DIR = Path("{sibling_dir}")',
        )
        .replace(
            'OUTPUT_MANIFEST = Path("/app/build-provenance.json")',
            f'OUTPUT_MANIFEST = Path("{manifest_path}")',
        )
        .replace(
            'PIN_COMPARISON_PATH = Path("/workspace/sibling-pin-comparison.json")',
            f'PIN_COMPARISON_PATH = Path("{comparison_path}")',
        )
    )
    patched = tmp_path / "compute_workspace_provenance_patched.py"
    patched.write_text(script_src, encoding="utf-8")
    return patched


def _make_sibling_dirs(root: Path) -> None:
    for repo in ("omnibase_compat", "onex_change_control", "omnimarket"):
        repo_root = root / repo
        repo_root.mkdir(parents=True)
        (repo_root / "pyproject.toml").write_text(
            f'[project]\nname = "{repo}"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )


def test_provenance_script_folds_sibling_pin_comparison(tmp_path: Path) -> None:
    sibling_dir = tmp_path / "workspace" / "sibling-repos"
    _make_sibling_dirs(sibling_dir)
    manifest_path = tmp_path / "build-provenance.json"
    comparison_path = tmp_path / "sibling-pin-comparison.json"
    pin_payload = {
        "lock_source": "uv.lock",
        "allow_drift": False,
        "drift_count": 0,
        "comparisons": [{"package": "omnibase-infra", "matches": True}],
    }
    comparison_path.write_text(json.dumps(pin_payload), encoding="utf-8")
    patched = _write_patched_provenance_script(
        tmp_path, sibling_dir, manifest_path, comparison_path
    )

    subprocess.run(
        [sys.executable, str(patched)], capture_output=True, text=True, check=False
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sibling_pin_comparison"] == pin_payload


def test_provenance_script_uses_none_when_pin_comparison_absent(tmp_path: Path) -> None:
    sibling_dir = tmp_path / "workspace" / "sibling-repos"
    _make_sibling_dirs(sibling_dir)
    manifest_path = tmp_path / "build-provenance.json"
    comparison_path = tmp_path / "missing-sibling-pin-comparison.json"
    patched = _write_patched_provenance_script(
        tmp_path, sibling_dir, manifest_path, comparison_path
    )

    subprocess.run(
        [sys.executable, str(patched)], capture_output=True, text=True, check=False
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sibling_pin_comparison"] is None
