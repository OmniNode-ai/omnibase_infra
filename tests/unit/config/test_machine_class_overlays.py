# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for machine-class overlay profiles (OMN-8904 — Wave B)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAYS_DIR = REPO_ROOT / "config" / "overlays"
INFISICAL_PROJECTS = REPO_ROOT / "config" / "infisical_projects.yaml"
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed-infisical.py"

pytestmark = pytest.mark.unit

EXPECTED_MACHINE_CLASSES = ("mac-dev", "linux-server", "cloud-k8s")


# ---------------------------------------------------------------------------
# Overlay file existence and schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("machine_class", EXPECTED_MACHINE_CLASSES)
def test_overlay_file_exists(machine_class: str) -> None:
    path = OVERLAYS_DIR / f"{machine_class}.yaml"
    assert path.exists(), f"Missing overlay file: {path}"


@pytest.mark.parametrize("machine_class", EXPECTED_MACHINE_CLASSES)
def test_overlay_file_valid_yaml(machine_class: str) -> None:
    path = OVERLAYS_DIR / f"{machine_class}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{machine_class}.yaml must be a YAML mapping"


@pytest.mark.parametrize("machine_class", EXPECTED_MACHINE_CLASSES)
def test_overlay_has_required_fields(machine_class: str) -> None:
    data = yaml.safe_load(
        (OVERLAYS_DIR / f"{machine_class}.yaml").read_text(encoding="utf-8")
    )
    for field in ("version", "machine_class", "infisical_path", "overrides"):
        assert field in data, f"{machine_class}.yaml missing required field: {field}"


@pytest.mark.parametrize("machine_class", EXPECTED_MACHINE_CLASSES)
def test_overlay_machine_class_matches_filename(machine_class: str) -> None:
    data = yaml.safe_load(
        (OVERLAYS_DIR / f"{machine_class}.yaml").read_text(encoding="utf-8")
    )
    assert data["machine_class"] == machine_class


@pytest.mark.parametrize("machine_class", EXPECTED_MACHINE_CLASSES)
def test_overlay_infisical_path_correct(machine_class: str) -> None:
    data = yaml.safe_load(
        (OVERLAYS_DIR / f"{machine_class}.yaml").read_text(encoding="utf-8")
    )
    assert data["infisical_path"] == f"/machine-class/{machine_class}/"


@pytest.mark.parametrize("machine_class", EXPECTED_MACHINE_CLASSES)
def test_overlay_overrides_non_empty(machine_class: str) -> None:
    data = yaml.safe_load(
        (OVERLAYS_DIR / f"{machine_class}.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(data["overrides"], dict) and len(data["overrides"]) > 0


@pytest.mark.parametrize("machine_class", EXPECTED_MACHINE_CLASSES)
def test_overlay_overrides_contain_expected_keys(machine_class: str) -> None:
    """All overlays must declare: Postgres host, Kafka bootstrap, Valkey host."""
    data = yaml.safe_load(
        (OVERLAYS_DIR / f"{machine_class}.yaml").read_text(encoding="utf-8")
    )
    overrides = data["overrides"]
    for key in ("POSTGRES_HOST", "KAFKA_BOOTSTRAP_SERVERS", "VALKEY_HOST"):
        assert key in overrides, (
            f"{machine_class}.yaml missing expected override key: {key}"
        )


def test_overlays_no_secrets() -> None:
    """Overlay files must not contain secret values."""
    secret_suffixes = ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")
    for machine_class in EXPECTED_MACHINE_CLASSES:
        data = yaml.safe_load(
            (OVERLAYS_DIR / f"{machine_class}.yaml").read_text(encoding="utf-8")
        )
        overrides = data.get("overrides", {})
        for key in overrides:
            for suffix in secret_suffixes:
                assert not key.upper().endswith(suffix), (
                    f"{machine_class}.yaml contains likely-secret key '{key}' in overrides. "
                    "Secrets belong in Infisical /shared/<transport>/, not in overlay files."
                )


# ---------------------------------------------------------------------------
# infisical_projects.yaml schema
# ---------------------------------------------------------------------------


def test_infisical_projects_file_exists() -> None:
    assert INFISICAL_PROJECTS.exists(), f"Missing: {INFISICAL_PROJECTS}"


def test_infisical_projects_valid_yaml() -> None:
    data = yaml.safe_load(INFISICAL_PROJECTS.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_infisical_projects_has_machine_classes() -> None:
    data = yaml.safe_load(INFISICAL_PROJECTS.read_text(encoding="utf-8"))
    assert "machine_classes" in data
    names = {mc["name"] for mc in data["machine_classes"]}
    for cls in EXPECTED_MACHINE_CLASSES:
        assert cls in names, f"infisical_projects.yaml missing machine class: {cls}"


def test_infisical_projects_overlay_files_exist() -> None:
    data = yaml.safe_load(INFISICAL_PROJECTS.read_text(encoding="utf-8"))
    for mc in data["machine_classes"]:
        overlay_path = REPO_ROOT / mc["overlay_file"]
        assert overlay_path.exists(), (
            f"infisical_projects.yaml references non-existent overlay: {mc['overlay_file']}"
        )


# ---------------------------------------------------------------------------
# seed-infisical.py --machine-class loading contract
# ---------------------------------------------------------------------------


def _load_seed_module() -> object:
    """Import seed-infisical.py as a module."""
    spec = importlib.util.spec_from_file_location("seed_infisical", str(SEED_SCRIPT))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_load_machine_class_overlay_returns_overrides(tmp_path: Path) -> None:
    """_load_machine_class_overlay parses an overlay YAML and returns the overrides dict."""
    fake_overlay_dir = tmp_path / "overlays"
    fake_overlay_dir.mkdir()
    (fake_overlay_dir / "mac-dev.yaml").write_text(
        "version: '1.0'\n"
        "machine_class: mac-dev\n"
        "infisical_path: /machine-class/mac-dev/\n"
        "overrides:\n"
        "  POSTGRES_HOST: localhost\n"
        "  KAFKA_BOOTSTRAP_SERVERS: localhost:19092\n",
        encoding="utf-8",
    )

    mod = _load_seed_module()
    original = mod._OVERLAYS_DIR  # type: ignore[attr-defined]
    mod._OVERLAYS_DIR = fake_overlay_dir  # type: ignore[attr-defined]
    try:
        result = mod._load_machine_class_overlay("mac-dev")  # type: ignore[attr-defined]
    finally:
        mod._OVERLAYS_DIR = original  # type: ignore[attr-defined]

    assert result["POSTGRES_HOST"] == "localhost"
    assert result["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"


def test_seed_machine_class_dry_run_returns_zero(tmp_path: Path) -> None:
    """_seed_machine_class dry_run=True returns 0 without touching Infisical."""
    fake_overlay_dir = tmp_path / "overlays"
    fake_overlay_dir.mkdir()
    (fake_overlay_dir / "mac-dev.yaml").write_text(
        "version: '1.0'\n"
        "machine_class: mac-dev\n"
        "infisical_path: /machine-class/mac-dev/\n"
        "overrides:\n"
        "  POSTGRES_HOST: localhost\n"
        "  KAFKA_BOOTSTRAP_SERVERS: localhost:19092\n"
        "  VALKEY_HOST: localhost\n",
        encoding="utf-8",
    )

    mod = _load_seed_module()
    original = mod._OVERLAYS_DIR  # type: ignore[attr-defined]
    mod._OVERLAYS_DIR = fake_overlay_dir  # type: ignore[attr-defined]
    try:
        rc = mod._seed_machine_class(  # type: ignore[attr-defined]
            "mac-dev",
            create_missing=True,
            overwrite_existing=False,
            dry_run=True,
        )
    finally:
        mod._OVERLAYS_DIR = original  # type: ignore[attr-defined]

    assert rc == 0


def test_seed_machine_class_unknown_class_returns_one(tmp_path: Path) -> None:
    """_seed_machine_class returns 1 for an unknown machine class."""
    mod = _load_seed_module()
    rc = mod._seed_machine_class(  # type: ignore[attr-defined]
        "nonexistent-class",
        create_missing=True,
        overwrite_existing=False,
        dry_run=True,
    )
    assert rc == 1


def test_seed_script_cli_machine_class_unknown_exits_nonzero() -> None:
    """CLI rejects unknown --machine-class value at argparse level."""
    result = subprocess.run(
        [sys.executable, str(SEED_SCRIPT), "--machine-class", "bad-class"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
