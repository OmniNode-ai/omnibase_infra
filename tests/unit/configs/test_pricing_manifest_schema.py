# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Task 8 pricing manifest schema checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.pricing.model_pricing_table import _DEFAULT_MANIFEST_PATH

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LINT_SCRIPT = _REPO_ROOT / "scripts" / "lint_pricing_manifest.py"


def _load_lint_module() -> object:
    spec = importlib.util.spec_from_file_location("lint_pricing_manifest", _LINT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lint_pricing_manifest"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _walk_seed_keys(value: object, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key.startswith("seed_"):
                violations.append(child_path)
            violations.extend(_walk_seed_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_walk_seed_keys(child, f"{path}[{index}]"))
    return violations


@pytest.mark.unit
def test_pricing_manifest_has_no_seed_keys() -> None:
    data = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert _walk_seed_keys(data) == []


@pytest.mark.unit
def test_fallback_pricing_has_source_and_non_authoritative_evidence() -> None:
    data = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    for model_id, entry in data["models"].items():
        assert entry["source"], f"{model_id} missing pricing source"
        assert entry["evidence"], f"{model_id} missing pricing evidence"
        if entry["confidence"] == "LOW_CONFIDENCE":
            assert entry["evidence"]["authoritative"] is False


@pytest.mark.unit
def test_low_sample_manifest_models_are_low_confidence() -> None:
    data = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    for model_id, entry in data["models"].items():
        if entry["sample_count"] < 20:
            assert entry["confidence"] == "LOW_CONFIDENCE", model_id


@pytest.mark.unit
def test_lint_pricing_manifest_blocks_seed_keys(tmp_path: Path) -> None:
    module = _load_lint_module()
    manifest = tmp_path / "pricing_manifest.yaml"
    manifest.write_text(
        "schema_version: '1.0.0'\nmodels:\n  bad:\n    seed_price: 1\n",
        encoding="utf-8",
    )

    errors = module.lint_pricing_manifest(manifest)

    assert errors
    assert "seed_price" in errors[0]


@pytest.mark.unit
def test_lint_pricing_manifest_allows_non_seed_manifest(tmp_path: Path) -> None:
    module = _load_lint_module()
    manifest = tmp_path / "pricing_manifest.yaml"
    manifest.write_text(
        "schema_version: '1.0.0'\nmodels:\n  ok:\n    source: fallback\n",
        encoding="utf-8",
    )

    assert module.lint_pricing_manifest(manifest) == []
