# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: parity inventories follow declared dependencies and stay fail-closed."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.fixtures.dispatch_parity import harness
from tests.integration.runtime.test_dispatch_selection_parity import (
    test_required_corpus_packages_installed as assert_required_packages_installed,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "occ_requirement", ["onex-change-control==0.5.1", "onex_change_control>=0.5.1"]
)
def test_declared_occ_still_requires_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, occ_requirement: str
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\ndependencies = ["omnibase-core>=0.47.31"]\n'
        f'[dependency-groups]\ndev = ["{occ_requirement}"]\n'
    )
    required = harness.required_onex_nodes_packages(manifest)
    monkeypatch.setattr(harness, "REQUIRED_ONEX_NODES_PACKAGES", required)
    monkeypatch.setattr(
        harness,
        "_installed_onex_nodes_packages",
        lambda: {
            "omnibase-core": "0.47.39",
            "omnibase-infra": "0.38.72",
            "onex-change-control": None,
        },
    )
    with pytest.raises(AssertionError, match="onex-change-control"):
        assert_required_packages_installed()


def test_retired_occ_uses_captured_inventory_without_changing_selection(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text("[project]\ndependencies = []\n[dependency-groups]\ndev = []\n")
    required = harness.required_onex_nodes_packages(manifest)
    legacy = harness.load_selection_oracle(
        ("omnibase-core", "omnibase-infra", "onex-change-control")
    )
    retired = harness.load_selection_oracle(required)
    assert retired["header"]["required_missing_packages"] == []
    assert retired["header"]["required_onex_nodes_packages"] == [
        "omnibase-core",
        "omnibase-infra",
    ]
    assert legacy["header"]["corpus"]["contracts_discovered"] == 153
    assert retired["header"]["corpus"]["contracts_discovered"] == 149
    assert retired["header"]["corpus"]["contracts_by_package"] == {
        "omnibase_core": 21,
        "omnibase_infra": 128,
    }
    assert (
        "onex-change-control"
        in retired["header"]["expected_excluded_onex_nodes_packages"]
    )
    for section in ("dispatchers", "routes", "probes", "discovery_errors"):
        assert retired[section] == legacy[section], section
    assert len(retired["probes"]) == 1090


@pytest.mark.parametrize("missing", ["omnibase-core", "omnibase-infra"])
def test_retired_occ_does_not_make_core_or_infra_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text("[project]\ndependencies = []\n[dependency-groups]\ndev = []\n")
    monkeypatch.setattr(
        harness,
        "REQUIRED_ONEX_NODES_PACKAGES",
        harness.required_onex_nodes_packages(manifest),
    )
    installed: dict[str, str | None] = {
        "omnibase-core": "0.47.39",
        "omnibase-infra": "0.38.72",
    }
    installed[missing] = None
    monkeypatch.setattr(harness, "_installed_onex_nodes_packages", lambda: installed)
    with pytest.raises(AssertionError, match=missing):
        assert_required_packages_installed()


@pytest.mark.parametrize("drift", ["inventory", "route", "selection"])
def test_retired_oracle_still_refuses_runtime_drift(drift: str) -> None:
    committed = harness.load_selection_oracle(("omnibase-core", "omnibase-infra"))
    changed = copy.deepcopy(committed)
    if drift == "inventory":
        changed["header"]["corpus"]["contracts_by_package"]["omnibase_infra"] -= 1
    elif drift == "route":
        changed["routes"].pop()
    else:
        probe = next(iter(changed["probes"].values()))
        probe["selection"]["status"] = "planted-regression"
    assert json.dumps(
        harness.strip_volatile_header_fields(committed), sort_keys=True
    ) != json.dumps(harness.strip_volatile_header_fields(changed), sort_keys=True)


def test_indirect_dependency_group_cannot_silently_retire_occ(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        "[project]\ndependencies = []\n[dependency-groups]\n"
        'dev = [{include-group = "corpus"}]\ncorpus = ["onex-change-control==0.5.1"]\n'
    )
    with pytest.raises(ValueError, match="explicit dependency"):
        harness.required_onex_nodes_packages(manifest)
