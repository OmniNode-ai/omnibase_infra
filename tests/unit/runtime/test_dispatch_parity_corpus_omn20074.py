# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The parity corpus follows the manifest, retaining complete pinned oracles."""

import json
from pathlib import Path

import pytest

from tests.fixtures.dispatch_parity import harness

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "declaration",
    [
        '[project]\ndependencies = ["onex-change-control>=0.1.0"]\n',
        '[dependency-groups]\ndev = ["onex_change_control>=0.1.0"]\n',
        '[project.optional-dependencies]\ngovernance = ["onex-change-control>=0.1.0"]\n',
    ],
)
def test_declared_occ_remains_required(tmp_path: Path, declaration: str) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(declaration)
    assert harness.required_corpus_packages(manifest) == (
        "omnibase-core",
        "omnibase-infra",
        "onex-change-control",
    )


def test_retired_occ_is_excluded_but_core_and_infra_remain_required(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[dependency-groups]\ndev = ["pytest>=9", {include-group = "test"}]\n'
    )
    assert harness.required_corpus_packages(manifest) == (
        "omnibase-core",
        "omnibase-infra",
    )


@pytest.mark.parametrize("with_occ", [True, False])
def test_each_closure_selects_its_complete_committed_oracle(
    monkeypatch: pytest.MonkeyPatch, with_occ: bool
) -> None:
    required: tuple[str, ...] = ("omnibase-core", "omnibase-infra")
    if with_occ:
        required += ("onex-change-control",)
    monkeypatch.setattr(harness, "REQUIRED_ONEX_NODES_PACKAGES", required)
    oracle = harness.committed_oracle_path()
    assert oracle.name == (
        "selection-oracle-v2.json"
        if with_occ
        else "selection-oracle-without-occ-v2.json"
    )
    snapshot = json.loads(oracle.read_text())
    assert snapshot["header"]["required_onex_nodes_packages"] == list(required)
    assert snapshot["header"]["required_missing_packages"] == []
    assert snapshot["probes"]
    if not with_occ:
        assert (
            "onex-change-control"
            not in snapshot["header"]["corpus"]["contracts_by_package"]
        )
        assert (
            "onex-change-control"
            in snapshot["header"]["expected_excluded_onex_nodes_packages"]
        )


def test_retirement_preserves_every_committed_selection_probe() -> None:
    fixtures = Path(harness.__file__).parent
    with_occ = json.loads((fixtures / "selection-oracle-v2.json").read_text())
    without_occ = json.loads(
        (fixtures / "selection-oracle-without-occ-v2.json").read_text()
    )
    assert len(with_occ["probes"]) == 1090
    assert with_occ["probes"] == without_occ["probes"]
