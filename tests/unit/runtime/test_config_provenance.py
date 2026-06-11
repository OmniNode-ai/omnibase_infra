# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for runtime config provenance (OMN-12958)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.runtime.config_provenance import (
    PROVENANCE_SIDECAR_NAME,
    ModelConfigProvenance,
    build_config_provenance,
    compute_sha256,
    write_provenance_sidecar,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_compute_sha256_absent_returns_none(tmp_path: Path) -> None:
    assert compute_sha256(tmp_path / "missing.yaml") is None


def test_compute_sha256_is_byte_exact(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", "backends: []\n")
    b = _write(tmp_path / "b.yaml", "backends: []\n")
    c = _write(tmp_path / "c.yaml", "backends: [x]\n")
    assert compute_sha256(a) == compute_sha256(b)
    assert compute_sha256(a) != compute_sha256(c)


def test_in_sync_not_drifted(tmp_path: Path) -> None:
    deployed = _write(tmp_path / "deployed.yaml", "backends: []\n")
    source = _write(tmp_path / "source.yaml", "backends: []\n")
    prov = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=deployed,
        source_path=source,
    )
    assert prov.deployed_present
    assert prov.source_present
    assert not prov.has_drifted
    assert "in-sync" in prov.provenance_line()


def test_drift_detected(tmp_path: Path) -> None:
    deployed = _write(tmp_path / "deployed.yaml", "backends: [stale]\n")
    source = _write(tmp_path / "source.yaml", "backends: []\n")
    prov = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=deployed,
        source_path=source,
    )
    assert prov.has_drifted
    assert "DRIFT" in prov.provenance_line()
    assert prov.deployed_sha256 != prov.source_sha256


def test_absent_deployed_is_not_drift(tmp_path: Path) -> None:
    source = _write(tmp_path / "source.yaml", "backends: []\n")
    prov = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=tmp_path / "missing.yaml",
        source_path=source,
    )
    assert not prov.deployed_present
    assert prov.source_present
    # Missing deployed copy is reported via *_present, not has_drifted.
    assert not prov.has_drifted


def test_absent_source_is_not_drift(tmp_path: Path) -> None:
    deployed = _write(tmp_path / "deployed.yaml", "backends: []\n")
    prov = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=deployed,
        source_path=tmp_path / "missing.yaml",
    )
    assert prov.deployed_present
    assert not prov.source_present
    assert not prov.has_drifted


def test_sidecar_roundtrips(tmp_path: Path) -> None:
    deployed = _write(
        tmp_path / "delegation" / "bifrost_delegation.yaml", "backends: []\n"
    )
    source = _write(tmp_path / "source.yaml", "backends: [x]\n")
    prov = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=deployed,
        source_path=source,
    )
    sidecar = write_provenance_sidecar(prov, deployed_path=deployed)
    assert sidecar.name == PROVENANCE_SIDECAR_NAME
    assert sidecar.parent == deployed.parent

    loaded = ModelConfigProvenance.model_validate(
        json.loads(sidecar.read_text(encoding="utf-8"))
    )
    assert loaded == prov
    assert loaded.has_drifted


def test_model_is_frozen(tmp_path: Path) -> None:
    prov = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=tmp_path / "d.yaml",
        source_path=tmp_path / "s.yaml",
    )
    with pytest.raises(Exception):
        prov.config_name = "other"  # type: ignore[misc]
