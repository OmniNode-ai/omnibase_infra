# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest

from omnibase_infra.testing.rsd_postgres_acceptance_plugin import load_overlay


def test_overlay_is_typed_and_topology_free() -> None:
    path = (
        Path(__file__).parents[3]
        / "docker/lane-overlays/dev.rsd-postgres-acceptance.yaml"
    )
    overlay = load_overlay(path)
    assert overlay.lane == "dev" and overlay.locale == "lab"
    assert "postgresql" not in path.read_text()


def test_overlay_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\nlane: dev\nlocale: lab\nrsd_distribution_ref: omninode-rsd/0.1.0\npostgres_capability_ref: capability://rsd/postgres/acceptance\nhost: bad\n"
    )
    with pytest.raises(Exception):
        load_overlay(path)
