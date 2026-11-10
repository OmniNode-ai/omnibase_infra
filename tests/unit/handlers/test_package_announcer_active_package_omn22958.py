# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The announcer admits an entry point whose import package the lane lists as active (OMN-22958).

The dev lane lists ``omniintelligence`` in ``ONEX_ACTIVE_RUNTIME_PACKAGES``, but that
package ships in the distribution ``omninode-intelligence``. The announcer compared the
distribution name alone, so every omniintelligence entry point was left out of the
catalog and the runtime's ``GET /api/v1/patterns`` answered 404 on the dev runtime
after the entry points were kept in the image (OMN-22391). The control cases prove the
allowlist still excludes a package it does not list.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omnibase_infra.handlers import handler_package_announcer
from omnibase_infra.handlers.handler_package_announcer import read_package_inventory

pytestmark = pytest.mark.unit

_CONTRACT = "name: node_pattern_storage_effect\nnode_type: EFFECT_GENERIC\n"


def _entry_point(tmp_path: Path, *, dist: str, module: str) -> SimpleNamespace:
    contract = tmp_path / "contract.yaml"
    contract.write_text(_CONTRACT, encoding="utf-8")
    node = SimpleNamespace(contract_path=contract)
    return SimpleNamespace(
        name="node_pattern_storage_effect",
        value=f"{module}.nodes.node_pattern_storage_effect",
        dist=SimpleNamespace(name=dist, version="1.2.3"),
        load=lambda: node,
    )


def _read(entry: SimpleNamespace, active: frozenset[str] | None) -> list[str]:
    with patch.object(handler_package_announcer, "entry_points", return_value=[entry]):
        inventory = read_package_inventory(active)
    return [record.contract_name or "" for record in inventory.records]


def test_active_import_package_admits_a_differently_named_distribution(
    tmp_path: Path,
) -> None:
    entry = _entry_point(
        tmp_path, dist="omninode-intelligence", module="omniintelligence"
    )
    admitted = _read(entry, frozenset({"omnimarket", "omniintelligence"}))
    assert admitted == ["node_pattern_storage_effect"]


def test_a_package_the_lane_does_not_list_stays_out(tmp_path: Path) -> None:
    entry = _entry_point(
        tmp_path, dist="omninode-intelligence", module="omniintelligence"
    )
    assert _read(entry, frozenset({"omnibase_infra", "omnimarket"})) == []


def test_distribution_name_still_admits_when_it_is_listed(tmp_path: Path) -> None:
    entry = _entry_point(tmp_path, dist="omnimarket", module="omnimarket")
    assert _read(entry, frozenset({"omnimarket"})) == ["node_pattern_storage_effect"]


def test_no_allowlist_admits_everything(tmp_path: Path) -> None:
    entry = _entry_point(
        tmp_path, dist="omninode-intelligence", module="omniintelligence"
    )
    assert _read(entry, None) == ["node_pattern_storage_effect"]
