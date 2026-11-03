# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Discovery with omnimarket's announcer installed beside this package (OMN-22233).

The dev runtime image installs this package and omnimarket together. From
2026-11-03T00:02Z every dev runtime reported ``discovery_errors`` DEGRADED with
``1 contract(s) failed to load: node_package_announce_effect`` because both
packages shipped a contract of that name: omnimarket's announcer node (OMN-21597,
plan task A1b) and this package's boot step 0 contract (OMN-21590). Discovery keeps
the first and rejects the second as ``duplicate_contract_name``.

These tests run the real :func:`discover_contracts` over this package's real
``onex.nodes`` entry points with omnimarket's announcer planted beside them, which
is the set the dev image holds. omnimarket is deliberately absent from this
repository's environment, so the announcer is planted rather than installed; any
real omnimarket entry point is left out so a run where it is installed reads the
same set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as installed_entry_points
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

pytestmark = pytest.mark.integration

_EP_SOURCE = "omnibase_infra.handlers.handler_package_announcer.entry_points"
_GROUP = "onex.nodes"
_ANNOUNCER = "node_package_announce_effect"


@dataclass(frozen=True)
class _Dist:
    name: str
    version: str


@dataclass(frozen=True)
class _PlantedEntryPoint:
    name: str
    dist: _Dist
    node_cls: type

    def load(self) -> type:
        return self.node_cls


def _planted(tmp_path: Path, *, package: str, folder: str) -> _PlantedEntryPoint:
    """An entry point named like omnimarket's announcer, beside its own contract."""
    node_dir = tmp_path / folder
    node_dir.mkdir()
    contract = node_dir / "contract.yaml"
    contract.write_text(
        dedent(f"""\
            name: "{_ANNOUNCER}"
            node_type: "effect"
            contract_version:
              major: 1
              minor: 0
              patch: 0
            node_version: "1.0.0"
            description: "omnimarket's package announcer (OMN-21597), planted"
        """),
        encoding="utf-8",
    )
    node_cls = type("NodePackageAnnounceEffect", (), {"contract_path": str(contract)})
    return _PlantedEntryPoint(
        name=_ANNOUNCER, dist=_Dist(name=package, version="1.0.0"), node_cls=node_cls
    )


def _installed_plus(
    *planted: _PlantedEntryPoint,
) -> Callable[..., list[EntryPoint | _PlantedEntryPoint]]:
    def entry_points(*, group: str) -> list[EntryPoint | _PlantedEntryPoint]:
        found: list[EntryPoint | _PlantedEntryPoint] = [
            ep
            for ep in installed_entry_points(group=group)
            if ep.dist is None or ep.dist.name != "omnimarket"
        ]
        if group == _GROUP:
            found.extend(planted)
        return found

    return entry_points


@pytest.fixture(autouse=True)
def _fresh_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_AUTOWIRING_DISCOVERY_CACHE", "0")
    monkeypatch.delenv("ONEX_ACTIVE_RUNTIME_PACKAGES", raising=False)


def test_announcer_beside_this_package_loads_once_with_no_duplicate(
    tmp_path: Path,
) -> None:
    market = _planted(tmp_path, package="omnimarket", folder="market_announcer")
    with patch(_EP_SOURCE, side_effect=_installed_plus(market)):
        manifest = discover_contracts()

    duplicates = [e for e in manifest.errors if e.reason == "duplicate_contract_name"]
    assert duplicates == []
    owners = {c.name: c.package_name for c in manifest.contracts}
    assert owners.get(_ANNOUNCER) == "omnimarket"
    assert owners.get("node_boot_catalog_effect") == "omnibase_infra"


def test_positive_control_a_second_owner_of_the_announcer_is_rejected(
    tmp_path: Path,
) -> None:
    """The same read reports the incident when a second package ships the name."""
    market = _planted(tmp_path, package="omnimarket", folder="market_announcer")
    second = _planted(tmp_path, package="second_package", folder="second_announcer")
    with patch(_EP_SOURCE, side_effect=_installed_plus(market, second)):
        manifest = discover_contracts()

    duplicates = [e for e in manifest.errors if e.reason == "duplicate_contract_name"]
    assert [(e.package_name, e.entry_point_name) for e in duplicates] == [
        ("second_package", _ANNOUNCER)
    ]
