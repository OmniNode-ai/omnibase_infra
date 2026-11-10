# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The announcer admits an entry point by its import package (OMN-22958).

``ONEX_ACTIVE_RUNTIME_PACKAGES`` names import packages: the dev lane lists
``omniintelligence``, while the package ships as the distribution
``omninode-intelligence``. When the announcer admitted entries by distribution
name only, every omniintelligence contract fell out of ``records`` on that lane
and the runtime booted without those nodes.

These tests run the real :func:`read_package_inventory` over this package's real
``onex.nodes`` entry points with an omniintelligence node planted beside them,
which is the set the dev image holds. omninode-intelligence is deliberately
absent from this repository's environment, so the node is planted rather than
installed; any real entry point of that distribution is left out so a run where
it is installed reads the same set. The active set is read the way the runtime
reads it, through :func:`get_active_runtime_packages` against the env var.
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

from omnibase_infra.handlers.handler_package_announcer import read_package_inventory
from omnibase_infra.utils.util_runtime_packages import (
    ENV_ACTIVE_RUNTIME_PACKAGES,
    get_active_runtime_packages,
)

pytestmark = pytest.mark.integration

_EP_SOURCE = "omnibase_infra.handlers.handler_package_announcer.entry_points"
_GROUP = "onex.nodes"
_DIST = "omninode-intelligence"
_VERSION = "1.0.0"
_IMPORT_PACKAGE = "omniintelligence"
_NODE = "node_pattern_storage_effect"
_BROKEN = "node_broken_import_effect"


@dataclass(frozen=True)
class _Dist:
    name: str
    version: str


@dataclass(frozen=True)
class _PlantedEntryPoint:
    name: str
    dist: _Dist
    value: str
    node_cls: type
    load_error: Exception | None = None

    def load(self) -> type:
        if self.load_error is not None:
            raise self.load_error
        return self.node_cls


def _planted(tmp_path: Path, *, folder: str, name: str) -> _PlantedEntryPoint:
    """An entry point of omninode-intelligence whose import package is omniintelligence."""
    node_dir = tmp_path / folder
    node_dir.mkdir()
    contract = node_dir / "contract.yaml"
    contract.write_text(
        dedent(f"""\
            name: "{name}"
            node_type: "EFFECT_GENERIC"
            contract_version:
              major: 1
              minor: 0
              patch: 0
        """),
        encoding="utf-8",
    )
    node_cls_name = "".join(part.capitalize() for part in name.split("_"))
    node_cls = type(node_cls_name, (), {"contract_path": str(contract)})
    module = f"{_IMPORT_PACKAGE}.nodes.{name}"
    return _PlantedEntryPoint(
        name=name,
        dist=_Dist(name=_DIST, version=_VERSION),
        value=f"{module}:{node_cls_name}",
        node_cls=node_cls,
    )


def _broken_import(name: str) -> _PlantedEntryPoint:
    """An entry point of the same import package whose module import fails."""
    node_cls_name = "".join(part.capitalize() for part in name.split("_"))
    module = f"{_IMPORT_PACKAGE}.nodes.{name}"
    return _PlantedEntryPoint(
        name=name,
        dist=_Dist(name=_DIST, version=_VERSION),
        value=f"{module}:{node_cls_name}",
        node_cls=type(node_cls_name, (), {}),
        load_error=ImportError(f"planted: no module named '{module}'"),
    )


def _installed_plus(
    *planted: _PlantedEntryPoint,
) -> Callable[..., list[EntryPoint | _PlantedEntryPoint]]:
    def entry_points(*, group: str) -> list[EntryPoint | _PlantedEntryPoint]:
        found: list[EntryPoint | _PlantedEntryPoint] = [
            ep
            for ep in installed_entry_points(group=group)
            if ep.dist is None or ep.dist.name != _DIST
        ]
        if group == _GROUP:
            found.extend(planted)
        return found

    return entry_points


def _active(
    monkeypatch: pytest.MonkeyPatch, raw_value: str | None
) -> frozenset[str] | None:
    """Set the allowlist env var and read the active set as the runtime does."""
    if raw_value is None:
        monkeypatch.delenv(ENV_ACTIVE_RUNTIME_PACKAGES, raising=False)
    else:
        monkeypatch.setenv(ENV_ACTIVE_RUNTIME_PACKAGES, raw_value)
    return get_active_runtime_packages()


@pytest.fixture(autouse=True)
def _fresh_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_AUTOWIRING_DISCOVERY_CACHE", "0")
    monkeypatch.delenv(ENV_ACTIVE_RUNTIME_PACKAGES, raising=False)


def test_listing_the_import_package_admits_the_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dev lane lists ``omniintelligence`` and the dist is ``omninode-intelligence``."""
    planted = _planted(tmp_path, folder="pattern_storage", name=_NODE)
    active = _active(monkeypatch, "omnibase_infra,omnimarket,omniintelligence")
    with patch(_EP_SOURCE, side_effect=_installed_plus(planted)):
        inventory = read_package_inventory(active)

    records = [r for r in inventory.records if r.entry_point == _NODE]
    assert len(records) == 1
    record = records[0]
    assert record.contract_name == _NODE
    assert record.distribution == _DIST
    assert record.version == _VERSION
    assert record.error_stage is None


def test_unlisted_import_package_keeps_the_entry_point_out_of_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither name listed: no record, but the entry stays in the fingerprint."""
    planted = _planted(tmp_path, folder="pattern_storage", name=_NODE)
    active = _active(monkeypatch, "omnibase_infra,omnimarket")
    with patch(_EP_SOURCE, side_effect=_installed_plus(planted)):
        inventory = read_package_inventory(active)

    assert [r for r in inventory.records if r.distribution == _DIST] == []
    assert (_NODE, _DIST, _VERSION) in inventory.fingerprint


def test_no_allowlist_admits_the_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env allowlist means no filtering at all, so the node is announced."""
    planted = _planted(tmp_path, folder="pattern_storage", name=_NODE)
    active = _active(monkeypatch, None)
    assert active is None
    with patch(_EP_SOURCE, side_effect=_installed_plus(planted)):
        inventory = read_package_inventory(active)

    records = [r for r in inventory.records if r.entry_point == _NODE]
    assert len(records) == 1
    assert records[0].contract_name == _NODE
    assert records[0].distribution == _DIST


def test_load_failure_of_a_listed_import_package_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admitted by import package, then failing to load: a record, not a drop."""
    planted = _planted(tmp_path, folder="pattern_storage", name=_NODE)
    broken = _broken_import(_BROKEN)
    active = _active(monkeypatch, "omnibase_infra,omnimarket,omniintelligence")
    with patch(_EP_SOURCE, side_effect=_installed_plus(planted, broken)):
        inventory = read_package_inventory(active)

    records = [r for r in inventory.records if r.entry_point == _BROKEN]
    assert len(records) == 1
    record = records[0]
    assert record.error_stage == "load"
    assert record.distribution == _DIST
    assert record.contract_name is None
    assert _BROKEN in str(record.error)
    assert (_BROKEN, _DIST, _VERSION) in inventory.fingerprint
