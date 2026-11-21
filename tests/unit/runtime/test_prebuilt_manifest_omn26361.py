# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime image carries one prebuilt node catalog and manifest (OMN-26361).

The fixture package set is two fake distributions whose metadata lists three
contracts. Every rule has a planted violation beside a control: the zero-parse
claim is read against a counter that is shown to move on the fallback path.
"""

from __future__ import annotations

import contextlib
import logging
import textwrap
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from omnibase_core.enums.enum_catalog_entry_state import EnumCatalogEntryState
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_infra.handlers import handler_package_announcer as announcer
from omnibase_infra.handlers.handler_prebuilt_manifest import (
    merge_runtime_announcements,
    run_boot_catalog,
)
from omnibase_infra.models.node_announce.model_node_announcement_topic import (
    ModelNodeAnnouncementTopic,
)
from omnibase_infra.models.node_announce.model_package_announce_request import (
    ModelPackageAnnounceRequest,
)
from omnibase_infra.models.node_announce.model_prebuilt_node_manifest import (
    ModelPrebuiltNodeManifest,
)
from omnibase_infra.nodes.node_boot_catalog_effect.runtime_prebuilt_manifest import (
    build_prebuilt_manifest,
)
from omnibase_infra.runtime.auto_wiring import discovery
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_cache_clear,
)

_EP = "omnibase_infra.handlers.handler_package_announcer.entry_points"
_EP_STAMP = "omnibase_infra.handlers.handler_prebuilt_manifest.entry_points"


@contextlib.contextmanager
def installed(entry_points: list[MagicMock]) -> Iterator[None]:
    """The installed ``onex.nodes`` listing, as the announcer and the stamp check both read it."""
    with (
        patch(_EP, return_value=entry_points),
        patch(_EP_STAMP, return_value=entry_points),
    ):
        yield


def _contract(name: str) -> str:
    return textwrap.dedent(f"""\
        name: "{name}"
        node_type: "EFFECT_GENERIC"
        contract_version:
          major: 1
          minor: 0
          patch: 0
        node_version: "1.0.0"
        description: "fixture"
        runtime_profiles: [main]
        event_bus:
          subscribe_topics:
            - "onex.evt.platform.{name.replace("_", "-")}-in.v1"
          publish_topics:
            - "onex.evt.platform.{name.replace("_", "-")}-out.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - operation: run
              handler:
                name: HandlerFixture
                module: omnibase_infra.fixture
    """)


def _entry_point(name: str, dist: str, version: str, path: Path) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.dist = MagicMock()
    ep.dist.name = dist
    ep.dist.version = version
    ep.load.return_value = type("FixtureNode", (), {"contract_path": path})
    return ep


def _write(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "contract.yaml"
    path.write_text(_contract(name), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fresh_memo() -> Iterator[None]:
    discover_contracts_cache_clear()
    yield
    discover_contracts_cache_clear()


@pytest.fixture
def entry_points(tmp_path: Path) -> list[MagicMock]:
    return [
        _entry_point(
            "node_alpha_effect",
            "pkg-one",
            "1.0.0",
            _write(tmp_path, "node_alpha_effect"),
        ),
        _entry_point(
            "node_beta_effect", "pkg-one", "1.0.0", _write(tmp_path, "node_beta_effect")
        ),
        _entry_point(
            "node_gamma_effect",
            "pkg-two",
            "2.1.0",
            _write(tmp_path, "node_gamma_effect"),
        ),
    ]


@pytest.fixture
def prebuilt_file(tmp_path: Path, entry_points: list[MagicMock]) -> Path:
    """The file the image build step writes, from the fixture package set."""
    path = tmp_path / "image" / "prebuilt_node_manifest.json"
    with installed(entry_points):
        result = build_prebuilt_manifest(path, workload="main")
    assert result is not None
    discover_contracts_cache_clear()
    return path


class _YamlLoads:
    """Counts every yaml parse a code path makes, whichever entry point it uses."""

    def __init__(self) -> None:
        self.count = 0

    def wrap(self, real: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return real(*args, **kwargs)

        return counted


@pytest.fixture
def yaml_loads() -> Iterator[_YamlLoads]:
    counter = _YamlLoads()
    with (
        patch.object(yaml, "load", counter.wrap(yaml.load)),
        patch.object(yaml, "safe_load", counter.wrap(yaml.safe_load)),
        patch.object(
            announcer, "read_contract_yaml", counter.wrap(announcer.read_contract_yaml)
        ),
        patch.object(
            discovery, "read_contract_yaml", counter.wrap(discovery.read_contract_yaml)
        ),
    ):
        yield counter


@pytest.mark.unit
def test_prebuilt_manifest_stamp_carries_distributions_versions_and_every_contract_hash(
    prebuilt_file: Path, entry_points: list[MagicMock]
) -> None:
    prebuilt = ModelPrebuiltNodeManifest.model_validate_json(prebuilt_file.read_bytes())

    assert prebuilt.stamp.fingerprint == (
        ("node_alpha_effect", "pkg-one", "1.0.0"),
        ("node_beta_effect", "pkg-one", "1.0.0"),
        ("node_gamma_effect", "pkg-two", "2.1.0"),
    )
    assert len(prebuilt.stamp.contract_hashes) == 3
    for path, content_hash in prebuilt.stamp.contract_hashes:
        assert content_hash == ModelNodeAnnouncement.canonical_content_hash(
            Path(path).read_bytes()
        )
    assert {c.name for c in prebuilt.manifest.contracts} == {
        "node_alpha_effect",
        "node_beta_effect",
        "node_gamma_effect",
    }
    assert set(prebuilt.announced.catalog.entries) == {
        c.name for c in prebuilt.manifest.contracts
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_boot_from_prebuilt_manifest_zero_yaml_loads(
    prebuilt_file: Path, entry_points: list[MagicMock], yaml_loads: _YamlLoads
) -> None:
    for ep in entry_points:
        ep.load.reset_mock()
    with installed(entry_points):
        boot = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=prebuilt_file
        )

    assert boot.manifest is not None
    assert boot.manifest.total_discovered == 3
    assert yaml_loads.count == 0
    for ep in entry_points:
        ep.load.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_boot_from_prebuilt_manifest_counter_moves_on_the_announce_path(
    tmp_path: Path, entry_points: list[MagicMock], yaml_loads: _YamlLoads
) -> None:
    """Positive control: with no file, the same counter sees every contract parsed."""
    with installed(entry_points):
        boot = await run_boot_catalog(
            workload="main",
            announce=announcer.run_boot_step0,
            path=tmp_path / "missing" / "prebuilt.json",
        )
        discover_contracts(boot.announced)

    assert boot.manifest is None
    assert yaml_loads.count >= 3


def _bump_version(entry_points: list[MagicMock], tmp_path: Path) -> list[MagicMock]:
    entry_points[0].dist.version = "1.0.1"
    return entry_points


def _add_distribution(entry_points: list[MagicMock], tmp_path: Path) -> list[MagicMock]:
    return [
        *entry_points,
        _entry_point(
            "node_delta_effect",
            "pkg-three",
            "0.1.0",
            _write(tmp_path, "node_delta_effect"),
        ),
    ]


def _remove_distribution(
    entry_points: list[MagicMock], tmp_path: Path
) -> list[MagicMock]:
    return entry_points[:2]


def _edit_contract(entry_points: list[MagicMock], tmp_path: Path) -> list[MagicMock]:
    path = tmp_path / "node_beta_effect" / "contract.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    return entry_points


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "expected_field"),
    [
        (_bump_version, "distribution_version:pkg-one:node_alpha_effect:1.0.0->1.0.1"),
        (_add_distribution, "distribution_added:pkg-three:node_delta_effect"),
        (_remove_distribution, "distribution_removed:pkg-two:node_gamma_effect"),
        (_edit_contract, "contract_hash:"),
    ],
    ids=["version", "added", "removed", "contract_hash"],
)
async def test_prebuilt_manifest_stale_stamp_falls_back(
    prebuilt_file: Path,
    entry_points: list[MagicMock],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    change: Callable[[list[MagicMock], Path], list[MagicMock]],
    expected_field: str,
) -> None:
    current = change(entry_points, tmp_path)
    with installed(current), caplog.at_level(logging.WARNING):
        boot = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=prebuilt_file
        )

    assert boot.manifest is None
    assert len(boot.announced.catalog.entries) == len(current)
    assert any(expected_field in record.getMessage() for record in caplog.records)
    assert any("differing stamp fields" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prebuilt_manifest_stale_stamp_falls_back_on_a_changed_active_package_set(
    prebuilt_file: Path,
    entry_points: list[MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "pkg-one")
    with installed(entry_points), caplog.at_level(logging.WARNING):
        boot = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=prebuilt_file
        )

    assert boot.manifest is None
    assert any("active_packages" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prebuilt_manifest_stale_stamp_falls_back_on_an_unreadable_file(
    tmp_path: Path, entry_points: list[MagicMock]
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with installed(entry_points):
        boot = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=broken
        )

    assert boot.manifest is None
    assert len(boot.announced.catalog.entries) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prebuilt_manifest_equals_runtime_manifest(
    prebuilt_file: Path, entry_points: list[MagicMock]
) -> None:
    with installed(entry_points):
        loaded = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=prebuilt_file
        )
        discover_contracts_cache_clear()
        step0 = await announcer.run_boot_step0(workload="main")
        runtime_manifest = discover_contracts(step0)

    assert loaded.manifest is not None
    assert loaded.manifest == runtime_manifest
    assert {
        name: entry.content_hash
        for name, entry in loaded.announced.catalog.entries.items()
    } == {name: entry.content_hash for name, entry in step0.catalog.entries.items()}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hotloaded_package_merges_into_prebuilt_catalog(
    prebuilt_file: Path, entry_points: list[MagicMock], tmp_path: Path
) -> None:
    with installed(entry_points):
        loaded = await run_boot_catalog(
            workload="main", announce=announcer.run_boot_step0, path=prebuilt_file
        )
    assert loaded.manifest is not None

    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    hot = _entry_point(
        "node_hot_effect", "pkg-hot", "9.0.0", _write(hot_dir, "node_hot_effect")
    )
    with patch(_EP, return_value=[hot]):
        inventory = announcer.read_package_inventory(None)
    request = ModelPackageAnnounceRequest(
        topic=ModelNodeAnnouncementTopic.suffix_for("main"),
        announcer_id=announcer.announcer_identity("test.hot"),
        generation=1,
        announced_at=datetime.now(UTC),
    )
    merged = merge_runtime_announcements(
        loaded.announced,
        announcer.build_announcements(inventory, request),
        announcer.installed_file_reader(inventory),
    )

    assert set(merged.catalog.entries) == {
        "node_alpha_effect",
        "node_beta_effect",
        "node_gamma_effect",
        "node_hot_effect",
    }
    assert all(
        e.state is EnumCatalogEntryState.ADMITTED
        for e in merged.catalog.entries.values()
    )
    assert set(loaded.announced.catalog.entries) == {
        "node_alpha_effect",
        "node_beta_effect",
        "node_gamma_effect",
    }


_DOCKERFILE = Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.runtime"


@pytest.mark.unit
def test_prebuilt_manifest_stamp_build_step_is_declared_in_the_runtime_dockerfile() -> (
    None
):
    """AC1: the image build writes the file from a declared step, with no new script."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    step = (
        "python -m omnibase_infra.nodes.node_boot_catalog_effect"
        ".runtime_prebuilt_manifest build"
    )
    assert step in text
    inventory_guard = text.index("omnibase_infra.runtime.node_inventory import main")
    healthcheck = text.index("\nHEALTHCHECK ")
    assert inventory_guard < text.index(step) < healthcheck


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prebuilt_manifest_equals_runtime_manifest_for_the_installed_packages(
    tmp_path: Path,
) -> None:
    """The real contracts on disk, no patched metadata: the file serves the manifest discovery builds."""
    path = tmp_path / "prebuilt_node_manifest.json"
    assert build_prebuilt_manifest(path, workload="main") is not None
    discover_contracts_cache_clear()

    loaded = await run_boot_catalog(
        workload="main", announce=announcer.run_boot_step0, path=path
    )
    discover_contracts_cache_clear()
    runtime_manifest = discover_contracts(
        await announcer.run_boot_step0(workload="main")
    )

    assert loaded.manifest is not None
    assert loaded.manifest.total_discovered > 0
    assert loaded.manifest == runtime_manifest


@pytest.mark.unit
def test_prebuilt_manifest_gates_unchanged_healthcheck_start_period_stays_120s() -> (
    None
):
    """AC7: the image's health check and its start period are what they were."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3"
        in text
    )
