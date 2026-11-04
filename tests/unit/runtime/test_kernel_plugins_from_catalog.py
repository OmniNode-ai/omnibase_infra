# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kernel plugins are announced and enabled by overlay, not looked up by name (OMN-21591, A3).

Every rule runs for both built-in plugins, ``delegation`` and ``intelligence``, with a
planted violation beside a control. The fixture package set is two fake distributions
whose metadata lists one kernel plugin each; the plugin classes are defined here.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_announced_node_kind import EnumAnnouncedNodeKind
from omnibase_core.enums.enum_catalog_entry_state import EnumCatalogEntryState
from omnibase_infra.handlers.handler_kernel_plugin_selection import (
    HandlerKernelPluginSelection,
    KernelPluginNotAnnouncedError,
    load_kernel_plugin,
)
from omnibase_infra.handlers.handler_package_announcer import (
    build_catalog_direct,
    local_announce_request,
    read_package_inventory,
)
from omnibase_infra.models.node_announce.model_kernel_plugin_selection_request import (
    ModelKernelPluginSelectionRequest,
)
from omnibase_infra.models.node_announce.model_kernel_plugin_selection_result import (
    ModelKernelPluginSelectionResult,
)
from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)
from omnibase_infra.models.node_announce.model_selected_kernel_plugin import (
    ModelSelectedKernelPlugin,
)
from omnibase_infra.models.runtime.model_runtime_kernel_plugins import (
    ENV_RUNTIME_KERNEL_PLUGINS,
    ModelRuntimeKernelPlugins,
)
from omnibase_infra.runtime.protocol_domain_plugin import RegistryDomainPlugin
from omnibase_infra.runtime.service_kernel import register_explicit_domain_plugins
from tests.unit.runtime.test_domain_plugin_discovery_seam import PluginSmoke

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_KERNEL = _REPO / "src" / "omnibase_infra" / "runtime" / "service_kernel.py"
_EP = "omnibase_infra.handlers.handler_package_announcer.entry_points"
_PLUGIN_NAMES = ("delegation", "intelligence")
_DIST = {"delegation": "omnimarket", "intelligence": "omniintelligence"}


class FixturePluginDelegation(PluginSmoke):
    """Stands in for the delegation plugin class."""

    @property
    def plugin_id(self) -> str:
        return "fixture-delegation"


class FixturePluginIntelligence(PluginSmoke):
    """Stands in for the intelligence plugin class."""

    @property
    def plugin_id(self) -> str:
        return "fixture-intelligence"


class FixturePluginThatFailsToStart(PluginSmoke):
    """A plugin whose constructor raises, as a broken installed plugin would."""

    def __init__(self) -> None:
        raise RuntimeError("planted plugin start failure")


_CLASS = {
    "delegation": FixturePluginDelegation,
    "intelligence": FixturePluginIntelligence,
}


def _entry_point(name: str, *, group_value: str | None = None) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.value = group_value or f"{__name__}:{_CLASS[name].__name__}"
    ep.dist.name = _DIST[name]
    ep.dist.version = "1.2.3"
    return ep


@pytest.fixture
def announced() -> ModelPackageAnnounceResult:
    """Both fixture plugins announced as kernel_plugin entries through the real reader."""

    def entry_points(*, group: str) -> list[MagicMock]:
        if group == "onex.domain_plugins":
            return [_entry_point(n) for n in _PLUGIN_NAMES]
        return []

    with patch(_EP, side_effect=entry_points):
        inventory = read_package_inventory(None, include_plugins=True)
    return build_catalog_direct(
        inventory, local_announce_request(None, announce_kernel_plugins=True)
    )


def _select(
    announced: ModelPackageAnnounceResult | None, *enabled: str
) -> tuple[str, ...]:
    result = HandlerKernelPluginSelection().handle(
        ModelKernelPluginSelectionRequest(announced=announced, enabled=enabled)
    )
    return tuple(p.name for p in result.selected)


class TestSelectionFromTheCatalog:
    @pytest.mark.parametrize("name", _PLUGIN_NAMES)
    def test_announced_and_named_is_registered(
        self, announced: ModelPackageAnnounceResult, name: str
    ) -> None:
        entry = announced.catalog.lookup(name)
        assert entry is not None
        assert entry.node_kind is EnumAnnouncedNodeKind.KERNEL_PLUGIN
        assert entry.state is EnumCatalogEntryState.ADMITTED
        result = HandlerKernelPluginSelection().handle(
            ModelKernelPluginSelectionRequest(announced=announced, enabled=(name,))
        )
        assert [p.name for p in result.selected] == [name]
        assert isinstance(load_kernel_plugin(result.selected[0]), _CLASS[name])

    @pytest.mark.parametrize("name", _PLUGIN_NAMES)
    def test_announced_and_not_named_is_not_registered(
        self, announced: ModelPackageAnnounceResult, name: str
    ) -> None:
        other = next(n for n in _PLUGIN_NAMES if n != name)
        assert announced.catalog.lookup(name) is not None
        assert _select(announced, other) == (other,)
        assert _select(announced) == ()

    @pytest.mark.parametrize("name", _PLUGIN_NAMES)
    def test_named_and_not_announced_is_a_typed_boot_error(
        self, announced: ModelPackageAnnounceResult, name: str
    ) -> None:
        other = next(n for n in _PLUGIN_NAMES if n != name)
        only_other = announced.model_copy(
            update={
                "catalog": announced.catalog.model_copy(
                    update={
                        "entries": {
                            k: v
                            for k, v in announced.catalog.entries.items()
                            if k == other
                        }
                    }
                )
            }
        )
        with pytest.raises(KernelPluginNotAnnouncedError) as raised:
            _select(only_other, other, name)
        assert raised.value.plugin_names == (name,)
        assert _select(only_other, other) == (other,)  # control: the announced name

    @pytest.mark.parametrize("name", _PLUGIN_NAMES)
    def test_no_catalog_at_all_cannot_confirm_a_named_plugin(self, name: str) -> None:
        with pytest.raises(KernelPluginNotAnnouncedError):
            _select(None, name)
        assert _select(None) == ()  # control: nothing named, nothing needed

    def test_a_node_entry_of_the_same_name_is_not_a_plugin(
        self, announced: ModelPackageAnnounceResult
    ) -> None:
        entry = announced.catalog.entries["delegation"]
        as_node = entry.model_copy(update={"node_kind": EnumAnnouncedNodeKind.NODE})
        swapped = announced.model_copy(
            update={
                "catalog": announced.catalog.model_copy(
                    update={
                        "entries": {**announced.catalog.entries, "delegation": as_node}
                    }
                )
            }
        )
        with pytest.raises(KernelPluginNotAnnouncedError):
            _select(swapped, "delegation")

    def test_selection_follows_the_overlay_order(
        self, announced: ModelPackageAnnounceResult
    ) -> None:
        assert _select(announced, "intelligence", "delegation") == (
            "intelligence",
            "delegation",
        )


class TestTheOverlayKey:
    def test_names_are_read_in_order_deduplicated(self) -> None:
        env = {ENV_RUNTIME_KERNEL_PLUGINS: " delegation, intelligence ,delegation,,"}
        assert ModelRuntimeKernelPlugins.from_env(env).names == (
            "delegation",
            "intelligence",
        )

    @pytest.mark.parametrize("env", [{}, {ENV_RUNTIME_KERNEL_PLUGINS: "  "}])
    def test_unset_or_blank_names_nothing_and_the_code_holds_no_default(
        self, env: dict[str, str]
    ) -> None:
        assert ModelRuntimeKernelPlugins.from_env(env).names == ()


class TestTheAnnouncerReadsPluginsFromMetadata:
    def test_a_plugin_is_announced_only_when_the_request_asks_for_it(self) -> None:
        def entry_points(*, group: str) -> list[MagicMock]:
            return (
                [_entry_point("delegation")] if group == "onex.domain_plugins" else []
            )

        with patch(_EP, side_effect=entry_points):
            off = build_catalog_direct(
                read_package_inventory(None), local_announce_request(None)
            )
            on = build_catalog_direct(
                read_package_inventory(None, include_plugins=True),
                local_announce_request(None, announce_kernel_plugins=True),
            )
        assert off.catalog.lookup("delegation") is None
        assert on.catalog.lookup("delegation") is not None

    def test_an_inactive_package_announces_no_plugin(self) -> None:
        def entry_points(*, group: str) -> list[MagicMock]:
            return [_entry_point(n) for n in _PLUGIN_NAMES]

        with patch(_EP, side_effect=entry_points):
            inventory = read_package_inventory(
                frozenset({"omnimarket"}), include_plugins=True
            )
        assert [p.name for p in inventory.plugins] == ["delegation"]


class TestTheKernelCarriesNoPluginName:
    def test_the_kernel_source_names_neither_plugin(self) -> None:
        """AC2: the quoted names ``git grep`` looks for are absent from the kernel."""
        pattern = re.compile(r"""(["'])(delegation|intelligence)\1""")
        assert pattern.findall(_KERNEL.read_text()) == []
        # Positive controls: the pattern finds a name in either quote style.
        assert pattern.findall('x = "delegation"')
        assert pattern.findall("x = 'intelligence'")

    def test_the_kernel_selects_before_the_container_exists(self) -> None:
        source = _KERNEL.read_text()
        step0 = source.index("await run_boot_step0(")
        selection = source.index(
            "kernel_plugin_selection = HandlerKernelPluginSelection().select("
        )
        container = source.index("container = ModelONEXContainer()")
        assert step0 < selection < container
        assert "announce_kernel_plugins=True" in source

    def test_the_kernel_no_longer_scans_the_plugin_group(self) -> None:
        source = _KERNEL.read_text()
        assert 'entry_points(group="onex.domain_plugins")' not in source
        assert "delegation_eps" not in source
        assert "intel_eps" not in source


def _registered_ids(selection: ModelKernelPluginSelectionResult) -> list[str]:
    """Run the kernel's own registration step on a real registry; return its order."""
    registry = RegistryDomainPlugin()
    register_explicit_domain_plugins(registry, selection, uuid4())
    return [plugin.plugin_id for plugin in registry.get_all()]


class TestTheKernelRegistersTheSelection:
    """Behavior of the functions ``bootstrap()`` calls, not the kernel's source text."""

    @pytest.mark.parametrize("name", _PLUGIN_NAMES)
    def test_a_named_plugin_is_registered_and_an_unnamed_one_is_not(
        self, announced: ModelPackageAnnounceResult, name: str
    ) -> None:
        env = {ENV_RUNTIME_KERNEL_PLUGINS: name}
        ids = _registered_ids(HandlerKernelPluginSelection().select(announced, env))
        assert ids == ["llm", "dlq", f"fixture-{name}"]
        # Control: nothing named, only the built-ins register.
        assert _registered_ids(
            HandlerKernelPluginSelection().select(announced, {})
        ) == ["llm", "dlq"]

    @pytest.mark.parametrize(
        "order", [("delegation", "intelligence"), ("intelligence", "delegation")]
    )
    def test_built_ins_register_first_then_the_overlay_order(
        self, announced: ModelPackageAnnounceResult, order: tuple[str, ...]
    ) -> None:
        env = {ENV_RUNTIME_KERNEL_PLUGINS: ",".join(order)}
        ids = _registered_ids(HandlerKernelPluginSelection().select(announced, env))
        assert ids == ["llm", "dlq", *(f"fixture-{n}" for n in order)]

    @pytest.mark.parametrize("name", _PLUGIN_NAMES)
    def test_a_plugin_that_fails_to_start_is_skipped_and_the_rest_register(
        self, name: str
    ) -> None:
        broken = ModelSelectedKernelPlugin(
            name="broken",
            distribution="planted",
            target=f"{__name__}:{FixturePluginThatFailsToStart.__name__}",
        )
        good = ModelSelectedKernelPlugin(
            name=name,
            distribution=_DIST[name],
            target=f"{__name__}:{_CLASS[name].__name__}",
        )
        ids = _registered_ids(ModelKernelPluginSelectionResult(selected=(broken, good)))
        assert ids == ["llm", "dlq", f"fixture-{name}"]

    @pytest.mark.parametrize("name", _PLUGIN_NAMES)
    def test_step0_failed_and_a_plugin_is_named_is_a_typed_boot_error(
        self, name: str
    ) -> None:
        """Step 0 failing leaves the kernel no catalog (``None``); a named plugin fails boot."""
        with pytest.raises(KernelPluginNotAnnouncedError) as raised:
            HandlerKernelPluginSelection().select(
                None, {ENV_RUNTIME_KERNEL_PLUGINS: name}
            )
        assert raised.value.plugin_names == (name,)
        # Control: step 0 failed and nothing is named, so boot goes on with no plugin.
        assert HandlerKernelPluginSelection().select(None, {}).selected == ()
