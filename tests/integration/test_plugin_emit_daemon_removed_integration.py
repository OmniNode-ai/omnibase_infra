# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests verifying PluginEmitDaemon removal (OMN-10122).

Confirms that the bespoke PluginEmitDaemon plugin module is fully removed and
no entry points referencing the legacy emit-daemon plugin ID remain registered
under the onex.domain_plugins group.

The replacement (node_emit_daemon) is discovered via the onex.nodes entry-point
group from omnimarket (OMN-10121) — that code path is intentionally NOT tested
here as it lives in a different package.
"""

from __future__ import annotations

import importlib
import importlib.util
from importlib.metadata import entry_points

import pytest

pytestmark = pytest.mark.integration


class TestPluginEmitDaemonRemoved:
    """PluginEmitDaemon module and entry points are fully absent after OMN-10122."""

    def test_plugin_emit_daemon_module_not_importable(self) -> None:
        """The deleted plugin module cannot be imported."""
        spec = importlib.util.find_spec("omnibase_infra.plugins.plugin_emit_daemon")
        assert spec is None, (
            "omnibase_infra.plugins.plugin_emit_daemon should not be importable "
            "after OMN-10122 deletion, but importlib found it. "
            "Verify the file was actually removed from the package."
        )

    def test_plugin_emit_daemon_class_not_importable(self) -> None:
        """PluginEmitDaemon class cannot be imported from any omnibase_infra path."""
        with pytest.raises(ImportError):
            importlib.import_module("omnibase_infra.plugins.plugin_emit_daemon")

    def test_no_emit_daemon_entry_point_in_domain_plugins_group(self) -> None:
        """No entry point named 'emit-daemon' exists in onex.domain_plugins."""
        eps = entry_points(group="onex.domain_plugins")
        emit_daemon_eps = [ep for ep in eps if ep.name == "emit-daemon"]
        assert len(emit_daemon_eps) == 0, (
            f"Expected zero 'emit-daemon' entry points in onex.domain_plugins, "
            f"found {len(emit_daemon_eps)}: {[ep.value for ep in emit_daemon_eps]}. "
            "The PluginEmitDaemon entry point must be removed when the module is deleted."
        )

    def test_no_plugin_emit_daemon_references_in_domain_plugins_group(self) -> None:
        """No entry point in onex.domain_plugins references plugin_emit_daemon module."""
        eps = entry_points(group="onex.domain_plugins")
        stale_refs = [ep for ep in eps if "plugin_emit_daemon" in ep.value]
        assert len(stale_refs) == 0, (
            f"Found entry point(s) still referencing plugin_emit_daemon: "
            f"{[(ep.name, ep.value) for ep in stale_refs]}. "
            "Remove stale entry point declarations from pyproject.toml."
        )


__all__: list[str] = ["TestPluginEmitDaemonRemoved"]
