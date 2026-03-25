# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test that RuntimeHostProcess exposes event_bus_wiring for kernel access."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestRuntimeHostProcessWiringAccess:
    """Verify RuntimeHostProcess.event_bus_wiring property exists and is read-only."""

    def test_has_event_bus_wiring_property(self) -> None:
        """RuntimeHostProcess must expose event_bus_wiring as a public property."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        assert hasattr(RuntimeHostProcess, "event_bus_wiring"), (
            "RuntimeHostProcess must expose event_bus_wiring property "
            "so the kernel can pass it to WiringHealthChecker as consumption_source"
        )

    def test_event_bus_wiring_is_property_descriptor(self) -> None:
        """event_bus_wiring must be a @property, not a plain attribute."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        attr = getattr(RuntimeHostProcess, "event_bus_wiring", None)
        assert isinstance(attr, property), (
            "event_bus_wiring must be a @property for read-only access"
        )

    def test_event_bus_wiring_is_read_only(self) -> None:
        """event_bus_wiring must not have a setter."""
        from omnibase_infra.runtime.service_runtime_host_process import (
            RuntimeHostProcess,
        )

        prop = getattr(RuntimeHostProcess, "event_bus_wiring", None)
        assert isinstance(prop, property)
        assert prop.fset is None, "event_bus_wiring must be read-only (no setter)"
