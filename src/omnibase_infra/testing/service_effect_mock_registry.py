# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Service effect mock registry for testing.

Provides a registry for mapping service protocol names to mock/stub
implementations during testing. This allows test code to register mock
effect services (database adapters, HTTP clients, message bus publishers,
etc.) and resolve them by protocol name, similar to how
``ModelONEXContainer`` resolves real services at runtime.

Design:
    The registry is intentionally NOT thread-safe. Each instance maintains
    its own internal dict of protocol name to mock callable mappings.
    For thread-safe usage, see :mod:`service_effect_mock_registry_thread_local`.

Example::

    registry = EffectMockRegistry()
    registry.register("ProtocolPostgresAdapter", StubPostgresAdapter())
    registry.register("ProtocolConsulClient", StubConsulClient())

    adapter = registry.resolve("ProtocolPostgresAdapter")
    assert isinstance(adapter, StubPostgresAdapter)

Related:
    - OMN-1336: Add thread-local utility for EffectMockRegistry
    - OMN-1147: Effect Classification System
"""

from __future__ import annotations


class EffectMockRegistry:
    """Registry for mock/stub effect service implementations.

    Maps protocol names (strings) to mock or stub instances for use in
    tests. Provides a lightweight alternative to full container wiring
    when only effect-layer mocks are needed.

    This class is NOT thread-safe. For concurrent test execution, use
    :func:`~omnibase_infra.testing.service_effect_mock_registry_thread_local.get_thread_local_registry`
    to obtain a per-thread instance.

    Attributes:
        _services: Internal mapping of protocol name to mock instance.

    Example::

        registry = EffectMockRegistry()
        registry.register("ProtocolEventBus", mock_bus)

        bus = registry.resolve("ProtocolEventBus")
        assert bus is mock_bus
    """

    def __init__(self) -> None:
        """Initialize an empty mock registry."""
        self._services: dict[str, object] = {}

    def register(  # stub-ok: implemented
        self, protocol_name: str, mock: object
    ) -> None:
        """Register a mock implementation for a protocol name.

        Args:
            protocol_name: The protocol identifier (e.g. ``"ProtocolEventBus"``).
            mock: The mock or stub instance to associate with the protocol.

        Raises:
            ValueError: If ``protocol_name`` is empty.
        """
        if not protocol_name:
            raise ValueError("protocol_name must be a non-empty string")
        self._services[protocol_name] = mock

    def resolve(self, protocol_name: str) -> object:
        """Resolve a mock implementation by protocol name.

        Args:
            protocol_name: The protocol identifier to look up.

        Returns:
            The registered mock instance.

        Raises:
            KeyError: If no mock is registered for the given protocol name.
        """
        if protocol_name not in self._services:
            registered = ", ".join(sorted(self._services.keys())) or "(none)"
            raise KeyError(
                f"No mock registered for '{protocol_name}'. "
                f"Registered protocols: {registered}"
            )
        return self._services[protocol_name]

    def has(self, protocol_name: str) -> bool:
        """Check whether a mock is registered for a protocol name.

        Args:
            protocol_name: The protocol identifier to check.

        Returns:
            True if a mock is registered, False otherwise.
        """
        return protocol_name in self._services

    def unregister(self, protocol_name: str) -> None:
        """Remove a mock registration.

        Args:
            protocol_name: The protocol identifier to remove.

        Raises:
            KeyError: If no mock is registered for the given protocol name.
        """
        if protocol_name not in self._services:
            raise KeyError(f"Cannot unregister '{protocol_name}': not registered")
        del self._services[protocol_name]

    def clear(self) -> None:
        """Remove all registered mocks."""
        self._services.clear()

    @property
    def registered_protocols(self) -> list[str]:
        """Return sorted list of registered protocol names.

        Returns:
            Sorted list of protocol name strings.
        """
        return sorted(self._services.keys())

    def __len__(self) -> int:
        """Return the number of registered mocks."""
        return len(self._services)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        protocols = ", ".join(self.registered_protocols)
        return f"EffectMockRegistry(protocols=[{protocols}])"
