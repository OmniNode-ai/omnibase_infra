# SPDX-License-Identifier: Apache-2.0
"""Thread-local utility for EffectMockRegistry.

Provides thread-safe access to :class:`EffectMockRegistry` instances
via ``threading.local()``. Each thread receives its own isolated registry,
preventing cross-thread contamination during parallel test execution
(e.g. ``pytest -n auto`` with ``pytest-xdist``).

Usage Patterns:

    **Simple thread-local access**::

        from omnibase_infra.testing import get_thread_local_registry

        registry = get_thread_local_registry()
        registry.register("ProtocolEventBus", mock_bus)

    **Scoped context manager** (auto-cleanup)::

        from omnibase_infra.testing import scoped_effect_mock_registry

        with scoped_effect_mock_registry() as registry:
            registry.register("ProtocolEventBus", mock_bus)
            # ... test code ...
        # registry is cleared on exit

Design:
    The core ``EffectMockRegistry`` is deliberately NOT thread-safe
    to keep it simple and explicit. This module provides the opt-in
    thread-local wrapper for users who need thread safety.

Related:
    - OMN-1336: Add thread-local utility for EffectMockRegistry
    - OMN-1147: Effect Classification System
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager

from omnibase_infra.testing.service_effect_mock_registry import (
    EffectMockRegistry,
)

_thread_local = threading.local()


def get_thread_local_registry() -> EffectMockRegistry:
    """Get or create a thread-local mock registry instance.

    Each thread gets its own ``EffectMockRegistry``. The instance
    persists for the lifetime of the thread (or until
    :func:`clear_thread_local_registry` is called).

    Returns:
        The thread-local ``EffectMockRegistry`` instance.

    Example::

        registry = get_thread_local_registry()
        registry.register("ProtocolPostgresAdapter", stub_adapter)
        resolved = registry.resolve("ProtocolPostgresAdapter")
    """
    if not hasattr(_thread_local, "registry"):
        _thread_local.registry = EffectMockRegistry()
    registry: EffectMockRegistry = _thread_local.registry
    return registry


def clear_thread_local_registry() -> None:
    """Clear and remove the thread-local registry for the current thread.

    After calling this, the next call to :func:`get_thread_local_registry`
    will create a fresh registry instance.

    This is useful in test teardown to prevent state leakage between tests
    when running sequentially in the same thread.
    """
    if hasattr(_thread_local, "registry"):
        _thread_local.registry.clear()
        del _thread_local.registry


@contextmanager
def scoped_effect_mock_registry() -> Generator[EffectMockRegistry, None, None]:
    """Context manager providing a scoped thread-local mock registry.

    Creates (or reuses) the thread-local registry on entry and clears it
    on exit, ensuring no mock registrations leak between test scopes.

    Yields:
        The thread-local ``EffectMockRegistry`` instance.

    Example::

        with scoped_effect_mock_registry() as registry:
            registry.register("ProtocolEventBus", mock_bus)
            # ... test code using the registry ...
        # All registrations are cleared here

    Note:
        If the thread-local registry already has registrations when the
        context manager is entered, they will be preserved during the
        scope but cleared on exit.
    """
    registry = get_thread_local_registry()
    try:
        yield registry
    finally:
        clear_thread_local_registry()
