# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for per-node explicit dependency declaration (OMN-9198).

Exercises the ``declare_explicit_dependencies()`` classmethod across multiple
per-node registries together, proving that the declarative surface:

1. loads cleanly when the registry modules are imported through their public
   package entry points (not via direct module-path imports),
2. returns consistent, non-conflicting shapes across the set of per-node
   registries covered by Task 6 of the HandlerResolver plan (the full
   end-to-end resolver-exercise lives in Task 8 / OMN-9204),
3. survives repeated invocation across the live import graph without
   constructing any runtime state (projection_reader / reducer / projector /
   catalog_service / container).

This is an INTEGRATION test (not a unit test) because:

* it imports each registry via its ``omnibase_infra.nodes.<node>.registry``
  public package boundary (exercises ``__init__`` re-exports),
* it asserts cross-registry invariants that a single-registry unit test
  cannot see,
* it asserts the Task 6 Tradeoff Note invariant (declaration is pure) in
  the actual deployed import graph, not against a mock.

See ``docs/plans/2026-04-18-handler-resolver-architecture.md`` Task 6.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import pytest


@pytest.mark.integration
def test_both_per_node_registries_expose_declaration_classmethod() -> None:
    """Both Task-6 registries expose ``declare_explicit_dependencies`` via their
    package boundary (not via private module path)."""
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )
    from omnibase_infra.nodes.node_registration_reducer.registry import (
        RegistryInfraNodeRegistrationReducer,
    )

    # Both classmethods are callable via the public package export.
    orch_shape = (
        RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()
    )
    red_shape = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()

    assert isinstance(orch_shape, Mapping)
    assert isinstance(red_shape, Mapping)


@pytest.mark.integration
def test_declaration_shapes_are_disjoint_across_nodes() -> None:
    """No handler name collides across per-node declarations.

    Each handler class name should belong to exactly one node's declaration.
    Collisions would indicate contract duplication that would confuse the
    HandlerResolver at wiring time.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )
    from omnibase_infra.nodes.node_registration_reducer.registry import (
        RegistryInfraNodeRegistrationReducer,
    )

    orch_handlers = set(
        RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies().keys()
    )
    red_handlers = set(
        RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies().keys()
    )

    collisions = orch_handlers & red_handlers
    assert not collisions, (
        f"Handler name collision across nodes: {sorted(collisions)}. "
        f"Each handler class name must belong to exactly one node's declaration."
    )


@pytest.mark.integration
def test_declarations_are_immutable_proxies() -> None:
    """Declared shapes must be ``MappingProxyType`` — not plain dicts.

    This is the Task 6 Tradeoff Note invariant: the declaration side must not
    be mutable (which would allow a caller to silently inject or drop a
    handler between declaration-time and wiring-time). Verified across the
    live import graph so any accidental regression to a plain ``dict`` is
    caught in integration even if a unit test is missed.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )
    from omnibase_infra.nodes.node_registration_reducer.registry import (
        RegistryInfraNodeRegistrationReducer,
    )

    for cls in (
        RegistryInfraNodeRegistrationOrchestrator,
        RegistryInfraNodeRegistrationReducer,
    ):
        shape = cls.declare_explicit_dependencies()
        assert isinstance(shape, MappingProxyType), (
            f"{cls.__name__}.declare_explicit_dependencies() must return "
            f"MappingProxyType to prevent caller mutation, got {type(shape).__name__}."
        )


@pytest.mark.integration
def test_declarations_never_require_container_or_runtime_services() -> None:
    """The declaration classmethods must load and return shapes WITHOUT any
    runtime infrastructure — no container, no event bus, no projector, no
    Kafka, no database.

    This test asserts the Task 6 invariant that discovery is pure: the
    HandlerResolver can invoke ``declare_explicit_dependencies`` at
    contract-discovery time BEFORE the runtime has bootstrapped.

    Strategy: import both registry classes and call the classmethod. If
    either accidentally depended on a live container or side-effectful
    module-level import, this test would fail with an AttributeError,
    ConnectionError, or ProtocolConfigurationError.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )
    from omnibase_infra.nodes.node_registration_reducer.registry import (
        RegistryInfraNodeRegistrationReducer,
    )

    # Call each classmethod multiple times — cached proxy must return same object.
    orch_first = (
        RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()
    )
    orch_second = (
        RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()
    )
    red_first = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()
    red_second = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()

    # Identity check: cached proxy, not a fresh allocation per call.
    assert orch_first is orch_second
    assert red_first is red_second


@pytest.mark.integration
def test_declared_tuples_are_hashable_and_deterministic() -> None:
    """Every declared dep-key tuple is hashable (suitable as a dict key or
    set member) and its ordering is deterministic across calls.

    The HandlerResolver will use these tuples as cache keys during auto-wiring;
    non-deterministic ordering or unhashable types would break caching.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )

    first = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()
    second = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()

    for handler_name, deps in first.items():
        # Hashable — tuples of strings are always hashable, but make it
        # explicit so a future regression to List[str] would fail here.
        _ = hash(deps)

        # Deterministic ordering.
        deps_second = second[handler_name]
        assert deps == deps_second, (
            f"{handler_name} dep tuple changed between calls: "
            f"{deps!r} vs {deps_second!r}"
        )
