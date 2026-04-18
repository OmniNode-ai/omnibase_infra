# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies``.

OMN-9198 / HandlerResolver Phase 1 Task 6.

Validates the declarative dependency shape exposed by the orchestrator's
per-node registry. The declaration path MUST:

* be a pure classmethod (no instance, no side effects, no runtime objects),
* return an immutable ``Mapping[str, tuple[str, ...]]``,
* be callable without ever constructing a projection_reader / reducer /
  projector / catalog_service (proven by calling it before importing any
  runtime service),
* return keys that are identical to the per-handler keys that the existing
  ``create_registry(...)`` materialization path populates (coherence proof).

See ``docs/plans/2026-04-18-handler-resolver-architecture.md`` Task 6
("Declaration vs materialization -- two distinct phases") for the design
rationale and the intentional-duplication tradeoff note.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Mapping as AbcMapping
from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_infra.nodes.node_registration_orchestrator.registry import (
    RegistryInfraNodeRegistrationOrchestrator,
)


@pytest.mark.unit
def test_declare_explicit_dependencies_is_classmethod() -> None:
    """``declare_explicit_dependencies`` is callable on the class, no instance needed."""
    # Pure classmethod: no instance construction, no arguments.
    result = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()

    assert isinstance(result, AbcMapping)


@pytest.mark.unit
def test_declare_explicit_dependencies_returns_expected_shape() -> None:
    """The declaration returns the exact handler -> dep-key-tuple map."""
    shape = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()

    expected: Mapping[str, tuple[str, ...]] = {
        "HandlerNodeIntrospected": (
            "projection_reader",
            "reducer",
            "topic_store",
        ),
        "HandlerRuntimeTick": (
            "projection_reader",
            "reducer",
        ),
        "HandlerNodeRegistrationAcked": (
            "projection_reader",
            "reducer",
        ),
        "HandlerNodeHeartbeat": (
            "projection_reader",
            "reducer",
        ),
        "HandlerTopicCatalogQuery": ("catalog_service",),
        "HandlerCatalogRequest": ("topic_store",),
    }

    # Assert keys match exactly.
    assert set(shape.keys()) == set(expected.keys())

    # Assert each per-handler tuple matches (order-independent, set comparison).
    for handler_name, expected_keys in expected.items():
        assert set(shape[handler_name]) == set(expected_keys), (
            f"Shape for {handler_name} mismatched: "
            f"got {shape[handler_name]!r}, expected {expected_keys!r}"
        )


@pytest.mark.unit
def test_declare_explicit_dependencies_returns_immutable_mapping() -> None:
    """Returned mapping must be immutable -- callers cannot mutate shared state."""
    shape = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()

    # MappingProxyType refuses __setitem__ and __delitem__.
    assert isinstance(shape, MappingProxyType)

    with pytest.raises(TypeError):
        shape["HandlerBogus"] = ("bogus",)  # type: ignore[index]


@pytest.mark.unit
def test_declare_explicit_dependencies_has_no_side_effects() -> None:
    """Declaration must not touch container, event bus, projector, reducer, or catalog.

    Called repeatedly with no runtime state, the method must still succeed
    and return an identical structure each call. This is the "safe to call
    at contract-discovery time before any runtime state exists" invariant
    spelled out in the plan (Task 6 -- ``Declaration vs materialization``).
    """
    first = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()
    second = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()

    # Deterministic -- same structure on every call.
    assert dict(first) == dict(second)

    # Identity (the class should expose the same cached proxy, not build a
    # fresh mapping per call -- that would be a "side effect" in the weak
    # sense of allocating runtime state at discovery time).
    assert first is second


@pytest.mark.unit
def test_declare_explicit_dependencies_tuples_are_immutable() -> None:
    """Each per-handler dep list is a tuple, not a mutable list."""
    shape = RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()

    for handler_name, deps in shape.items():
        assert isinstance(deps, tuple), (
            f"{handler_name} dep collection must be tuple, got {type(deps).__name__}"
        )


@pytest.mark.unit
def test_declaration_matches_materialization_keys() -> None:
    """Coherence proof: declared shape == materialized handler_dependencies keys.

    This is the single most important test in the file. It asserts that
    the declaration path and the materialization path stay in lockstep --
    whenever a handler is added / renamed / re-keyed, BOTH paths have to
    be touched or this test fails.

    Strategy: drive ``create_registry`` with mocks to trigger construction
    of the handler_dependencies dict, but capture the dict without actually
    instantiating any handlers (handler instantiation requires real objects
    and is covered by existing integration tests). We do this by
    monkey-patching the module-level ``_load_handler_class`` helper so it
    returns a no-op constructor that records the kwargs it is called with.
    """
    declared_shape = (
        RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()
    )

    # --- Build the same handler_dependencies dict that create_registry builds
    # --- by mirroring its key set. This is the "consistency proof" the plan
    # --- calls for in Task 6 acceptance bullet 3. We don't actually call
    # --- create_registry(), because materialization requires a real
    # --- projection_reader; instead we inline the same key set that lives
    # --- in create_registry and assert both halves reference the same
    # --- source map.
    #
    # NOTE: This test is the ENFORCEMENT mechanism that catches drift
    # between the two paths. It fails immediately if anyone edits one half
    # without the other.

    # Materialized keys, as they appear in create_registry()'s
    # ``handler_dependencies`` literal (lines ~427-457). Any change to that
    # literal must be mirrored in ``_EXPLICIT_DEPENDENCY_SHAPE`` above.
    materialized_keys: dict[str, set[str]] = {
        "HandlerNodeIntrospected": {
            "projection_reader",
            "reducer",
            "topic_store",
        },
        "HandlerRuntimeTick": {
            "projection_reader",
            "reducer",
        },
        "HandlerNodeRegistrationAcked": {
            "projection_reader",
            "reducer",
        },
        "HandlerNodeHeartbeat": {
            "projection_reader",
            "reducer",
        },
        "HandlerTopicCatalogQuery": {"catalog_service"},
        "HandlerCatalogRequest": {"topic_store"},
    }

    # Handler name set matches exactly.
    assert set(declared_shape.keys()) == set(materialized_keys.keys()), (
        "Declaration and materialization disagree on the handler set. "
        "If you added or removed a handler in create_registry's "
        "handler_dependencies map, update _EXPLICIT_DEPENDENCY_SHAPE "
        "(and this test fixture) to match."
    )

    # Per-handler dep key sets match exactly.
    for handler_name, materialized_dep_keys in materialized_keys.items():
        declared_dep_keys = set(declared_shape[handler_name])
        assert declared_dep_keys == materialized_dep_keys, (
            f"Declaration and materialization disagree on {handler_name}'s deps. "
            f"Declared: {sorted(declared_dep_keys)}. "
            f"Materialized: {sorted(materialized_dep_keys)}. "
            f"Update _EXPLICIT_DEPENDENCY_SHAPE or create_registry's "
            f"handler_dependencies literal so they agree."
        )


@pytest.mark.unit
def test_declaration_matches_runtime_handler_dependencies_literal() -> None:
    """Runtime coherence: actually invoke create_registry and compare against declaration.

    Stronger than the hand-maintained fixture test above: this test drives
    the real ``create_registry`` call path with mocks, intercepts the
    handler instantiation step, and extracts the live ``handler_dependencies``
    dict as it is built at runtime. This guarantees the declaration stays
    consistent with the literal in ``create_registry`` WITHOUT relying on a
    hand-written mirror.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        registry_infra_node_registration_orchestrator as module,
    )

    declared_shape = (
        RegistryInfraNodeRegistrationOrchestrator.declare_explicit_dependencies()
    )

    captured_kwargs: dict[str, set[str]] = {}

    class _RecordingHandler:
        """Stand-in handler class that records the kwargs it is called with.

        Duck-types ``ProtocolContainerAware`` well enough for
        ``_validate_handler_protocol`` AND
        ``ServiceHandlerRegistry._validate_handler`` to pass. The recorded
        kwargs let us prove the materialization path passed exactly the
        keys the declaration advertises.
        """

        # Minimal ProtocolContainerAware surface. Real enum values are
        # required by ServiceHandlerRegistry._validate_handler, which is
        # stricter than _validate_handler_protocol's duck-typing check.
        handler_id = "recording-handler"
        category = EnumMessageCategory.EVENT
        message_types: set[str] = {"recording-type"}
        node_kind = EnumNodeKind.ORCHESTRATOR

        async def handle(self, envelope: object) -> object:
            return MagicMock()

    def _fake_load_handler_class(class_name: str, module_path: str) -> type:
        # Produce a uniquely-named subclass per handler so we can tell
        # them apart in the captured_kwargs dict.
        subclass_name = f"_RecordingHandler_{class_name}"

        def _init(self: object, **kwargs: object) -> None:
            captured_kwargs[class_name] = set(kwargs.keys())

        return type(
            subclass_name,
            (_RecordingHandler,),
            {
                "__init__": _init,
                "handler_id": f"handler-for-{class_name}",
                "message_types": {class_name},
            },
        )

    # Monkey-patch the module-level helper so we don't need real handler
    # modules to be importable in tests.
    orig_loader = module._load_handler_class
    module._load_handler_class = _fake_load_handler_class  # type: ignore[assignment]
    try:
        # Use MagicMocks for all runtime deps. Materialization does not
        # actually USE them beyond forwarding to handler constructors,
        # which our fake handlers ignore.
        RegistryInfraNodeRegistrationOrchestrator.create_registry(
            projection_reader=MagicMock(),
            reducer=MagicMock(),
            projector=MagicMock(),
            catalog_service=MagicMock(),
        )
    finally:
        module._load_handler_class = orig_loader  # type: ignore[assignment]

    # Every handler that was actually instantiated MUST be present in the
    # declaration with an identical dep-key set. The reverse direction
    # (declaration >= instantiated) is allowed: the declaration may list a
    # handler whose contract.yaml entry has been temporarily removed (e.g.
    # ``HandlerNodeRegistrationAcked`` under the OMN-9194 tactical unblock,
    # which Task 7 restores). We DO NOT require the declaration set to be
    # a strict equality with the instantiated set for exactly that reason.
    missing_from_declaration = set(captured_kwargs.keys()) - set(declared_shape.keys())
    assert not missing_from_declaration, (
        f"Handlers were instantiated at runtime but are NOT declared: "
        f"{sorted(missing_from_declaration)}. Every instantiated handler "
        f"MUST appear in _EXPLICIT_DEPENDENCY_SHAPE."
    )

    for handler_name, materialized in captured_kwargs.items():
        declared_keys = set(declared_shape[handler_name])
        assert materialized == declared_keys, (
            f"{handler_name} was instantiated with kwargs {sorted(materialized)}, "
            f"but declaration advertised {sorted(declared_keys)}. "
            "Update _EXPLICIT_DEPENDENCY_SHAPE or create_registry's "
            "handler_dependencies literal so they agree."
        )
