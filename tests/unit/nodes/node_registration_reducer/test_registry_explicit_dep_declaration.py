# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies``.

OMN-9198 / HandlerResolver Phase 1 Task 6.

The reducer node declares no event handlers, so its declaration shape is
an empty mapping. The classmethod still exists so the HandlerResolver
auto-wiring path (Task 3) can treat every per-node registry uniformly at
contract-discovery time.

See ``docs/plans/2026-04-18-handler-resolver-architecture.md`` Task 6.
"""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from types import MappingProxyType

import pytest

from omnibase_infra.nodes.node_registration_reducer.registry import (
    RegistryInfraNodeRegistrationReducer,
)


@pytest.mark.unit
def test_declare_explicit_dependencies_is_classmethod() -> None:
    """Callable on the class without an instance -- the resolver never builds one."""
    result = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()

    assert isinstance(result, AbcMapping)


@pytest.mark.unit
def test_declare_explicit_dependencies_is_empty_for_handler_less_node() -> None:
    """The reducer contract has no ``handler_routing`` -- shape must be empty."""
    shape = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()

    assert len(shape) == 0
    assert dict(shape) == {}


@pytest.mark.unit
def test_declare_explicit_dependencies_returns_immutable_mapping() -> None:
    """Returned mapping must be immutable."""
    shape = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()

    assert isinstance(shape, MappingProxyType)

    with pytest.raises(TypeError):
        shape["HandlerBogus"] = ("bogus",)  # type: ignore[index]


@pytest.mark.unit
def test_declare_explicit_dependencies_has_no_side_effects() -> None:
    """Deterministic and cached -- same mapping object on every call."""
    first = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()
    second = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()

    assert dict(first) == dict(second)
    assert first is second


@pytest.mark.unit
def test_declare_explicit_dependencies_does_not_require_container() -> None:
    """Declaration is a classmethod -- it must not touch an instance container.

    Ensures the resolver can discover the declarative shape before any
    runtime state (container, event bus, projector) has been constructed.
    """
    # Calling the classmethod without ever instantiating the registry
    # must succeed. If this method accidentally touched ``self._container``
    # it would fail with AttributeError.
    shape = RegistryInfraNodeRegistrationReducer.declare_explicit_dependencies()

    assert shape is not None
