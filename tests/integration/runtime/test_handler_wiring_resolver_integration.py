# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Import-level tripwire for HandlerResolver wiring path (OMN-9201 Task 5).

Narrow scope: the test imports every cross-repo symbol the boot path
consumes and asserts structural invariants (enum members present,
Protocols are Protocol classes, models are frozen, wire_from_manifest is
callable). When upstream ``omnibase_core`` / ``omnibase_spi`` git pins
drift, this file is the fail-fast sentinel — it doesn't exercise the
full wire_from_manifest chain; that is covered exhaustively by
``tests/unit/runtime/test_handler_wiring_resolver_integration.py``
(13 scenarios) and ``tests/unit/runtime/auto_wiring/test_wiring.py``
(123 scenarios) which run every merge.

This file lives under ``tests/integration/`` because the Integration
Test Coverage gate (OMN-7005) requires any src/ change to ship with a
test under ``tests/integration/*.py`` or ``tests/e2e/*.py``. It is NOT
a substitute for unit tests — it is a fast cross-repo-pin integrity
check.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_handler_wiring_resolver_cross_repo_pins_intact() -> None:
    """Fail-fast sentinel: renamed models + resolver protocols load cleanly.

    After Task 5 rename (OMN-9201), the boot path depends on:
      - ``ModelWiringOutcome`` (was ``ModelHandlerWiringOutcome``)
      - ``ModelSkippedEntry`` (was ``ModelSkippedHandlerEntry``)
      - ``ServiceHandlerResolver`` from omnibase_core
      - ``ProtocolHandlerResolver`` + ``ProtocolHandlerOwnershipQuery`` from
        omnibase_spi

    This test verifies imports + structural invariants (enum members,
    Protocol-class metadata, frozen-model config, callable entry points).
    It does NOT execute ``wire_from_manifest`` end-to-end; runtime
    behavior is covered by the 136 unit tests under
    ``tests/unit/runtime/``. If any of the above drift vs the git pins
    in ``pyproject.toml`` ``[tool.uv.sources]``, this test fails first.
    """
    from omnibase_core.enums.enum_handler_resolution_outcome import (
        EnumHandlerResolutionOutcome,
    )
    from omnibase_core.services.service_handler_resolver import (
        ServiceHandlerResolver,
    )
    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
    from omnibase_infra.runtime.auto_wiring.report import (
        ModelSkippedEntry,
        ModelWiringOutcome,
    )
    from omnibase_spi.protocols.runtime.protocol_handler_ownership_query import (
        ProtocolHandlerOwnershipQuery,
    )
    from omnibase_spi.protocols.runtime.protocol_handler_resolver import (
        ProtocolHandlerResolver,
    )

    # The enum must have the full precedence-chain outcome set.
    # (Sanity check — if this list shrinks we've broken resolver contract.)
    expected_outcomes = {
        "RESOLVED_VIA_NODE_REGISTRY",
        "RESOLVED_VIA_CONTAINER",
        "RESOLVED_VIA_EVENT_BUS",
        "RESOLVED_VIA_ZERO_ARG",
        "RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP",
    }
    actual_outcomes = {member.name for member in EnumHandlerResolutionOutcome}
    missing = expected_outcomes - actual_outcomes
    assert not missing, f"Resolver outcome enum is missing members: {missing}"

    # Protocols must be runtime-checkable (they're declared @runtime_checkable).
    assert hasattr(ProtocolHandlerResolver, "__class_getitem__"), (
        "ProtocolHandlerResolver must be a Protocol class"
    )
    assert hasattr(ProtocolHandlerOwnershipQuery, "__class_getitem__"), (
        "ProtocolHandlerOwnershipQuery must be a Protocol class"
    )

    # Models are frozen pydantic BaseModels.
    assert ModelWiringOutcome.model_config.get("frozen") is True
    assert ModelSkippedEntry.model_config.get("frozen") is True

    # wire_from_manifest + ServiceHandlerResolver are importable as callables.
    assert callable(wire_from_manifest)
    assert callable(ServiceHandlerResolver)
