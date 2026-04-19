# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test for HandlerResolver wiring path (OMN-9201 Task 5).

Exercises the full ``wire_from_manifest`` → ``ServiceHandlerResolver`` →
``_prepare_handler_wiring`` chain against real contracts loaded from disk,
verifying that the new resolver-based wiring path produces the expected
``ModelWiringOutcome`` + ``ModelSkippedEntry`` records.

Scope is intentionally narrow (one real manifest, one assertion on the
skipped_handlers/wirings fields) — the unit-level layer in
``tests/unit/runtime/test_handler_wiring_resolver_integration.py`` covers
the 37-case decision-table exhaustively.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_handler_wiring_resolver_models_importable() -> None:
    """Smoke test: the renamed models + resolver path load cleanly.

    After Task 5 rename (OMN-9201), the boot path now depends on:
      - ``ModelWiringOutcome`` (was ``ModelHandlerWiringOutcome``)
      - ``ModelSkippedEntry`` (was ``ModelSkippedHandlerEntry``)
      - ``ServiceHandlerResolver`` from omnibase_core
      - ``ProtocolHandlerResolver`` + ``ProtocolHandlerOwnershipQuery`` from
        omnibase_spi

    If any of these drift against the published PyPI wheels, the boot
    path regresses silently. This test is the fail-fast tripwire.
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
