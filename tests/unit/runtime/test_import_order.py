# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Regression tests for runtime module import order.

This module contains tests to prevent circular import regressions in the
runtime module. The import order in runtime/__init__.py is critical because
chain_aware_dispatch imports ModelEventEnvelope which triggers complex
import chains in omnibase_core.

The fix (OMN-951) was to:
1. Add `# isort: off/on` markers to control import order
2. Move chain_aware_dispatch import to the END of the file
3. Warm the sys.modules cache with other imports first

If the import order is broken, these tests will fail with ImportError
due to circular imports.

Related:
    - OMN-951: Enforce Correlation and Causation Chain Validation
    - src/omnibase_infra/runtime/__init__.py
"""

from __future__ import annotations


class TestRuntimeModuleImports:
    """Tests for runtime module import order and circular import prevention."""

    def test_runtime_module_imports_without_circular_import(self) -> None:
        """Verify runtime module can be imported without circular import errors.

        Regression test for OMN-951: chain_aware_dispatch must be imported LAST
        in runtime/__init__.py to avoid circular import in omnibase_core.

        This test will fail with ImportError if circular import occurs due to
        incorrect import order in the runtime module.
        """
        # This will fail with ImportError if circular import occurs
        from omnibase_infra import runtime

        # Verify key exports are available
        assert hasattr(runtime, "ChainAwareDispatcher")
        assert hasattr(runtime, "MessageDispatchEngine")
        assert hasattr(runtime, "DispatchContextEnforcer")

    def test_chain_aware_dispatch_direct_import(self) -> None:
        """Verify chain_aware_dispatch can be imported after runtime is loaded.

        This test ensures the chain_aware_dispatch module can be imported
        directly after the runtime module has been loaded (warming sys.modules).
        """
        # First ensure runtime is loaded (warms the cache)

        # Now chain_aware_dispatch should import cleanly
        from omnibase_infra.runtime.chain_aware_dispatch import (
            ChainAwareDispatcher,
            propagate_chain_context,
            validate_dispatch_chain,
        )

        assert ChainAwareDispatcher is not None
        assert propagate_chain_context is not None
        assert validate_dispatch_chain is not None

    def test_runtime_exports_chain_aware_dispatch_functions(self) -> None:
        """Verify runtime module exports all chain_aware_dispatch functions.

        These functions should be accessible from omnibase_infra.runtime directly.
        """
        from omnibase_infra import runtime

        # All chain-aware dispatch exports should be available
        assert hasattr(runtime, "ChainAwareDispatcher")
        assert hasattr(runtime, "propagate_chain_context")
        assert hasattr(runtime, "validate_dispatch_chain")

    def test_runtime_exports_core_components(self) -> None:
        """Verify runtime module exports core dispatch components.

        These are the components that must be imported BEFORE chain_aware_dispatch.
        """
        from omnibase_infra import runtime

        # Core components that warm the import cache
        assert hasattr(runtime, "DispatchContextEnforcer")
        assert hasattr(runtime, "DispatcherRegistry")
        assert hasattr(runtime, "ProtocolMessageDispatcher")
        assert hasattr(runtime, "MessageDispatchEngine")

    def test_runtime_exports_registry_components(self) -> None:
        """Verify runtime module exports message type registry components."""
        from omnibase_infra import runtime

        # Message type registry (OMN-937)
        assert hasattr(runtime, "MessageTypeRegistry")
        assert hasattr(runtime, "MessageTypeRegistryError")
        assert hasattr(runtime, "ModelMessageTypeEntry")
        assert hasattr(runtime, "ModelDomainConstraint")
        assert hasattr(runtime, "ProtocolMessageTypeRegistry")


class TestImportOrderIndependence:
    """Tests that verify imports work regardless of which module is imported first."""

    def test_import_chain_aware_dispatch_first(self) -> None:
        """Test that chain_aware_dispatch can be imported directly.

        In a fresh Python interpreter, this would test the import order fix.
        In pytest, modules may already be cached, but this still validates
        the module structure is correct.
        """
        from omnibase_infra.runtime.chain_aware_dispatch import (
            ChainAwareDispatcher,
            propagate_chain_context,
            validate_dispatch_chain,
        )

        assert ChainAwareDispatcher is not None
        assert propagate_chain_context is not None
        assert validate_dispatch_chain is not None

    def test_import_dispatch_context_enforcer_first(self) -> None:
        """Test that dispatch_context_enforcer can be imported directly."""
        from omnibase_infra.runtime.dispatch_context_enforcer import (
            DispatchContextEnforcer,
        )

        assert DispatchContextEnforcer is not None

    def test_import_message_dispatch_engine_first(self) -> None:
        """Test that message_dispatch_engine can be imported directly."""
        from omnibase_infra.runtime.message_dispatch_engine import (
            MessageDispatchEngine,
        )

        assert MessageDispatchEngine is not None

    def test_import_dispatcher_registry_first(self) -> None:
        """Test that dispatcher_registry can be imported directly."""
        from omnibase_infra.runtime.dispatcher_registry import (
            DispatcherRegistry,
            ProtocolMessageDispatcher,
        )

        assert DispatcherRegistry is not None
        assert ProtocolMessageDispatcher is not None
