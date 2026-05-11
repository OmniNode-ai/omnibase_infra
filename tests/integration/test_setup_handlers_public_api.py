# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Integration test: setup node handlers are accessible via public __init__.py API.

Verifies that OMN-10705 correctly exposes each setup handler through its node's
handlers/__init__.py public surface, replacing direct sub-module imports in
scripts/onex-setup.py. This is the observable side-effect the ticket targets.

Ticket: OMN-10705
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestSetupHandlersPublicApi:
    """Verify that setup handlers are reachable from the public handlers package."""

    def test_handler_infisical_full_setup_importable_via_init(self) -> None:
        from omnibase_infra.nodes.node_setup_infisical_effect.handlers import (
            HandlerInfisicalFullSetup,
        )

        assert HandlerInfisicalFullSetup is not None

    def test_handler_setup_orchestrator_importable_via_init(self) -> None:
        from omnibase_infra.nodes.node_setup_orchestrator.handlers import (
            HandlerSetupOrchestrator,
        )

        assert HandlerSetupOrchestrator is not None

    def test_handler_service_validate_importable_via_init(self) -> None:
        from omnibase_infra.nodes.node_setup_validate_effect.handlers import (
            HandlerServiceValidate,
        )

        assert HandlerServiceValidate is not None

    def test_handler_infisical_full_setup_in_all(self) -> None:
        import omnibase_infra.nodes.node_setup_infisical_effect.handlers as pkg

        assert "HandlerInfisicalFullSetup" in pkg.__all__

    def test_handler_setup_orchestrator_in_all(self) -> None:
        import omnibase_infra.nodes.node_setup_orchestrator.handlers as pkg

        assert "HandlerSetupOrchestrator" in pkg.__all__

    def test_handler_service_validate_in_all(self) -> None:
        import omnibase_infra.nodes.node_setup_validate_effect.handlers as pkg

        assert "HandlerServiceValidate" in pkg.__all__

    def test_public_import_matches_direct_import_infisical(self) -> None:
        from omnibase_infra.nodes.node_setup_infisical_effect.handlers import (
            HandlerInfisicalFullSetup as PublicHandler,
        )
        from omnibase_infra.nodes.node_setup_infisical_effect.handlers.handler_infisical_full_setup import (
            HandlerInfisicalFullSetup as DirectHandler,
        )

        assert PublicHandler is DirectHandler

    def test_public_import_matches_direct_import_orchestrator(self) -> None:
        from omnibase_infra.nodes.node_setup_orchestrator.handlers import (
            HandlerSetupOrchestrator as PublicHandler,
        )
        from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
            HandlerSetupOrchestrator as DirectHandler,
        )

        assert PublicHandler is DirectHandler

    def test_public_import_matches_direct_import_validate(self) -> None:
        from omnibase_infra.nodes.node_setup_validate_effect.handlers import (
            HandlerServiceValidate as PublicHandler,
        )
        from omnibase_infra.nodes.node_setup_validate_effect.handlers.handler_service_validate import (
            HandlerServiceValidate as DirectHandler,
        )

        assert PublicHandler is DirectHandler
