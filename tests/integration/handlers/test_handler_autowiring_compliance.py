# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OMN-8735 auto-wiring constructor compliance.

Verifies that all handlers updated for OMN-8735 can be instantiated with
no constructor arguments, as required by the strict auto-wiring framework.

The auto-wiring framework calls ``handler_class()`` with no arguments during
node discovery. Any handler that requires positional constructor arguments
will cause a crash-loop in omninode-runtime.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestHandlerAutowiringCompliance:
    """Verify OMN-8735: handlers instantiate with no constructor arguments."""

    def test_handler_contract_file_watcher_no_args(self) -> None:
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        handler = HandlerContractFileWatcher()
        assert handler is not None

    def test_handler_chain_retrieval_no_args(self) -> None:
        from omnibase_infra.nodes.node_chain_retrieval_effect.handlers.handler_chain_retrieval import (
            HandlerChainRetrieval,
        )

        handler = HandlerChainRetrieval()
        assert handler is not None

    def test_handler_chain_store_no_args(self) -> None:
        from omnibase_infra.nodes.node_chain_store_effect.handlers.handler_chain_store import (
            HandlerChainStore,
        )

        handler = HandlerChainStore()
        assert handler is not None

    def test_handler_consumer_health_triage_no_args(self) -> None:
        from omnibase_infra.nodes.node_consumer_health_triage_effect.handlers.handler_consumer_health_triage import (
            HandlerConsumerHealthTriage,
        )

        handler = HandlerConsumerHealthTriage()
        assert handler is not None

    def test_handler_intent_no_args(self) -> None:
        from omnibase_infra.handlers.handler_intent import HandlerIntent

        handler = HandlerIntent()
        assert handler is not None

    def test_handler_ledger_projection_no_args(self) -> None:
        from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
            HandlerLedgerProjection,
        )

        handler = HandlerLedgerProjection()
        assert handler is not None

    def test_handler_llm_openai_compatible_no_args(self) -> None:
        from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
            HandlerLlmOpenaiCompatible,
        )

        handler = HandlerLlmOpenaiCompatible()
        assert handler is not None

    def test_handler_upsert_merge_gate_no_args(self) -> None:
        from omnibase_infra.nodes.node_merge_gate_effect.handlers.handler_upsert_merge_gate import (
            HandlerUpsertMergeGate,
        )

        handler = HandlerUpsertMergeGate()
        assert handler is not None

    def test_handler_onboarding_no_args(self) -> None:
        from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
            HandlerOnboarding,
        )

        handler = HandlerOnboarding()
        assert handler is not None

    def test_handler_node_introspected_no_args(self) -> None:
        from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
            HandlerNodeIntrospected,
        )

        handler = HandlerNodeIntrospected()
        assert handler is not None

    def test_handler_runtime_error_triage_no_args(self) -> None:
        from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers.handler_runtime_error_triage import (
            HandlerRuntimeErrorTriage,
        )

        handler = HandlerRuntimeErrorTriage()
        assert handler is not None

    def test_handler_scope_check_initiate_no_args(self) -> None:
        from omnibase_infra.nodes.node_scope_workflow_orchestrator.handlers.handler_scope_check_initiate import (
            HandlerScopeCheckInitiate,
        )

        handler = HandlerScopeCheckInitiate()
        assert handler is not None

    def test_handler_scope_extract_complete_no_args(self) -> None:
        from omnibase_infra.nodes.node_scope_workflow_orchestrator.handlers.handler_scope_extract_complete import (
            HandlerScopeExtractComplete,
        )

        handler = HandlerScopeExtractComplete()
        assert handler is not None

    def test_handler_scope_file_read_complete_no_args(self) -> None:
        from omnibase_infra.nodes.node_scope_workflow_orchestrator.handlers.handler_scope_file_read_complete import (
            HandlerScopeFileReadComplete,
        )

        handler = HandlerScopeFileReadComplete()
        assert handler is not None

    def test_handler_scope_manifest_write_complete_no_args(self) -> None:
        from omnibase_infra.nodes.node_scope_workflow_orchestrator.handlers.handler_scope_manifest_write_complete import (
            HandlerScopeManifestWriteComplete,
        )

        handler = HandlerScopeManifestWriteComplete()
        assert handler is not None

    def test_handler_llm_cli_subprocess_no_args(self) -> None:
        from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
            HandlerLlmCliSubprocess,
        )

        handler = HandlerLlmCliSubprocess()
        assert handler is not None

    def test_handler_runtime_tick_no_args(self) -> None:
        from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick import (
            HandlerRuntimeTick,
        )

        handler = HandlerRuntimeTick()
        assert handler is not None
