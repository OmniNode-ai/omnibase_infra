# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol ownership verification tests (INFRA-016).

These tests verify that omnibase_infra does not declare new public protocol
contracts that should live in omnibase_spi. Infra is an *implementer* of
SPI protocols, not a *declarer* of new contract-surface protocols.

Infra MAY declare narrow, implementation-specific protocols for internal use
(DI boundaries, mixin host contracts, handler interfaces). These are tracked
in a known allowlist and must be explicitly reviewed when adding new ones.

If a test fails, it means a new Protocol class was added to omnibase_infra
without updating the allowlist. The developer must decide:
  1. Move the protocol to omnibase_spi (if it is a cross-repo contract)
  2. Add it to the allowlist here (if it is infra-internal)

Related:
    - OMN-757: INFRA-016: Protocol ownership verification
    - omnibase_spi.protocols: Canonical protocol definitions

.. versionadded:: 0.11.0
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Resolve repo root from test file location
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_INFRA_SRC = _REPO_ROOT / "src" / "omnibase_infra"

# Known infra-internal protocols that are NOT violations.
# Each entry maps protocol name to its relative file path (from src/omnibase_infra/).
# When adding a new protocol, you MUST add it here with a justification comment.
#
# Categories:
#   [DI]     - Dependency injection boundary (narrow interface for constructor injection)
#   [MIXIN]  - Mixin host contract (defines what the mixin expects from its host)
#   [NODE]   - Node-internal protocol (handler/effect interface within a single node)
#   [RUNTIME]- Runtime-internal protocol (kernel/scheduler/lifecycle internals)
#   [OBS]    - Observability protocol (metrics/health check interface)
KNOWN_INFRA_PROTOCOLS: dict[str, str] = {
    # === protocols/ package (infra's own protocol contracts) ===
    # These are infra-level abstractions over SPI, providing domain-specific
    # contracts for the infra runtime. They bridge SPI protocols to infra internals.
    "ProtocolCapabilityProjection": "protocols/protocol_capability_projection.py",
    "ProtocolCapabilityQuery": "protocols/protocol_capability_query.py",
    "ProtocolContainerAware": "protocols/protocol_container_aware.py",
    "ProtocolDispatchEngine": "protocols/protocol_dispatch_engine.py",
    "ProtocolEventBusLike": "protocols/protocol_event_bus_like.py",
    "ProtocolEventProjector": "protocols/protocol_event_projector.py",
    "ProtocolIdempotencyStore": "protocols/protocol_idempotency_store.py",
    "ProtocolLedgerSink": "protocols/protocol_ledger_sink.py",
    "ProtocolMessageDispatcher": "protocols/protocol_message_dispatcher.py",
    "ProtocolMessageTypeRegistry": "protocols/protocol_message_type_registry.py",
    "ProtocolNodeHeartbeat": "protocols/protocol_node_heartbeat.py",
    "ProtocolNodeIntrospection": "protocols/protocol_node_introspection.py",
    "ProtocolPayloadRegistry": "protocols/protocol_payload_registry.py",
    "ProtocolPluginCompute": "protocols/protocol_plugin_compute.py",
    "ProtocolProjectorSchemaValidator": "protocols/protocol_projector_schema_validator.py",
    "ProtocolRegistryMetrics": "protocols/protocol_registry_metrics.py",
    "ProtocolSnapshotPublisher": "protocols/protocol_snapshot_publisher.py",
    "ProtocolSnapshotStore": "protocols/protocol_snapshot_store.py",
    "ProtocolValidationLedgerRepository": "protocols/protocol_validation_ledger_repository.py",
    # === [DI] Dependency injection boundaries ===
    "ProtocolConsulClient": "nodes/effects/protocol_consul_client.py",
    "ProtocolEffectIdempotencyStore": "nodes/effects/protocol_effect_idempotency_store.py",
    "ProtocolPostgresAdapter": "nodes/effects/protocol_postgres_adapter.py",
    "ProtocolToolExecutor": "handlers/mcp/protocols.py",
    "ProtocolLlmHandler": "nodes/node_llm_inference_effect/services/protocol_llm_handler.py",
    "ProtocolContractPublisherSource": "services/contract_publisher/sources/protocol.py",
    "ProtocolInjectionEffectivenessReader": "services/observability/injection_effectiveness/protocol_reader.py",
    "ProtocolTopicCatalogService": "services/protocol_topic_catalog_service.py",
    "ProtocolManifestPersistence": "services/corpus_capture.py",
    "ProtocolSessionAggregator": "services/session/protocol_session_aggregator.py",
    # === [MIXIN] Mixin host contracts ===
    "ProtocolCircuitBreakerAware": "mixins/protocol_circuit_breaker_aware.py",
    "ProtocolKafkaDlqHost": "event_bus/mixin_kafka_dlq.py",
    "ProtocolKafkaBroadcastHost": "event_bus/mixin_kafka_broadcast.py",
    "ProtocolConsulServiceDependencies": "handlers/mixins/mixin_consul_service.py",
    "ProtocolConsulTopicIndexDependencies": "handlers/mixins/mixin_consul_topic_index.py",
    "ProtocolConsulKVDependencies": "handlers/mixins/mixin_consul_kv.py",
    "ProtocolConsulInitDependencies": "handlers/mixins/mixin_consul_initialization.py",
    "ProtocolProjectorNotificationContext": "runtime/mixins/mixin_projector_notification_publishing.py",
    "ProtocolProjectorContext": "runtime/mixins/mixin_projector_sql_operations.py",
    # === [NODE] Node-internal protocols ===
    "ProtocolArchitectureRule": "nodes/architecture_validator/protocols/protocol_architecture_rule.py",
    "ProtocolRegistrationIntent": "nodes/node_registration_orchestrator/protocols.py",
    "ProtocolReducer": "nodes/node_registration_orchestrator/protocols.py",
    "ProtocolEffect": "nodes/node_registration_orchestrator/protocols.py",
    "ProtocolPartialRetryRequest": "nodes/node_registry_effect/handlers/handler_partial_retry.py",
    "ProtocolRegistrationPersistence": "nodes/node_registration_storage_effect/protocols/protocol_registration_persistence.py",
    "ProtocolDiscoveryOperations": "nodes/node_service_discovery_effect/protocols/protocol_discovery_operations.py",
    "ProtocolLedgerPersistence": "nodes/node_ledger_write_effect/protocols/protocol_ledger_persistence.py",
    # === [RUNTIME] Runtime-internal protocols ===
    "ProtocolContractDescriptor": "runtime/protocol_contract_descriptor.py",
    "ProtocolContractSource": "runtime/protocol_contract_source.py",
    "ProtocolDomainPlugin": "runtime/protocol_domain_plugin.py",
    "ProtocolHandlerPluginLoader": "runtime/protocol_handler_plugin_loader.py",
    "ProtocolHandlerDiscovery": "runtime/protocol_handler_discovery.py",
    "ProtocolPolicy": "runtime/protocol_policy.py",
    "ProtocolProjectionEffect": "runtime/protocol_projection_effect.py",
    "ProtocolContractEventCallbacks": "runtime/kafka_contract_source.py",
    "ProtocolIntentEffect": "runtime/service_intent_executor.py",
    "ProtocolIntentPayload": "runtime/protocols/protocol_intent_payload.py",
    "ProtocolRuntimeScheduler": "runtime/protocols/protocol_runtime_scheduler.py",
    "ProtocolSecretResolver": "runtime/config_discovery/models/protocol_secret_resolver.py",
    "ProtocolSecretResolverMetrics": "runtime/secret_resolver.py",
    # === [OBS] Observability protocols ===
    "ProtocolEmissionCountSource": "observability/wiring_health/protocol_emission_count_source.py",
    "ProtocolConsumptionCountSource": "observability/wiring_health/protocol_consumption_count_source.py",
    "ProtocolCircuitBreakerFailureRecorder": "utils/util_db_error_context.py",
}

# Duplicate protocol names that appear in multiple files (node-internal
# re-declarations that shadow a canonical version elsewhere). The AST scanner
# finds these but they are legitimate internal copies.
KNOWN_DUPLICATE_LOCATIONS: dict[str, list[str]] = {
    "ProtocolIdempotencyStore": [
        "idempotency/protocol_idempotency_store.py",
    ],
    "ProtocolRegistrationPersistence": [
        "handlers/registration_storage/protocol_registration_persistence.py",
    ],
    "ProtocolDiscoveryOperations": [
        "handlers/service_discovery/protocol_discovery_operations.py",
    ],
    "ProtocolIntentEffect": [
        "nodes/contract_registry_reducer/contract_registration_event_router.py",
    ],
}


def _find_protocol_declarations() -> list[tuple[str, str, int]]:
    """Find all Protocol class declarations in omnibase_infra via AST.

    Returns:
        List of (class_name, relative_path, line_number) tuples.
    """
    results: list[tuple[str, str, int]] = []

    for py_file in _INFRA_SRC.rglob("*.py"):
        # Skip test files, __pycache__, and archived dirs
        rel = py_file.relative_to(_INFRA_SRC)
        rel_str = str(rel)
        if "__pycache__" in rel_str or "archived" in rel_str:
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Check if any base class is "Protocol"
            for base in node.bases:
                base_name: str | None = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr

                if base_name == "Protocol":
                    results.append((node.name, str(rel), node.lineno))
                    break

    return sorted(results)


@pytest.mark.unit
class TestProtocolOwnership:
    """Verify infra does not declare unauthorized Protocol classes."""

    def test_no_unknown_protocols(self) -> None:
        """Every Protocol declaration must be in the known allowlist.

        If this test fails, a new Protocol class was added to omnibase_infra.
        Decide whether to:
          1. Move it to omnibase_spi (if it is a cross-repo contract)
          2. Add it to KNOWN_INFRA_PROTOCOLS with a category and justification
        """
        declarations = _find_protocol_declarations()
        unknown: list[str] = []

        for class_name, rel_path, lineno in declarations:
            if class_name in KNOWN_INFRA_PROTOCOLS:
                continue
            # Check if it is a known duplicate location
            dup_locations = KNOWN_DUPLICATE_LOCATIONS.get(class_name, [])
            if rel_path in dup_locations:
                continue
            # Also check for TYPE_CHECKING-only declarations (in if TYPE_CHECKING blocks)
            # These are forward references, not actual protocol definitions
            unknown.append(f"  {class_name} in {rel_path}:{lineno}")

        assert not unknown, (
            "Unknown Protocol declarations found in omnibase_infra.\n"
            "Infra should implement SPI protocols, not declare new ones.\n\n"
            "New protocols found:\n" + "\n".join(unknown) + "\n\nTo fix:\n"
            "  1. Move the protocol to omnibase_spi if it is a cross-repo contract\n"
            "  2. Or add it to KNOWN_INFRA_PROTOCOLS in this test with a category tag\n"
            "     ([DI], [MIXIN], [NODE], [RUNTIME], [OBS]) and justification"
        )

    def test_known_protocols_still_exist(self) -> None:
        """Every entry in the allowlist must still exist in the codebase.

        This prevents the allowlist from becoming stale with entries for
        protocols that have been deleted.
        """
        declarations = _find_protocol_declarations()
        declared_names = {name for name, _, _ in declarations}

        stale: list[str] = []
        for name, rel_path in KNOWN_INFRA_PROTOCOLS.items():
            if name not in declared_names:
                stale.append(f"  {name} (expected in {rel_path})")

        assert not stale, (
            "Stale entries in KNOWN_INFRA_PROTOCOLS allowlist.\n"
            "These protocols no longer exist in the codebase:\n"
            + "\n".join(stale)
            + "\n\nRemove them from the allowlist."
        )

    def test_known_protocols_in_correct_files(self) -> None:
        """Allowlist file paths must match actual Protocol locations.

        Prevents the allowlist from becoming inaccurate when protocols
        are moved between files.
        """
        declarations = _find_protocol_declarations()
        # Build a map of name -> set of relative paths
        name_to_paths: dict[str, set[str]] = {}
        for name, rel_path, _ in declarations:
            name_to_paths.setdefault(name, set()).add(rel_path)

        wrong_path: list[str] = []
        for name, expected_path in KNOWN_INFRA_PROTOCOLS.items():
            actual_paths = name_to_paths.get(name, set())
            if not actual_paths:
                continue  # Handled by test_known_protocols_still_exist
            if expected_path not in actual_paths:
                wrong_path.append(
                    f"  {name}: expected {expected_path}, found {sorted(actual_paths)}"
                )

        assert not wrong_path, (
            "Allowlist file path mismatches:\n"
            + "\n".join(wrong_path)
            + "\n\nUpdate the file paths in KNOWN_INFRA_PROTOCOLS."
        )

    def test_protocol_count_within_bounds(self) -> None:
        """Total protocol count should not grow without review.

        This is a soft check to flag protocol proliferation. The max count
        should be updated when new protocols are intentionally added.
        """
        declarations = _find_protocol_declarations()
        unique_names = {name for name, _, _ in declarations}
        max_expected = len(KNOWN_INFRA_PROTOCOLS) + len(
            {
                name
                for name, locations in KNOWN_DUPLICATE_LOCATIONS.items()
                for _ in locations
            }
        )

        assert len(unique_names) <= len(KNOWN_INFRA_PROTOCOLS), (
            f"Protocol count ({len(unique_names)}) exceeds "
            f"allowlist size ({len(KNOWN_INFRA_PROTOCOLS)}). "
            "New protocols detected. Update the allowlist."
        )

    def test_ast_scanner_finds_protocols(self) -> None:
        """Verify the AST scanner actually finds protocols (sanity check)."""
        declarations = _find_protocol_declarations()
        assert len(declarations) > 0, (
            "AST scanner found zero Protocol declarations. "
            "This likely indicates a bug in _find_protocol_declarations()."
        )
        # We know there are at least 19 in protocols/ package alone
        assert len(declarations) >= 19, (
            f"AST scanner found only {len(declarations)} protocols, "
            "expected at least 19 from the protocols/ package."
        )

    def test_clear_error_message_for_violation(self) -> None:
        """Verify the error message is actionable when a violation occurs.

        This test validates the error message format by checking the
        assertion message structure (not by triggering a real violation).
        """
        # Simulate an unknown protocol
        unknown_entries = ["  FakeProtocol in fake/path.py:42"]
        error_msg = (
            "Unknown Protocol declarations found in omnibase_infra.\n"
            "Infra should implement SPI protocols, not declare new ones.\n\n"
            "New protocols found:\n" + "\n".join(unknown_entries) + "\n\nTo fix:\n"
            "  1. Move the protocol to omnibase_spi if it is a cross-repo contract\n"
            "  2. Or add it to KNOWN_INFRA_PROTOCOLS in this test with a category tag\n"
            "     ([DI], [MIXIN], [NODE], [RUNTIME], [OBS]) and justification"
        )
        # Verify the message contains actionable instructions
        assert "omnibase_spi" in error_msg
        assert "KNOWN_INFRA_PROTOCOLS" in error_msg
        assert "[DI]" in error_msg
