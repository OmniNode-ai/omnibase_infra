"""Unit tests for CapabilityInferenceRules.

Tests the code-driven capability inference engine.
All tests should verify deterministic, sorted output.
"""

from __future__ import annotations

import pytest

from omnibase_infra.capabilities import CapabilityInferenceRules


class TestInferFromIntentTypes:
    """Tests for intent type pattern matching."""

    def test_postgres_intent_pattern(self) -> None:
        """postgres.* intents should infer postgres.storage tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["postgres.upsert", "postgres.query"])
        assert result == ["postgres.storage"]

    def test_consul_intent_pattern(self) -> None:
        """consul.* intents should infer consul.registration tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["consul.register", "consul.deregister"])
        assert result == ["consul.registration"]

    def test_kafka_intent_pattern(self) -> None:
        """kafka.* intents should infer kafka.messaging tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["kafka.produce", "kafka.consume"])
        assert result == ["kafka.messaging"]

    def test_vault_intent_pattern(self) -> None:
        """vault.* intents should infer vault.secrets tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["vault.read", "vault.write"])
        assert result == ["vault.secrets"]

    def test_valkey_intent_pattern(self) -> None:
        """valkey.* intents should infer valkey.caching tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["valkey.get", "valkey.set"])
        assert result == ["valkey.caching"]

    def test_http_intent_pattern(self) -> None:
        """http.* intents should infer http.transport tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["http.get", "http.post"])
        assert result == ["http.transport"]

    def test_multiple_patterns(self) -> None:
        """Multiple intent patterns should return multiple tags."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["postgres.upsert", "consul.register"])
        assert result == ["consul.registration", "postgres.storage"]  # sorted

    def test_all_patterns_combined(self) -> None:
        """All intent patterns combined should return all tags sorted."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(
            [
                "postgres.upsert",
                "consul.register",
                "kafka.produce",
                "vault.read",
                "valkey.get",
                "http.post",
            ]
        )
        expected = [
            "consul.registration",
            "http.transport",
            "kafka.messaging",
            "postgres.storage",
            "valkey.caching",
            "vault.secrets",
        ]
        assert result == expected

    def test_unrecognized_pattern(self) -> None:
        """Unrecognized patterns should return empty list."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["unknown.intent"])
        assert result == []

    def test_empty_list(self) -> None:
        """Empty input should return empty list."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types([])
        assert result == []

    def test_deduplication(self) -> None:
        """Duplicate patterns should be deduplicated."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(
            [
                "postgres.upsert",
                "postgres.query",
                "postgres.delete",
            ]
        )
        assert result == ["postgres.storage"]  # Only one tag

    def test_mixed_recognized_unrecognized(self) -> None:
        """Mix of recognized and unrecognized patterns should only return recognized tags."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(
            [
                "postgres.upsert",
                "unknown.intent",
                "consul.register",
                "another.unknown",
            ]
        )
        assert result == ["consul.registration", "postgres.storage"]

    def test_case_sensitive_patterns(self) -> None:
        """Intent patterns should be case-sensitive (startswith match)."""
        rules = CapabilityInferenceRules()
        # Uppercase should not match
        result = rules.infer_from_intent_types(["POSTGRES.upsert"])
        assert result == []
        # Correct case should match
        result = rules.infer_from_intent_types(["postgres.upsert"])
        assert result == ["postgres.storage"]

    def test_partial_prefix_no_match(self) -> None:
        """Patterns that don't start with known prefix should not match."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["mypostgres.upsert"])
        assert result == []

    def test_exact_prefix_match(self) -> None:
        """Exact prefix 'postgres.' should match, not just 'postgres'."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["postgresext.upsert"])
        assert result == []  # No dot after postgres


class TestInferFromProtocols:
    """Tests for protocol name matching."""

    def test_reducer_protocol(self) -> None:
        """ProtocolReducer should infer state.reducer tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(["ProtocolReducer"])
        assert result == ["state.reducer"]

    def test_database_adapter_protocol(self) -> None:
        """ProtocolDatabaseAdapter should infer database.adapter tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(["ProtocolDatabaseAdapter"])
        assert result == ["database.adapter"]

    def test_event_bus_protocol(self) -> None:
        """ProtocolEventBus should infer event.bus tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(["ProtocolEventBus"])
        assert result == ["event.bus"]

    def test_cache_adapter_protocol(self) -> None:
        """ProtocolCacheAdapter should infer cache.adapter tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(["ProtocolCacheAdapter"])
        assert result == ["cache.adapter"]

    def test_service_discovery_protocol(self) -> None:
        """ProtocolServiceDiscovery should infer service.discovery tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(["ProtocolServiceDiscovery"])
        assert result == ["service.discovery"]

    def test_all_protocols_combined(self) -> None:
        """All protocols combined should return all tags sorted."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(
            [
                "ProtocolReducer",
                "ProtocolDatabaseAdapter",
                "ProtocolEventBus",
                "ProtocolCacheAdapter",
                "ProtocolServiceDiscovery",
            ]
        )
        expected = [
            "cache.adapter",
            "database.adapter",
            "event.bus",
            "service.discovery",
            "state.reducer",
        ]
        assert result == expected

    def test_partial_match_suffix(self) -> None:
        """Protocol names ending with known suffix should match."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(["MyCustomProtocolReducer"])
        assert result == ["state.reducer"]

    def test_multiple_suffix_matches(self) -> None:
        """Multiple custom protocols with known suffixes should match."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(
            [
                "CustomProtocolReducer",
                "MyProtocolDatabaseAdapter",
            ]
        )
        assert result == ["database.adapter", "state.reducer"]

    def test_empty_list(self) -> None:
        """Empty input should return empty list."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols([])
        assert result == []

    def test_unrecognized_protocol(self) -> None:
        """Unrecognized protocol should return empty list."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(["SomeRandomProtocol"])
        assert result == []

    def test_deduplication_from_exact_and_suffix(self) -> None:
        """Same tag from exact match and suffix should be deduplicated."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(
            [
                "ProtocolReducer",
                "CustomProtocolReducer",
            ]
        )
        assert result == ["state.reducer"]  # Only one tag

    def test_mixed_recognized_unrecognized(self) -> None:
        """Mix of recognized and unrecognized protocols should only return recognized tags."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_protocols(
            [
                "ProtocolReducer",
                "UnknownProtocol",
                "ProtocolEventBus",
            ]
        )
        assert result == ["event.bus", "state.reducer"]

    def test_case_sensitive_protocols(self) -> None:
        """Protocol names should be case-sensitive."""
        rules = CapabilityInferenceRules()
        # Wrong case should not match
        result = rules.infer_from_protocols(["protocolreducer"])
        assert result == []
        # Correct case should match
        result = rules.infer_from_protocols(["ProtocolReducer"])
        assert result == ["state.reducer"]

    def test_prefix_only_no_match(self) -> None:
        """Protocol starting with known name but not ending should not suffix-match."""
        rules = CapabilityInferenceRules()
        # ProtocolReducerExtended does not end with ProtocolReducer
        result = rules.infer_from_protocols(["ProtocolReducerExtended"])
        assert result == []


class TestInferFromNodeType:
    """Tests for node type inference."""

    def test_effect_node_type(self) -> None:
        """Effect node type should infer node.effect tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("effect")
        assert result == ["node.effect"]

    def test_effect_generic_node_type(self) -> None:
        """EFFECT_GENERIC should normalize and infer node.effect tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("EFFECT_GENERIC")
        assert result == ["node.effect"]

    def test_compute_node_type(self) -> None:
        """Compute node type should infer node.compute tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("compute")
        assert result == ["node.compute"]

    def test_compute_generic_node_type(self) -> None:
        """COMPUTE_GENERIC should normalize and infer node.compute tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("COMPUTE_GENERIC")
        assert result == ["node.compute"]

    def test_reducer_node_type(self) -> None:
        """Reducer node type should infer node.reducer tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("reducer")
        assert result == ["node.reducer"]

    def test_reducer_generic_node_type(self) -> None:
        """REDUCER_GENERIC should normalize and infer node.reducer tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("REDUCER_GENERIC")
        assert result == ["node.reducer"]

    def test_orchestrator_node_type(self) -> None:
        """Orchestrator node type should infer node.orchestrator tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("orchestrator")
        assert result == ["node.orchestrator"]

    def test_orchestrator_generic_node_type(self) -> None:
        """ORCHESTRATOR_GENERIC should normalize and infer node.orchestrator tag."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("ORCHESTRATOR_GENERIC")
        assert result == ["node.orchestrator"]

    def test_unknown_node_type(self) -> None:
        """Unknown node type should return empty list."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("unknown")
        assert result == []

    def test_uppercase_node_type(self) -> None:
        """Uppercase node types should normalize correctly."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("EFFECT")
        assert result == ["node.effect"]

    def test_mixed_case_node_type(self) -> None:
        """Mixed case node types should normalize correctly."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("Effect")
        assert result == ["node.effect"]

    def test_empty_node_type(self) -> None:
        """Empty node type should return empty list."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_node_type("")
        assert result == []


class TestInferAll:
    """Tests for combined inference."""

    def test_all_sources_combined(self) -> None:
        """infer_all should combine tags from all sources."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(
            intent_types=["postgres.upsert"],
            protocols=["ProtocolReducer"],
            node_type="reducer",
        )
        # Should have: postgres.storage, state.reducer, node.reducer
        assert "postgres.storage" in result
        assert "state.reducer" in result
        assert "node.reducer" in result
        assert result == sorted(result)  # Must be sorted

    def test_none_inputs(self) -> None:
        """None inputs should be handled gracefully."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(
            intent_types=None,
            protocols=None,
            node_type=None,
        )
        assert result == []

    def test_partial_inputs_intent_only(self) -> None:
        """Partial inputs with only intent_types should work correctly."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(intent_types=["consul.register"])
        assert result == ["consul.registration"]

    def test_partial_inputs_protocols_only(self) -> None:
        """Partial inputs with only protocols should work correctly."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(protocols=["ProtocolEventBus"])
        assert result == ["event.bus"]

    def test_partial_inputs_node_type_only(self) -> None:
        """Partial inputs with only node_type should work correctly."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(node_type="orchestrator")
        assert result == ["node.orchestrator"]

    def test_deterministic_output(self) -> None:
        """Same input should always produce same output (deterministic)."""
        rules = CapabilityInferenceRules()
        input_data = {
            "intent_types": ["postgres.upsert", "consul.register"],
            "protocols": ["ProtocolReducer"],
            "node_type": "effect",
        }
        result1 = rules.infer_all(**input_data)
        result2 = rules.infer_all(**input_data)
        assert result1 == result2

    def test_deduplication_across_sources(self) -> None:
        """Tags appearing from multiple sources should be deduplicated."""
        rules = CapabilityInferenceRules()
        # If somehow intent_types and protocols infer same tag
        # (currently no overlap, but testing deduplication behavior)
        result = rules.infer_all(
            intent_types=["postgres.upsert", "postgres.query"],
            protocols=["ProtocolReducer", "ProtocolReducer"],
            node_type="reducer",
        )
        # Check no duplicates
        assert len(result) == len(set(result))

    def test_all_sources_with_all_patterns(self) -> None:
        """Comprehensive test with all patterns from all sources."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(
            intent_types=[
                "postgres.upsert",
                "consul.register",
                "kafka.produce",
                "vault.read",
                "valkey.get",
                "http.post",
            ],
            protocols=[
                "ProtocolReducer",
                "ProtocolDatabaseAdapter",
                "ProtocolEventBus",
                "ProtocolCacheAdapter",
                "ProtocolServiceDiscovery",
            ],
            node_type="orchestrator",
        )
        expected = sorted(
            [
                # Intent types
                "postgres.storage",
                "consul.registration",
                "kafka.messaging",
                "vault.secrets",
                "valkey.caching",
                "http.transport",
                # Protocols
                "state.reducer",
                "database.adapter",
                "event.bus",
                "cache.adapter",
                "service.discovery",
                # Node type
                "node.orchestrator",
            ]
        )
        assert result == expected

    def test_empty_lists_are_falsy(self) -> None:
        """Empty lists should be treated as falsy (not processed)."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(
            intent_types=[],
            protocols=[],
            node_type="effect",
        )
        # Empty lists are falsy in Python, but the current implementation
        # checks truthiness, so empty list won't add anything
        assert result == ["node.effect"]


class TestDeterminism:
    """Tests to verify deterministic, sorted output."""

    def test_output_is_always_sorted(self) -> None:
        """All output lists must be sorted."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(
            intent_types=["vault.read", "kafka.produce", "postgres.upsert"],
            protocols=["ProtocolEventBus", "ProtocolReducer"],
            node_type="orchestrator",
        )
        assert result == sorted(result)

    def test_no_duplicates(self) -> None:
        """Output should never contain duplicates."""
        rules = CapabilityInferenceRules()
        result = rules.infer_all(
            intent_types=["postgres.upsert", "postgres.query"],
            protocols=["ProtocolDatabaseAdapter"],
            node_type="effect",
        )
        assert len(result) == len(set(result))

    def test_multiple_runs_same_result(self) -> None:
        """Multiple runs with same input should produce identical results."""
        rules = CapabilityInferenceRules()
        inputs = {
            "intent_types": ["vault.read", "http.post", "consul.register"],
            "protocols": ["ProtocolCacheAdapter", "ProtocolReducer"],
            "node_type": "reducer",
        }
        results = [rules.infer_all(**inputs) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_order_independent_input(self) -> None:
        """Input order should not affect output."""
        rules = CapabilityInferenceRules()
        result1 = rules.infer_from_intent_types(
            [
                "postgres.upsert",
                "consul.register",
                "kafka.produce",
            ]
        )
        result2 = rules.infer_from_intent_types(
            [
                "kafka.produce",
                "postgres.upsert",
                "consul.register",
            ]
        )
        assert result1 == result2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_whitespace_in_intent_types(self) -> None:
        """Intent types with whitespace should not match (exact prefix)."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types([" postgres.upsert"])
        assert result == []  # Leading space prevents match

    def test_empty_string_intent_type(self) -> None:
        """Empty string in intent types should not cause errors."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types([""])
        assert result == []

    def test_special_characters_in_intent(self) -> None:
        """Special characters in intent types should be handled."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["postgres.$special"])
        assert result == ["postgres.storage"]  # Still matches prefix

    def test_very_long_intent_type(self) -> None:
        """Very long intent types should still match prefix."""
        rules = CapabilityInferenceRules()
        long_intent = "postgres." + "a" * 1000
        result = rules.infer_from_intent_types([long_intent])
        assert result == ["postgres.storage"]

    def test_unicode_in_intent_type(self) -> None:
        """Unicode characters in intent types should be handled."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["postgres.upsert_\u00e9"])
        assert result == ["postgres.storage"]

    def test_newline_in_intent_type(self) -> None:
        """Newline in intent type should not match."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["postgres\n.upsert"])
        assert result == []

    def test_singleton_pattern(self) -> None:
        """Single-item lists should work correctly."""
        rules = CapabilityInferenceRules()
        result = rules.infer_from_intent_types(["kafka.produce"])
        assert result == ["kafka.messaging"]
        result = rules.infer_from_protocols(["ProtocolReducer"])
        assert result == ["state.reducer"]


class TestPerformance:
    """Tests for performance with large inputs."""

    def test_large_intent_list(self) -> None:
        """Large list of intents should process correctly."""
        rules = CapabilityInferenceRules()
        # Generate 1000 intents
        intents = [f"postgres.operation_{i}" for i in range(1000)]
        result = rules.infer_from_intent_types(intents)
        assert result == ["postgres.storage"]  # All deduplicated to one

    def test_large_protocol_list(self) -> None:
        """Large list of protocols should process correctly."""
        rules = CapabilityInferenceRules()
        # Generate 1000 protocols (only some will match)
        protocols = [f"Unknown{i}" for i in range(900)]
        protocols.extend(["ProtocolReducer"] * 100)
        result = rules.infer_from_protocols(protocols)
        assert result == ["state.reducer"]

    def test_large_mixed_list(self) -> None:
        """Large mixed list should process correctly and deterministically."""
        rules = CapabilityInferenceRules()
        intents = (
            [f"postgres.op_{i}" for i in range(100)]
            + [f"consul.op_{i}" for i in range(100)]
            + [f"kafka.op_{i}" for i in range(100)]
        )
        result = rules.infer_from_intent_types(intents)
        expected = ["consul.registration", "kafka.messaging", "postgres.storage"]
        assert result == expected


class TestStatelessness:
    """Tests to verify the rules engine is stateless."""

    def test_separate_instances_same_result(self) -> None:
        """Different instances should produce same results."""
        rules1 = CapabilityInferenceRules()
        rules2 = CapabilityInferenceRules()
        input_data = ["postgres.upsert", "consul.register"]
        assert rules1.infer_from_intent_types(
            input_data
        ) == rules2.infer_from_intent_types(input_data)

    def test_repeated_calls_no_side_effects(self) -> None:
        """Repeated calls should not affect subsequent calls."""
        rules = CapabilityInferenceRules()
        # First call
        rules.infer_from_intent_types(["postgres.upsert"])
        # Second call with different input
        result = rules.infer_from_intent_types(["consul.register"])
        assert result == ["consul.registration"]

    def test_instance_reuse_no_state_leakage(self) -> None:
        """Reusing instance should not leak state between calls."""
        rules = CapabilityInferenceRules()
        result1 = rules.infer_all(
            intent_types=["postgres.upsert"],
            protocols=["ProtocolReducer"],
        )
        result2 = rules.infer_all(
            intent_types=["consul.register"],
        )
        assert result2 == ["consul.registration"]
        # Verify result1 is unchanged
        assert "postgres.storage" in result1
        assert "state.reducer" in result1
