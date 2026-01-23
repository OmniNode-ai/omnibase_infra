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

    def test_first_match_wins(self) -> None:
        """Only the first matching pattern is used (early exit behavior).

        Intent types should belong to a single capability category. When an intent
        contains multiple pattern prefixes (e.g., "postgres.kafka.hybrid"), only the
        first matching pattern is applied.
        """
        rules = CapabilityInferenceRules()
        # "postgres.kafka.hybrid" starts with "postgres." so matches postgres only
        # Even though "kafka." appears in the intent, it's not at the start
        result = rules.infer_from_intent_types(["postgres.kafka.hybrid"])
        assert result == ["postgres.storage"]
        # Verify kafka.messaging is NOT included (first match wins)
        assert "kafka.messaging" not in result


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

    def test_protocol_with_version_suffix_no_match(self) -> None:
        """Protocol with version suffix should NOT match known protocol.

        This verifies that "ProtocolReducerV2" does NOT match "ProtocolReducer"
        because the string ends with "V2", not "ProtocolReducer".
        """
        rules = CapabilityInferenceRules()
        # ProtocolReducerV2 ends with "V2", not "ProtocolReducer"
        result = rules.infer_from_protocols(["ProtocolReducerV2"])
        assert result == []
        # Also test other version patterns
        result = rules.infer_from_protocols(["ProtocolDatabaseAdapterV3"])
        assert result == []
        result = rules.infer_from_protocols(["ProtocolEventBus2"])
        assert result == []

    def test_multiple_suffix_words_no_match(self) -> None:
        """Protocol with multiple suffix words should NOT match known protocol.

        This verifies that "ProtocolReducerExtendedVersion" does NOT match
        "ProtocolReducer" because the string ends with "Version", not "ProtocolReducer".
        """
        rules = CapabilityInferenceRules()
        # Multiple suffix words after known protocol name
        result = rules.infer_from_protocols(["ProtocolReducerExtendedVersion"])
        assert result == []
        result = rules.infer_from_protocols(["ProtocolDatabaseAdapterEnhancedImpl"])
        assert result == []
        result = rules.infer_from_protocols(["ProtocolEventBusAsyncWrapper"])
        assert result == []
        # Verify that proper suffix still works for contrast
        result = rules.infer_from_protocols(["CustomProtocolReducer"])
        assert result == ["state.reducer"]


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


class TestDependencyInjection:
    """Tests for dependency injection of custom rule mappings."""

    def test_default_rules_when_no_args(self) -> None:
        """Constructor with no args should use default rules."""
        rules = CapabilityInferenceRules()
        # Verify default intent patterns work
        result = rules.infer_from_intent_types(["postgres.upsert"])
        assert result == ["postgres.storage"]

    def test_custom_intent_patterns_extend_defaults(self) -> None:
        """Custom intent patterns should extend default patterns."""
        rules = CapabilityInferenceRules(intent_patterns={"redis.": "redis.caching"})
        # New custom pattern should work
        result = rules.infer_from_intent_types(["redis.get"])
        assert result == ["redis.caching"]
        # Default patterns should still work
        result = rules.infer_from_intent_types(["postgres.upsert"])
        assert result == ["postgres.storage"]

    def test_custom_intent_patterns_override_defaults(self) -> None:
        """Custom intent patterns should override default patterns when key matches."""
        rules = CapabilityInferenceRules(
            intent_patterns={"postgres.": "custom.database"}
        )
        result = rules.infer_from_intent_types(["postgres.upsert"])
        assert result == ["custom.database"]
        # Other defaults should still work
        result = rules.infer_from_intent_types(["consul.register"])
        assert result == ["consul.registration"]

    def test_custom_protocol_tags_extend_defaults(self) -> None:
        """Custom protocol tags should extend default tags."""
        rules = CapabilityInferenceRules(
            protocol_tags={"ProtocolCustom": "custom.protocol"}
        )
        # New custom protocol should work
        result = rules.infer_from_protocols(["ProtocolCustom"])
        assert result == ["custom.protocol"]
        # Default protocols should still work
        result = rules.infer_from_protocols(["ProtocolReducer"])
        assert result == ["state.reducer"]

    def test_custom_protocol_tags_override_defaults(self) -> None:
        """Custom protocol tags should override defaults when key matches."""
        rules = CapabilityInferenceRules(
            protocol_tags={"ProtocolReducer": "custom.reducer"}
        )
        result = rules.infer_from_protocols(["ProtocolReducer"])
        assert result == ["custom.reducer"]
        # Other defaults should still work
        result = rules.infer_from_protocols(["ProtocolEventBus"])
        assert result == ["event.bus"]

    def test_custom_node_type_tags_extend_defaults(self) -> None:
        """Custom node type tags should extend default tags."""
        rules = CapabilityInferenceRules(node_type_tags={"gateway": "node.gateway"})
        # New custom node type should work
        result = rules.infer_from_node_type("gateway")
        assert result == ["node.gateway"]
        # Default node types should still work
        result = rules.infer_from_node_type("effect")
        assert result == ["node.effect"]

    def test_custom_node_type_tags_override_defaults(self) -> None:
        """Custom node type tags should override defaults when key matches."""
        rules = CapabilityInferenceRules(node_type_tags={"effect": "custom.effect"})
        result = rules.infer_from_node_type("effect")
        assert result == ["custom.effect"]
        # Other defaults should still work
        result = rules.infer_from_node_type("reducer")
        assert result == ["node.reducer"]

    def test_multiple_custom_rules_combined(self) -> None:
        """Multiple custom rules can be combined."""
        rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"},
            protocol_tags={"ProtocolCustom": "custom.protocol"},
            node_type_tags={"gateway": "node.gateway"},
        )
        # All custom rules should work
        result = rules.infer_all(
            intent_types=["redis.get"],
            protocols=["ProtocolCustom"],
            node_type="gateway",
        )
        assert "redis.caching" in result
        assert "custom.protocol" in result
        assert "node.gateway" in result

    def test_empty_custom_rules_use_defaults(self) -> None:
        """Empty custom rules should use defaults."""
        rules = CapabilityInferenceRules(
            intent_patterns={},
            protocol_tags={},
            node_type_tags={},
        )
        result = rules.infer_from_intent_types(["postgres.upsert"])
        assert result == ["postgres.storage"]

    def test_none_custom_rules_use_defaults(self) -> None:
        """None values for custom rules should use defaults."""
        rules = CapabilityInferenceRules(
            intent_patterns=None,
            protocol_tags=None,
            node_type_tags=None,
        )
        result = rules.infer_from_intent_types(["postgres.upsert"])
        assert result == ["postgres.storage"]

    def test_custom_rules_isolated_between_instances(self) -> None:
        """Custom rules should be isolated between instances."""
        rules1 = CapabilityInferenceRules(intent_patterns={"redis.": "redis.caching"})
        rules2 = CapabilityInferenceRules()

        # rules1 should have custom pattern
        result1 = rules1.infer_from_intent_types(["redis.get"])
        assert result1 == ["redis.caching"]

        # rules2 should NOT have custom pattern
        result2 = rules2.infer_from_intent_types(["redis.get"])
        assert result2 == []

    def test_default_constants_unchanged(self) -> None:
        """Creating instances with custom rules should not modify DEFAULT_* constants."""
        original_intent = dict(CapabilityInferenceRules.DEFAULT_INTENT_PATTERNS)
        original_protocol = dict(CapabilityInferenceRules.DEFAULT_PROTOCOL_TAGS)
        original_node = dict(CapabilityInferenceRules.DEFAULT_NODE_TYPE_TAGS)

        # Create instance with custom rules
        _ = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"},
            protocol_tags={"ProtocolCustom": "custom.protocol"},
            node_type_tags={"gateway": "node.gateway"},
        )

        # Verify DEFAULT_* constants are unchanged
        assert original_intent == CapabilityInferenceRules.DEFAULT_INTENT_PATTERNS
        assert original_protocol == CapabilityInferenceRules.DEFAULT_PROTOCOL_TAGS
        assert original_node == CapabilityInferenceRules.DEFAULT_NODE_TYPE_TAGS

    def test_infer_all_with_custom_rules(self) -> None:
        """infer_all should work correctly with custom rules."""
        rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching", "postgres.": "custom.db"},
            protocol_tags={"ProtocolReducer": "custom.reducer"},
            node_type_tags={"effect": "custom.effect"},
        )
        result = rules.infer_all(
            intent_types=["redis.get", "postgres.upsert"],
            protocols=["ProtocolReducer"],
            node_type="effect",
        )
        assert "redis.caching" in result
        assert "custom.db" in result
        assert "custom.reducer" in result
        assert "custom.effect" in result
        # Should be sorted
        assert result == sorted(result)


class TestDefaultConstants:
    """Tests for DEFAULT_* class constants."""

    def test_default_intent_patterns_exist(self) -> None:
        """DEFAULT_INTENT_PATTERNS should exist and have expected keys."""
        assert hasattr(CapabilityInferenceRules, "DEFAULT_INTENT_PATTERNS")
        patterns = CapabilityInferenceRules.DEFAULT_INTENT_PATTERNS
        assert "postgres." in patterns
        assert "consul." in patterns
        assert "kafka." in patterns
        assert "vault." in patterns
        assert "valkey." in patterns
        assert "http." in patterns

    def test_default_protocol_tags_exist(self) -> None:
        """DEFAULT_PROTOCOL_TAGS should exist and have expected keys."""
        assert hasattr(CapabilityInferenceRules, "DEFAULT_PROTOCOL_TAGS")
        tags = CapabilityInferenceRules.DEFAULT_PROTOCOL_TAGS
        assert "ProtocolReducer" in tags
        assert "ProtocolDatabaseAdapter" in tags
        assert "ProtocolEventBus" in tags
        assert "ProtocolCacheAdapter" in tags
        assert "ProtocolServiceDiscovery" in tags

    def test_default_node_type_tags_exist(self) -> None:
        """DEFAULT_NODE_TYPE_TAGS should exist and have expected keys."""
        assert hasattr(CapabilityInferenceRules, "DEFAULT_NODE_TYPE_TAGS")
        tags = CapabilityInferenceRules.DEFAULT_NODE_TYPE_TAGS
        assert "effect" in tags
        assert "compute" in tags
        assert "reducer" in tags
        assert "orchestrator" in tags

    def test_default_intent_patterns_values(self) -> None:
        """DEFAULT_INTENT_PATTERNS should have expected values."""
        patterns = CapabilityInferenceRules.DEFAULT_INTENT_PATTERNS
        assert patterns["postgres."] == "postgres.storage"
        assert patterns["consul."] == "consul.registration"
        assert patterns["kafka."] == "kafka.messaging"
        assert patterns["vault."] == "vault.secrets"
        assert patterns["valkey."] == "valkey.caching"
        assert patterns["http."] == "http.transport"

    def test_default_protocol_tags_values(self) -> None:
        """DEFAULT_PROTOCOL_TAGS should have expected values."""
        tags = CapabilityInferenceRules.DEFAULT_PROTOCOL_TAGS
        assert tags["ProtocolReducer"] == "state.reducer"
        assert tags["ProtocolDatabaseAdapter"] == "database.adapter"
        assert tags["ProtocolEventBus"] == "event.bus"
        assert tags["ProtocolCacheAdapter"] == "cache.adapter"
        assert tags["ProtocolServiceDiscovery"] == "service.discovery"

    def test_default_node_type_tags_values(self) -> None:
        """DEFAULT_NODE_TYPE_TAGS should have expected values."""
        tags = CapabilityInferenceRules.DEFAULT_NODE_TYPE_TAGS
        assert tags["effect"] == "node.effect"
        assert tags["compute"] == "node.compute"
        assert tags["reducer"] == "node.reducer"
        assert tags["orchestrator"] == "node.orchestrator"
