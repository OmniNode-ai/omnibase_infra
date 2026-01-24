"""Capability inference rules for deriving tags from contract structure.

This module provides deterministic pattern matching to infer capability tags
from contract fields like intent_types and protocols. Rules are injectable
via constructor for extensibility while providing sensible defaults.
"""

from __future__ import annotations


class CapabilityInferenceRules:
    """Code-driven capability inference rules with injectable patterns.

    Deterministic pattern matching for inferring capability_tags from
    contract structure. All rule mappings are injectable via constructor
    while providing sensible defaults.

    Args:
        intent_patterns: Custom/additional intent patterns (merged with defaults).
        protocol_tags: Custom/additional protocol tags (merged with defaults).
        node_type_tags: Custom/additional node type tags (merged with defaults).

    Example:
        # Use defaults
        rules = CapabilityInferenceRules()

        # Override specific pattern
        rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"}
        )

        # Override existing pattern
        rules = CapabilityInferenceRules(
            intent_patterns={"postgres.": "custom.database"}
        )
    """

    # Default intent pattern -> capability tag mappings
    DEFAULT_INTENT_PATTERNS: dict[str, str] = {
        "postgres.": "postgres.storage",
        "consul.": "consul.registration",
        "kafka.": "kafka.messaging",
        "vault.": "vault.secrets",
        "valkey.": "valkey.caching",
        "http.": "http.transport",
    }

    # Default protocol -> capability tag mappings
    DEFAULT_PROTOCOL_TAGS: dict[str, str] = {
        "ProtocolReducer": "state.reducer",
        "ProtocolDatabaseAdapter": "database.adapter",
        "ProtocolEventBus": "event.bus",
        "ProtocolCacheAdapter": "cache.adapter",
        "ProtocolServiceDiscovery": "service.discovery",
    }

    # Default node type -> base capability tag
    DEFAULT_NODE_TYPE_TAGS: dict[str, str] = {
        "effect": "node.effect",
        "compute": "node.compute",
        "reducer": "node.reducer",
        "orchestrator": "node.orchestrator",
    }

    def __init__(
        self,
        intent_patterns: dict[str, str] | None = None,
        protocol_tags: dict[str, str] | None = None,
        node_type_tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize with optional custom rules.

        Custom rules are merged with defaults. If a custom rule has the same
        key as a default rule, the custom rule takes precedence (override).

        Args:
            intent_patterns: Custom/additional intent patterns (merged with defaults).
            protocol_tags: Custom/additional protocol tags (merged with defaults).
            node_type_tags: Custom/additional node type tags (merged with defaults).
        """
        self._intent_patterns = {
            **self.DEFAULT_INTENT_PATTERNS,
            **(intent_patterns or {}),
        }
        self._protocol_tags = {
            **self.DEFAULT_PROTOCOL_TAGS,
            **(protocol_tags or {}),
        }
        self._node_type_tags = {
            **self.DEFAULT_NODE_TYPE_TAGS,
            **(node_type_tags or {}),
        }

    def infer_from_intent_types(self, intent_types: list[str]) -> list[str]:
        """Infer capability tags from intent type patterns.

        Pattern matching uses first-match-wins semantics: each intent is matched
        against patterns in iteration order, and only the FIRST matching pattern
        is used (early exit via break). This is intentional because intents should
        belong to a single capability category.

        Example:
            - "postgres.upsert" matches "postgres." -> "postgres.storage"
            - "postgres.kafka.hybrid" matches "postgres." only (NOT both postgres and kafka)

        Args:
            intent_types: List of intent type strings (e.g., ["postgres.upsert", "consul.register"])

        Returns:
            Sorted list of inferred capability tags (deduplicated)
        """
        tags: set[str] = set()
        for intent in intent_types:
            if intent is None:  # Skip None values
                continue
            for pattern, tag in self._intent_patterns.items():
                if intent.startswith(pattern):
                    tags.add(tag)
                    break
        return sorted(tags)

    def infer_from_protocols(self, protocols: list[str]) -> list[str]:
        """Infer capability tags from protocol names.

        Matching behavior:
            - Exact match: "ProtocolReducer" matches DEFAULT_PROTOCOL_TAGS["ProtocolReducer"]
            - Suffix match: "MyCustomProtocolReducer" also matches because it ends with "ProtocolReducer"
            - No match: "ProtocolReducerExtended" does NOT match (doesn't end with known protocol)

        This allows custom-prefixed protocol implementations to inherit base capability tags.

        Warning:
            Suffix matching can cause unexpected over-matching if your protocol name
            accidentally ends with a known protocol name. The matching is strict:
            the protocol name must END with the exact known protocol string.

        Examples:
            Protocols that MATCH (suffix ends with known protocol)::

                "ProtocolReducer" -> matches "ProtocolReducer" (exact match)
                "MyCustomProtocolReducer" -> matches "ProtocolReducer" (suffix match)
                "InfraProtocolDatabaseAdapter" -> matches "ProtocolDatabaseAdapter"
                "V2ProtocolEventBus" -> matches "ProtocolEventBus"

            Protocols that DO NOT MATCH (suffix has additional characters)::

                "ProtocolReducerV2" -> NO match (ends with "V2", not "ProtocolReducer")
                "ProtocolReducerExtended" -> NO match (ends with "Extended")
                "ProtocolReducerExtendedVersion" -> NO match (ends with "Version")
                "MyReducer" -> NO match (must end with full "ProtocolReducer")
                "ProtocolReducerImpl" -> NO match (ends with "Impl")

        Args:
            protocols: List of protocol class names

        Returns:
            Sorted list of inferred capability tags (deduplicated)
        """
        tags: set[str] = set()
        for protocol in protocols:
            if protocol is None:  # Skip None values
                continue
            # Check exact match
            if protocol in self._protocol_tags:
                tags.add(self._protocol_tags[protocol])
            # Also check if protocol name ends with known suffix
            for known_protocol, tag in self._protocol_tags.items():
                if protocol.endswith(known_protocol):
                    tags.add(tag)
        return sorted(tags)

    def infer_from_node_type(self, node_type: str) -> list[str]:
        """Infer base capability tag from node type.

        Args:
            node_type: Node type string (effect, compute, reducer, orchestrator)

        Returns:
            List with single node type capability tag, or empty if unknown
        """
        normalized = node_type.lower().replace("_generic", "")
        if normalized in self._node_type_tags:
            return [self._node_type_tags[normalized]]
        return []

    def infer_all(
        self,
        intent_types: list[str] | None = None,
        protocols: list[str] | None = None,
        node_type: str | None = None,
    ) -> list[str]:
        """Infer all capability tags from available contract data.

        Args:
            intent_types: Optional list of intent types
            protocols: Optional list of protocol names
            node_type: Optional node type string

        Returns:
            Sorted, deduplicated list of all inferred capability tags
        """
        tags: set[str] = set()

        if intent_types:
            tags.update(self.infer_from_intent_types(intent_types))
        if protocols:
            tags.update(self.infer_from_protocols(protocols))
        if node_type:
            tags.update(self.infer_from_node_type(node_type))

        return sorted(tags)
