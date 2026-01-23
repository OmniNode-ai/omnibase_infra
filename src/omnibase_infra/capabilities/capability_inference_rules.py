"""Capability inference rules for deriving tags from contract structure.

This module provides stateless, deterministic pattern matching to infer
capability tags from contract fields like intent_types and protocols.
"""

from __future__ import annotations


class CapabilityInferenceRules:
    """Code-driven capability inference rules.

    Stateless pure functions for deterministic pattern matching.
    Rules infer capability_tags from contract structure.
    """

    # Intent pattern -> capability tag mappings
    INTENT_PATTERNS: dict[str, str] = {
        "postgres.": "postgres.storage",
        "consul.": "consul.registration",
        "kafka.": "kafka.messaging",
        "vault.": "vault.secrets",
        "valkey.": "valkey.caching",
        "http.": "http.transport",
    }

    # Protocol -> capability tag mappings
    PROTOCOL_TAGS: dict[str, str] = {
        "ProtocolReducer": "state.reducer",
        "ProtocolDatabaseAdapter": "database.adapter",
        "ProtocolEventBus": "event.bus",
        "ProtocolCacheAdapter": "cache.adapter",
        "ProtocolServiceDiscovery": "service.discovery",
    }

    # Node type -> base capability tag
    NODE_TYPE_TAGS: dict[str, str] = {
        "effect": "node.effect",
        "compute": "node.compute",
        "reducer": "node.reducer",
        "orchestrator": "node.orchestrator",
    }

    def infer_from_intent_types(self, intent_types: list[str]) -> list[str]:
        """Infer capability tags from intent type patterns.

        Args:
            intent_types: List of intent type strings (e.g., ["postgres.upsert", "consul.register"])

        Returns:
            Sorted list of inferred capability tags (deduplicated)
        """
        tags: set[str] = set()
        for intent in intent_types:
            if intent is None:  # Skip None values
                continue
            for pattern, tag in self.INTENT_PATTERNS.items():
                if intent.startswith(pattern):
                    tags.add(tag)
                    break
        return sorted(tags)

    def infer_from_protocols(self, protocols: list[str]) -> list[str]:
        """Infer capability tags from protocol names.

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
            if protocol in self.PROTOCOL_TAGS:
                tags.add(self.PROTOCOL_TAGS[protocol])
            # Also check if protocol name ends with known suffix
            for known_protocol, tag in self.PROTOCOL_TAGS.items():
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
        if normalized in self.NODE_TYPE_TAGS:
            return [self.NODE_TYPE_TAGS[normalized]]
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
