# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Source Mode Enumeration for Handler Loading Strategy.

Defines the canonical source modes for handler loading in the ONEX runtime.
Each mode determines where handlers are discovered and loaded from, enabling
flexible deployment configurations and migration strategies.

Handler loading can operate in four modes:
    - BOOTSTRAP: Only use hardcoded bootstrap handlers.
                 Uses the legacy hardcoded handler registry. No YAML contract
                 discovery. Useful for minimal deployments or testing.
    - CONTRACT: Only use YAML contract-discovered handlers.
                All handlers must be defined in contract.yaml files. No
                hardcoded handlers are loaded. Enables fully declarative
                handler configuration.
    - HYBRID: Per-handler resolution with contract precedence.
              When a handler identity matches both bootstrap and contract,
              the contract-defined handler wins. Bootstrap handlers serve
              as fallback for handlers not defined in contracts.
    - KAFKA_EVENTS: Cache-only Kafka-based contract discovery.
                    Subscribes to platform-reserved contract topics and
                    maintains an in-memory cache of discovered handlers.
                    Does NOT wire business subscriptions dynamically.
                    For beta: discover + next restart applies model.

The HYBRID mode supports gradual migration from hardcoded to contract-based
handlers by allowing both sources to coexist with deterministic resolution.

See Also:
    - EnumHandlerSourceType: Defines validation error source types (different purpose)
    - HandlerPluginLoader: Uses this enum to determine loading strategy
    - ModelRuntimeConfig: Configuration model that holds the source mode setting
    - KafkaContractSource: Implements KAFKA_EVENTS mode (OMN-1654)
"""

from enum import Enum


class EnumHandlerSourceMode(str, Enum):
    """Handler source modes for handler loading strategy selection.

    These represent the different strategies for discovering and loading
    handlers at runtime. The mode determines whether handlers come from
    hardcoded registries, YAML contracts, Kafka events, or a combination.

    Attributes:
        BOOTSTRAP: Only use hardcoded bootstrap handlers.
            The runtime loads handlers from the legacy hardcoded registry only.
            No YAML contract discovery is performed. Use cases:
            - Minimal deployments without contract infrastructure
            - Testing with known handler set
            - Backwards compatibility during migration
            - Environments where contract files are not available
        CONTRACT: Only use YAML contract-discovered handlers.
            All handlers must be defined in contract.yaml files. The hardcoded
            bootstrap registry is ignored. Use cases:
            - Fully declarative handler configuration
            - Dynamic handler deployment via contracts
            - Environments requiring audit trail of handler changes
            - Production deployments with contract validation
        HYBRID: Per-handler resolution with contract precedence.
            Both bootstrap and contract sources are used. When a handler
            identity (module + class) matches in both sources, the contract-
            defined handler takes precedence. Bootstrap handlers serve as
            fallback for handlers not defined in contracts. Use cases:
            - Gradual migration from bootstrap to contract
            - Core handlers in bootstrap, extensions in contracts
            - Development environments with mixed configurations
            - A/B testing of handler implementations
        KAFKA_EVENTS: Cache-only Kafka-based contract discovery.
            Subscribes to platform-reserved contract topics (baseline-wired)
            and maintains an in-memory cache of discovered handlers. Does NOT
            wire business subscriptions dynamically. Use cases:
            - Distributed contract discovery via event bus
            - Dynamic contract registration/deregistration via Kafka
            - Beta: discover + next restart applies model
    """

    BOOTSTRAP = "bootstrap"
    CONTRACT = "contract"
    HYBRID = "hybrid"
    KAFKA_EVENTS = "kafka_events"


__all__ = ["EnumHandlerSourceMode"]
