"""Intent type extraction from ONEX contracts.

Extracts intent types from various contract structures including
event_type, consumed_events, published_events, and aggregation fields.

OMN-1136: Intent type extraction helper for ContractCapabilityExtractor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.contracts import ModelContractBase


class IntentTypeExtractor:
    """Extracts intent types from contract models.

    This class handles extraction of intent types from various contract
    structures. Different node types expose intent information in different
    locations:
    - Effect nodes: event_type.primary_events
    - Orchestrator nodes: consumed_events[].event_pattern, published_events[].event_name
    - Reducer nodes: aggregation.aggregation_functions

    This extractor is stateless and produces deterministic output.

    Example:
        extractor = IntentTypeExtractor()
        intent_types = extractor.extract_all(contract)
    """

    def extract_all(self, contract: ModelContractBase) -> list[str]:
        """Extract intent types from all applicable sources.

        Combines intent types from event_type, consumed_events,
        published_events, and aggregation fields.

        Args:
            contract: The contract model to extract intent types from.

        Returns:
            Combined list of intent types from all applicable sources.
        """
        intent_types: list[str] = []

        intent_types.extend(self._extract_from_event_type(contract))
        intent_types.extend(self._extract_from_consumed_events(contract))
        intent_types.extend(self._extract_from_published_events(contract))
        intent_types.extend(self._extract_from_aggregation(contract))

        return intent_types

    def _extract_from_event_type(self, contract: ModelContractBase) -> list[str]:
        """Extract intent types from event_type.primary_events.

        Used primarily by Effect nodes to declare the events they handle.

        Args:
            contract: The contract model to extract from.

        Returns:
            List of primary event types, or empty list if not present.
        """
        if not hasattr(contract, "event_type"):
            return []

        event_type = contract.event_type
        if event_type is None or not hasattr(event_type, "primary_events"):
            return []

        primary_events = event_type.primary_events
        if not primary_events:
            return []

        return list(primary_events)

    def _extract_from_consumed_events(self, contract: ModelContractBase) -> list[str]:
        """Extract intent types from consumed_events[].event_pattern.

        Used by Orchestrator nodes to declare which events they subscribe to.

        Args:
            contract: The contract model to extract from.

        Returns:
            List of event patterns from consumed events, or empty list if not present.
        """
        if not hasattr(contract, "consumed_events"):
            return []

        consumed_events = contract.consumed_events
        if not consumed_events:
            return []

        intent_types: list[str] = []
        for event in consumed_events:
            # ModelEventSubscription has event_pattern field
            if hasattr(event, "event_pattern") and event.event_pattern:
                intent_types.append(event.event_pattern)

        return intent_types

    def _extract_from_published_events(self, contract: ModelContractBase) -> list[str]:
        """Extract intent types from published_events[].event_name.

        Used by Orchestrator nodes to declare which events they publish.

        Args:
            contract: The contract model to extract from.

        Returns:
            List of event names from published events, or empty list if not present.
        """
        if not hasattr(contract, "published_events"):
            return []

        published_events = contract.published_events
        if not published_events:
            return []

        intent_types: list[str] = []
        for event in published_events:
            # ModelEventDescriptor has event_name field
            if hasattr(event, "event_name") and event.event_name:
                intent_types.append(event.event_name)

        return intent_types

    def _extract_from_aggregation(self, contract: ModelContractBase) -> list[str]:
        """Extract intent types from aggregation.aggregation_functions.

        Used by Reducer nodes to declare aggregation patterns based on
        the output fields of their aggregation functions.

        Args:
            contract: The contract model to extract from.

        Returns:
            List of aggregation intent types (prefixed with 'aggregate.'),
            or empty list if not present.
        """
        if not hasattr(contract, "aggregation"):
            return []

        aggregation = contract.aggregation
        if aggregation is None or not hasattr(aggregation, "aggregation_functions"):
            return []

        agg_funcs = aggregation.aggregation_functions
        if not agg_funcs:
            return []

        intent_types: list[str] = []
        for func in agg_funcs:
            if hasattr(func, "output_field") and func.output_field:
                intent_types.append(f"aggregate.{func.output_field}")

        return intent_types
