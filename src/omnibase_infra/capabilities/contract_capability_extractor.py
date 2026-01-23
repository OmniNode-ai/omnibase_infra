"""Contract capability extractor for ONEX nodes.

Extracts ModelContractCapabilities from typed contract models.
No side effects, deterministic output.

OMN-1136: ContractCapabilityExtractor - Main extractor implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.models.capabilities import ModelContractCapabilities
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.capabilities.capability_inference_rules import (
    CapabilityInferenceRules,
)

if TYPE_CHECKING:
    from omnibase_core.models.contracts import (
        ModelContractBase,
    )


class ContractCapabilityExtractor:
    """Extracts capabilities from typed contract models.

    Responsibilities:
    - Read capability-related fields from contract
    - Apply inference rules to derive additional tags
    - Union explicit + inferred capabilities
    - Return deterministic, sorted output

    This extractor is stateless and produces deterministic output
    for the same contract input.

    Args:
        rules: Optional custom CapabilityInferenceRules instance. If not provided,
            uses a default instance with standard rule mappings.

    Example:
        # Use default rules
        extractor = ContractCapabilityExtractor()

        # Use custom rules
        custom_rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"}
        )
        extractor = ContractCapabilityExtractor(rules=custom_rules)
    """

    def __init__(self, rules: CapabilityInferenceRules | None = None) -> None:
        """Initialize with optional custom inference rules.

        Args:
            rules: Custom CapabilityInferenceRules instance. If None, creates
                a default instance with standard rule mappings.
        """
        self._rules = rules if rules is not None else CapabilityInferenceRules()

    def extract(self, contract: ModelContractBase) -> ModelContractCapabilities | None:
        """Extract capabilities from a contract model.

        Args:
            contract: Typed contract model (Effect, Compute, Reducer, or Orchestrator)

        Returns:
            ModelContractCapabilities with extracted data, or None if contract is None

        Raises:
            Any exceptions from extraction propagate (fail-fast behavior).
        """
        if contract is None:
            return None

        # Extract contract type from node_type
        contract_type = self._extract_contract_type(contract)

        # Extract version
        contract_version = self._extract_version(contract)

        # Extract intent types (varies by node type)
        intent_types = self._extract_intent_types(contract)

        # Extract protocols from dependencies
        protocols = self._extract_protocols(contract)

        # Extract explicit capability tags from contract
        explicit_tags = self._extract_explicit_tags(contract)

        # Infer additional tags using rules
        inferred_tags = self._rules.infer_all(
            intent_types=intent_types,
            protocols=protocols,
            node_type=contract_type,
        )

        # Union explicit + inferred (deterministic)
        all_tags = sorted(set(explicit_tags) | set(inferred_tags))

        return ModelContractCapabilities(
            contract_type=contract_type,
            contract_version=contract_version,
            intent_types=sorted(set(intent_types)),
            protocols=sorted(set(protocols)),
            capability_tags=all_tags,
        )

    def _extract_contract_type(self, contract: ModelContractBase) -> str:
        """Extract normalized contract type string.

        Raises:
            ValueError: If contract does not have node_type field.
        """
        node_type = getattr(contract, "node_type", None)
        if node_type is None:
            raise ValueError("Contract must have node_type field")

        # Handle both enum and string, normalize to lowercase without _GENERIC suffix
        type_str = node_type.value if hasattr(node_type, "value") else str(node_type)
        return type_str.lower().replace("_generic", "")

    def _extract_version(self, contract: ModelContractBase) -> ModelSemVer:
        """Extract contract version.

        Raises:
            ValueError: If contract_version is missing or not a ModelSemVer instance.
        """
        version = getattr(contract, "contract_version", None)
        if isinstance(version, ModelSemVer):
            return version
        raise ValueError(
            f"Contract must have contract_version as ModelSemVer, "
            f"got {type(version).__name__ if version is not None else 'None'}"
        )

    def _extract_intent_types(self, contract: ModelContractBase) -> list[str]:
        """Extract intent types based on node type.

        Different contract types expose intent types in different locations:
        - Effect: event_type.primary_events
        - Orchestrator: consumed_events[].event_pattern
        - Reducer: aggregation, state_machine related intents
        """
        intent_types: list[str] = []

        # For effect nodes: check event_type.primary_events
        if hasattr(contract, "event_type"):
            event_type = contract.event_type
            if event_type is not None and hasattr(event_type, "primary_events"):
                primary_events = event_type.primary_events
                if primary_events:
                    intent_types.extend(primary_events)

        # For orchestrators: check consumed_events
        if hasattr(contract, "consumed_events"):
            consumed_events = contract.consumed_events
            if consumed_events:
                for event in consumed_events:
                    # ModelEventSubscription has event_pattern field
                    if hasattr(event, "event_pattern") and event.event_pattern:
                        intent_types.append(event.event_pattern)

        # For orchestrators: check published_events
        if hasattr(contract, "published_events"):
            published_events = contract.published_events
            if published_events:
                for event in published_events:
                    # ModelEventDescriptor has event_name field
                    if hasattr(event, "event_name") and event.event_name:
                        intent_types.append(event.event_name)

        # For reducers: check aggregation and state_machine for patterns
        if hasattr(contract, "aggregation"):
            aggregation = contract.aggregation
            if aggregation is not None:
                # Check for aggregation patterns
                if hasattr(aggregation, "aggregation_functions"):
                    agg_funcs = aggregation.aggregation_functions
                    if agg_funcs:
                        for func in agg_funcs:
                            if hasattr(func, "output_field") and func.output_field:
                                intent_types.append(f"aggregate.{func.output_field}")

        return intent_types

    def _extract_protocols(self, contract: ModelContractBase) -> list[str]:
        """Extract protocol names from dependencies and interfaces."""
        protocols: list[str] = []

        # From protocol_interfaces field
        if hasattr(contract, "protocol_interfaces"):
            protocol_interfaces = contract.protocol_interfaces
            if protocol_interfaces:
                for proto in protocol_interfaces:
                    if proto is not None:  # Skip None values
                        protocols.append(proto)

        # From dependencies where type is protocol
        if hasattr(contract, "dependencies"):
            dependencies = contract.dependencies
            if dependencies:
                for dep in dependencies:
                    # Check if it's a protocol dependency using is_protocol() method
                    if hasattr(dep, "is_protocol") and dep.is_protocol():
                        if hasattr(dep, "name") and dep.name:
                            protocols.append(dep.name)
                    # Fallback: check dependency_type directly
                    elif hasattr(dep, "dependency_type"):
                        dep_type = dep.dependency_type
                        type_str = (
                            dep_type.value
                            if hasattr(dep_type, "value")
                            else str(dep_type)
                        )
                        if type_str.upper() == "PROTOCOL":
                            if hasattr(dep, "name") and dep.name:
                                protocols.append(dep.name)

        return protocols

    def _extract_explicit_tags(self, contract: ModelContractBase) -> list[str]:
        """Extract explicitly declared capability tags from contract."""
        tags: list[str] = []

        # From top-level tags field (all contracts have this)
        if hasattr(contract, "tags"):
            contract_tags = contract.tags
            if contract_tags:
                for tag in contract_tags:
                    if tag is not None:  # Skip None values
                        tags.append(tag)

        return tags
