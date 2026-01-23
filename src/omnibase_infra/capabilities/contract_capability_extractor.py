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
from omnibase_infra.capabilities.intent_type_extractor import IntentTypeExtractor

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
        self._intent_extractor = IntentTypeExtractor()

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

        # Union explicit + inferred (deterministic), filtering out invalid tags
        combined_tags = set(explicit_tags) | set(inferred_tags)
        all_tags = sorted(tag for tag in combined_tags if self._validate_tag(tag))

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

        Delegates to IntentTypeExtractor which handles extraction from
        multiple sources based on node type.

        Args:
            contract: The contract model to extract intent types from.

        Returns:
            Combined list of intent types from all applicable sources.
        """
        return self._intent_extractor.extract_all(contract)

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

    def _validate_tag(self, tag: str | None) -> bool:
        """Validate a tag string.

        Args:
            tag: Tag string to validate.

        Returns:
            True if the tag is valid, False otherwise.
            Invalid tags include: None, empty strings, whitespace-only strings.
        """
        if tag is None:
            return False
        if not isinstance(tag, str):
            return False
        if tag == "":
            return False
        if tag.strip() == "":
            return False
        return True

    def _extract_explicit_tags(self, contract: ModelContractBase) -> list[str]:
        """Extract explicitly declared capability tags from contract."""
        tags: list[str] = []

        # From top-level tags field (all contracts have this)
        if hasattr(contract, "tags"):
            contract_tags = contract.tags
            if contract_tags:
                for tag in contract_tags:
                    if self._validate_tag(tag):
                        tags.append(tag)

        return tags
