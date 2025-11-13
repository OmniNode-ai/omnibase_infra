#!/usr/bin/env python3
"""
Contract Introspection Utility for ONEX Code Generation.

Provides introspection and validation of omnibase_core contract models to ensure
generated code includes all required fields for successful validation.

This utility prevents common code generation bugs by:
- Identifying required vs optional fields in contract models
- Extracting field types and default values
- Providing helpers to generate valid contract instances
- Validating template context has all necessary data

ONEX v2.0 Compliance:
- Works with all contract types (Effect, Compute, Reducer, Orchestrator)
- Ensures generated contracts pass omnibase_core validation
- Prevents missing required field errors before file generation
"""

import logging
from typing import Any, ClassVar

from omnibase_core.models.contracts.model_contract_compute import ModelContractCompute
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer
from pydantic import BaseModel

from .node_classifier import EnumNodeType

logger = logging.getLogger(__name__)


class ContractIntrospector:
    """
    Introspect contract models to extract field requirements.

    Provides utilities for analyzing omnibase_core contract models to ensure
    generated code includes all required fields and passes validation.

    Thread-safe and stateless - can be instantiated once and reused.

    Example:
        >>> introspector = ContractIntrospector()
        >>> required = introspector.get_required_fields(ModelContractEffect)
        >>> print(required)
        ['name', 'version', 'description', 'node_type', 'input_model',
         'output_model', 'io_operations']
    """

    # Map of node type to contract class
    CONTRACT_CLASS_MAP: ClassVar[dict[EnumNodeType, type[BaseModel]]] = {
        EnumNodeType.EFFECT: ModelContractEffect,
        EnumNodeType.COMPUTE: ModelContractCompute,
        EnumNodeType.REDUCER: ModelContractReducer,
        EnumNodeType.ORCHESTRATOR: ModelContractOrchestrator,
    }

    def get_required_fields(self, contract_class: type[BaseModel]) -> list[str]:
        """
        Get list of required field names for a contract class.

        Analyzes Pydantic model_fields to determine which fields are required
        (i.e., have no default value and are not Optional).

        Args:
            contract_class: Pydantic contract model class (e.g., ModelContractEffect)

        Returns:
            List of required field names

        Example:
            >>> introspector = ContractIntrospector()
            >>> required = introspector.get_required_fields(ModelContractEffect)
            >>> assert 'io_operations' in required  # Effect-specific required field
            >>> assert 'name' in required  # Base contract required field
        """
        required_fields = []

        for field_name, field_info in contract_class.model_fields.items():
            if field_info.is_required():
                required_fields.append(field_name)

        logger.debug(
            f"Identified {len(required_fields)} required fields for {contract_class.__name__}: {required_fields}"
        )
        return required_fields

    def get_field_defaults(self, contract_class: type[BaseModel]) -> dict[str, Any]:
        """
        Get default values for optional fields in a contract class.

        Extracts default values from Pydantic model_fields for fields that
        have defaults specified.

        Args:
            contract_class: Pydantic contract model class

        Returns:
            Dict mapping field names to their default values

        Example:
            >>> introspector = ContractIntrospector()
            >>> defaults = introspector.get_field_defaults(ModelContractEffect)
            >>> assert defaults.get('idempotent_operations') == True
            >>> assert defaults.get('audit_trail_enabled') == True
        """
        defaults = {}

        for field_name, field_info in contract_class.model_fields.items():
            if not field_info.is_required() and field_info.default is not None:
                # Store the default value
                defaults[field_name] = field_info.default

        logger.debug(
            f"Extracted {len(defaults)} default values for {contract_class.__name__}"
        )
        return defaults

    def validate_contract_data(
        self, contract_class: type[BaseModel], data: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate that data dict contains all required fields for contract.

        Checks if provided data has all required fields needed to construct
        a valid contract instance. Does NOT validate field types or values.

        Args:
            contract_class: Pydantic contract model class
            data: Dict of field name -> value pairs

        Returns:
            Tuple of (is_valid: bool, missing_fields: list[str])
            - is_valid: True if all required fields present
            - missing_fields: List of missing required field names (empty if valid)

        Example:
            >>> introspector = ContractIntrospector()
            >>> data = {'name': 'test', 'version': {'major': 1, 'minor': 0, 'patch': 0}}
            >>> is_valid, missing = introspector.validate_contract_data(
            ...     ModelContractEffect, data
            ... )
            >>> assert not is_valid
            >>> assert 'io_operations' in missing  # Missing required field
        """
        required_fields = self.get_required_fields(contract_class)
        missing_fields = [field for field in required_fields if field not in data]

        is_valid = len(missing_fields) == 0

        if not is_valid:
            logger.warning(
                f"Contract data validation failed for {contract_class.__name__}: "
                f"missing {len(missing_fields)} required fields: {missing_fields}"
            )
        else:
            logger.debug(
                f"Contract data validation passed for {contract_class.__name__}"
            )

        return is_valid, missing_fields

    def get_field_type(
        self, contract_class: type[BaseModel], field_name: str
    ) -> str | None:
        """
        Get type annotation for a specific field.

        Args:
            contract_class: Pydantic contract model class
            field_name: Name of field to get type for

        Returns:
            String representation of field type, or None if field doesn't exist

        Example:
            >>> introspector = ContractIntrospector()
            >>> field_type = introspector.get_field_type(ModelContractEffect, 'name')
            >>> assert 'str' in field_type
        """
        field_info = contract_class.model_fields.get(field_name)
        if field_info is None:
            return None

        return str(field_info.annotation)

    def generate_minimal_contract(self, node_type: EnumNodeType) -> dict[str, Any]:
        """
        Generate minimal valid contract data for a node type.

        Creates a dict with all required fields populated with sensible defaults.
        Useful for test scaffolding and template context validation.

        Args:
            node_type: Node type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)

        Returns:
            Dict with all required fields populated

        Raises:
            ValueError: If node_type is invalid or contract class not found

        Example:
            >>> introspector = ContractIntrospector()
            >>> minimal = introspector.generate_minimal_contract(EnumNodeType.EFFECT)
            >>> assert 'name' in minimal
            >>> assert 'io_operations' in minimal
            >>> assert isinstance(minimal['io_operations'], list)
        """
        contract_class = self.CONTRACT_CLASS_MAP.get(node_type)
        if contract_class is None:
            raise ValueError(f"Unknown node type: {node_type}")

        required_fields = self.get_required_fields(contract_class)

        # Build minimal contract data
        minimal_contract = {
            # Base contract fields
            "name": "test_node",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "description": "Test node description",
            "node_type": node_type.value,
            "input_model": "ModelTestRequest",
            "output_model": "ModelTestResponse",
        }

        # Add node-type-specific required fields
        if node_type == EnumNodeType.EFFECT:
            # Effect nodes require io_operations
            minimal_contract["io_operations"] = [
                {
                    "operation_type": "database_query",
                    "atomic": True,
                    "timeout_seconds": 30,
                    "validation_enabled": True,
                }
            ]

        elif node_type == EnumNodeType.COMPUTE:
            # Compute nodes require algorithm config
            minimal_contract["algorithm"] = {
                "algorithm_type": "weighted_factor_algorithm",
                "factors": [{"name": "test_factor", "weight": 1.0}],
            }

        # Reducer and Orchestrator don't have additional required fields
        # beyond base contract fields

        logger.debug(
            f"Generated minimal contract for {node_type.value} with {len(minimal_contract)} fields"
        )
        return minimal_contract

    def get_required_fields_for_node_type(self, node_type: EnumNodeType) -> list[str]:
        """
        Get required fields for a specific node type.

        Convenience method that combines node type lookup with field extraction.

        Args:
            node_type: Node type enum value

        Returns:
            List of required field names

        Raises:
            ValueError: If node_type is invalid

        Example:
            >>> introspector = ContractIntrospector()
            >>> required = introspector.get_required_fields_for_node_type(EnumNodeType.EFFECT)
            >>> assert 'io_operations' in required
        """
        contract_class = self.CONTRACT_CLASS_MAP.get(node_type)
        if contract_class is None:
            raise ValueError(f"Unknown node type: {node_type}")

        return self.get_required_fields(contract_class)

    def validate_template_context_for_node_type(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate that template context has all data needed for contract generation.

        Checks if template context dict contains enough data to generate a valid
        contract YAML for the specified node type.

        Args:
            node_type: Node type being generated
            context: Template context dict (from TemplateEngine._build_template_context)

        Returns:
            Tuple of (is_valid: bool, missing_data: list[str])
            - is_valid: True if context has all necessary data
            - missing_data: List of missing required data keys

        Example:
            >>> introspector = ContractIntrospector()
            >>> context = {'service_name': 'test', 'node_class_name': 'NodeTest'}
            >>> is_valid, missing = introspector.validate_template_context_for_node_type(
            ...     EnumNodeType.EFFECT, context
            ... )
            >>> # Will report missing io_operations configuration
        """
        # Check base contract fields are present in context
        required_context_keys = [
            "service_name",
            "node_class_name",
            "node_type",
            "node_type_upper",
            "business_description",
            "version_dict",
            "input_model",
            "output_model",
        ]

        missing_keys = [key for key in required_context_keys if key not in context]

        # For effect nodes, verify we have io_operations configuration
        if node_type == EnumNodeType.EFFECT:
            # The template should have operations data that will be used to generate io_operations
            if "operations" not in context or not context["operations"]:
                missing_keys.append("operations (needed for io_operations)")

        # For compute nodes, verify we have algorithm configuration
        elif node_type == EnumNodeType.COMPUTE:
            # The template should have data to generate algorithm config
            if "operations" not in context or not context["operations"]:
                missing_keys.append("operations (needed for algorithm config)")

        is_valid = len(missing_keys) == 0

        if not is_valid:
            logger.warning(
                f"Template context validation failed for {node_type.value}: "
                f"missing {len(missing_keys)} required keys: {missing_keys}"
            )
        else:
            logger.debug(f"Template context validation passed for {node_type.value}")

        return is_valid, missing_keys


# Export
__all__ = ["ContractIntrospector"]
