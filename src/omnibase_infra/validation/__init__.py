"""
ONEX Infrastructure Validation Module.

Re-exports validators from omnibase_core with infrastructure-specific defaults.
"""

from omnibase_core.validation import (
    validate_all,
    validate_architecture,
    validate_contracts,
    validate_patterns,
    validate_union_usage,
)
from omnibase_core.validation.circular_import_validator import CircularImportValidator
from omnibase_core.validation.contract_validator import ProtocolContractValidator

# Infrastructure-specific wrappers will be imported from infra_validators
from omnibase_infra.validation.infra_validators import (
    get_validation_summary,
    validate_infra_all,
    validate_infra_architecture,
    validate_infra_circular_imports,
    validate_infra_contract_deep,
    validate_infra_contracts,
    validate_infra_patterns,
    validate_infra_union_usage,
)

__all__ = [
    # Direct re-exports from omnibase_core
    "validate_architecture",
    "validate_contracts",
    "validate_patterns",
    "validate_union_usage",
    "validate_all",
    "ProtocolContractValidator",
    "CircularImportValidator",
    # Infrastructure-specific wrappers
    "validate_infra_architecture",
    "validate_infra_contracts",
    "validate_infra_patterns",
    "validate_infra_contract_deep",
    "validate_infra_union_usage",
    "validate_infra_circular_imports",
    "validate_infra_all",
    "get_validation_summary",
]
