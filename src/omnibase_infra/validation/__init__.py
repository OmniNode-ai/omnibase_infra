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

# AST-based execution shape validation for CI gate (OMN-958)
from omnibase_infra.validation.execution_shape_validator import (
    EXECUTION_SHAPE_RULES as AST_EXECUTION_SHAPE_RULES,
)
from omnibase_infra.validation.execution_shape_validator import (
    ExecutionShapeValidator,
    HandlerInfo,
    get_execution_shape_rules,
    validate_execution_shapes,
    validate_execution_shapes_ci,
)

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

# Routing coverage validation for startup fail-fast (OMN-958)
from omnibase_infra.validation.routing_coverage_validator import (
    RoutingCoverageError,
    RoutingCoverageValidator,
    check_routing_coverage_ci,
    discover_message_types,
    discover_registered_routes,
    validate_routing_coverage_on_startup,
)

# Runtime shape validation for ONEX 4-node architecture
from omnibase_infra.validation.runtime_shape_validator import (
    EXECUTION_SHAPE_RULES,
    ExecutionShapeViolationError,
    RuntimeShapeValidator,
    detect_message_category,
    enforce_execution_shape,
)

# Topic category validation for execution shape enforcement
from omnibase_infra.validation.topic_category_validator import (
    HANDLER_EXPECTED_CATEGORIES,
    TOPIC_CATEGORY_PATTERNS,
    TOPIC_SUFFIXES,
    TopicCategoryASTVisitor,
    TopicCategoryValidator,
    validate_message_on_topic,
    validate_topic_categories_in_directory,
    validate_topic_categories_in_file,
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
    # Runtime shape validation
    "EXECUTION_SHAPE_RULES",
    "ExecutionShapeViolationError",
    "RuntimeShapeValidator",
    "detect_message_category",
    "enforce_execution_shape",
    # Topic category validation
    "TopicCategoryValidator",
    "TopicCategoryASTVisitor",
    "TOPIC_CATEGORY_PATTERNS",
    "TOPIC_SUFFIXES",
    "HANDLER_EXPECTED_CATEGORIES",
    "validate_topic_categories_in_file",
    "validate_topic_categories_in_directory",
    "validate_message_on_topic",
    # Routing coverage validation (OMN-958)
    "RoutingCoverageError",
    "RoutingCoverageValidator",
    "discover_message_types",
    "discover_registered_routes",
    "validate_routing_coverage_on_startup",
    "check_routing_coverage_ci",
    # AST-based execution shape validation (OMN-958)
    "ExecutionShapeValidator",
    "HandlerInfo",
    "validate_execution_shapes",
    "validate_execution_shapes_ci",
    "get_execution_shape_rules",
    "AST_EXECUTION_SHAPE_RULES",
]
