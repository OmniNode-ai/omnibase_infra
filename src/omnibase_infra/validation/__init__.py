"""
ONEX Infrastructure Validation Module.

Re-exports validators from omnibase_core with infrastructure-specific defaults.

Security Design (Intentional Fail-Open Architecture):
    ONEX validation modules use a FAIL-OPEN design by default. This is an
    INTENTIONAL architectural decision, NOT a security vulnerability.
    Understanding the rationale is critical for proper security architecture.

    **What Fail-Open Means in ONEX Validation**:

    Execution Shape Validators (AST and Runtime):
        - Syntax errors: Logged and returned as violations, but validation continues
        - Unknown handler types: Allowed by default without blocking
        - Undetectable message categories: Skipped, output is permitted
        - I/O errors (permissions, missing files): Logged, file skipped, continues
        - Missing rules: No exception, output permitted

    Routing Coverage Validators:
        - Missing routes: Reported as gaps, but startup is not blocked by default
        - Discovery errors: Logged, partial results returned

    **Why Fail-Open is Correct for These Validators**:

    1. **Architectural Validators, NOT Security Boundaries**:
       These validators enforce ONEX design patterns (4-node architecture,
       execution shapes, routing coverage) to catch developer mistakes at
       build/test time. They are NOT designed to prevent malicious inputs
       or unauthorized access.

    2. **CI Pipeline Resilience**:
       CI should not break on transient validator errors (I/O issues, syntax
       errors in non-critical files, evolving codebases). Fail-open allows
       pipelines to collect all violations and report comprehensively rather
       than failing on the first error.

    3. **Defense-in-Depth Model**:
       Security boundaries should be implemented at the infrastructure layer:
       - Authentication: API gateway, OAuth, JWT, mTLS
       - Authorization: Service layer RBAC/ABAC
       - Input Validation: Schema validation at entry points
       - Network Security: Firewall rules, service mesh

       These validators operate AFTER security layers, on trusted internal
       code, making fail-open safe and appropriate.

    4. **Extensibility and Forward Compatibility**:
       New handler types, message categories, or validation rules should
       work by default without requiring immediate updates to all validators.
       Fail-closed would break valid code during evolution.

    **When Strict (Fail-Closed) Validation is Needed**:

    If your use case requires fail-closed behavior (e.g., security-critical
    enforcement, production gate validation), implement one of these approaches:

    1. **Check for blocking violations explicitly**::

           passed, violations = validate_execution_shapes_ci(directory)
           if not passed:
               sys.exit(1)  # Fail the pipeline

    2. **Use strict wrapper around validators**::

           violations = validate_execution_shapes(directory)
           if any(v.severity == "error" for v in violations):
               raise ValidationError("Blocking violations found")

    3. **Implement fail-closed policy in CI configuration**::

           # In CI script
           violations=$(validate_execution_shapes src/handlers)
           if [ -n "$violations" ]; then
               echo "$violations"
               exit 1
           fi

    **Security Responsibility Boundaries**:

    | Layer | Responsibility |
    |-------|----------------|
    | This validator | Architectural pattern enforcement (developer guardrails) |
    | Infrastructure layer | Authentication, authorization, input validation |
    | Application layer | Business logic validation, access control |
    | Network layer | TLS, firewall rules, service mesh policies |

    See individual validator modules for detailed fail-open documentation:
    - execution_shape_validator.py: AST-based static analysis (lines 280-370)
    - runtime_shape_validator.py: Runtime validation (lines 52-121)
    - routing_coverage_validator.py: Routing gap detection
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
# NOTE: EXECUTION_SHAPE_RULES is defined ONLY in execution_shape_validator.py
# (the canonical single source of truth). This import re-exports it for public API
# convenience. See execution_shape_validator.py lines 112-149 for the definition.
from omnibase_infra.validation.execution_shape_validator import (
    EXECUTION_SHAPE_RULES,
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
# NOTE: RuntimeShapeValidator uses EXECUTION_SHAPE_RULES from execution_shape_validator.py
# (not a separate definition). See runtime_shape_validator.py lines 66-69 for the import.
from omnibase_infra.validation.runtime_shape_validator import (
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
]
