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

           result = validate_execution_shapes_ci(directory)
           if not result.passed:
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
    - validator_execution_shape.py: AST-based static analysis (Limitations section in module docstring)
    - validator_runtime_shape.py: Runtime validation (Security Design section in module docstring)
    - validator_routing_coverage.py: Routing gap detection (module docstring)
"""

# Topic suffix validation models (OMN-1537)
from omnibase_core.models.validation import (
    ModelTopicSuffixParts,
    ModelTopicValidationResult,
)
from omnibase_core.validation import (
    CircularImportValidator,
    ModelModuleImportResult,
    compose_full_topic,
    is_valid_topic_suffix,
    parse_topic_suffix,
    validate_all,
    validate_architecture,
    validate_contracts,
    validate_patterns,
    validate_topic_suffix,
    validate_union_usage,
)

# Contract linting for CI gate (PR #57)
from omnibase_infra.validation.enums.enum_contract_violation_severity import (
    EnumContractViolationSeverity,
)

# Infrastructure-specific wrappers will be imported from infra_validators
from omnibase_infra.validation.infra_validators import (
    INFRA_MAX_UNIONS,
    get_validation_summary,
    is_isinstance_union,
    validate_infra_all,
    validate_infra_architecture,
    validate_infra_circular_imports,
    validate_infra_contracts,
    validate_infra_patterns,
    validate_infra_union_usage,
)
from omnibase_infra.validation.linter_contract import (
    ContractLinter,
    lint_contract_file,
    lint_contracts_ci,
    lint_contracts_in_directory,
)
from omnibase_infra.validation.models.model_contract_lint_result import (
    ModelContractLintResult,
)
from omnibase_infra.validation.models.model_contract_violation import (
    ModelContractViolation,
)

# Validation error aggregation and reporting for startup (OMN-1091)
from omnibase_infra.validation.service_validation_aggregator import ValidationAggregator

# AST-based Any type validation for strong typing policy (OMN-1276)
from omnibase_infra.validation.validator_any_type import (
    AnyTypeDetector,
    ModelAnyTypeValidationResult,
    validate_any_types,
    validate_any_types_ci,
    validate_any_types_in_file,
)

# Chain propagation validation for correlation and causation chains (OMN-951)
from omnibase_infra.validation.validator_chain_propagation import (
    ChainPropagationError,
    ChainPropagationValidator,
    enforce_chain_propagation,
    validate_message_chain,
)

# AST-based execution shape validation for CI gate (OMN-958)
# NOTE: EXECUTION_SHAPE_RULES is defined ONLY in validator_execution_shape.py
# (the canonical single source of truth). This import re-exports it for public API
# convenience. See validator_execution_shape.py lines 111-148 for the definition.
from omnibase_infra.validation.validator_execution_shape import (
    EXECUTION_SHAPE_RULES,
    ExecutionShapeValidator,
    ModelDetectedNodeInfo,
    ModelExecutionShapeValidationResult,
    get_execution_shape_rules,
    validate_execution_shapes,
    validate_execution_shapes_ci,
)

# LocalHandler import validation for production code policy (OMN-743)
from omnibase_infra.validation.validator_localhandler import (
    ModelLocalHandlerValidationResult,
    ModelLocalHandlerViolation,
    validate_localhandler,
    validate_localhandler_ci,
    validate_localhandler_in_file,
)

# Registration-time security validation for handlers (OMN-1098)
from omnibase_infra.validation.validator_registration_security import (
    RegistrationSecurityValidator,
    validate_handler_registration,
)

# Routing coverage validation for startup fail-fast (OMN-958)
from omnibase_infra.validation.validator_routing_coverage import (
    RoutingCoverageError,
    RoutingCoverageValidator,
    check_routing_coverage_ci,
    discover_message_types,
    discover_registered_routes,
    validate_routing_coverage_on_startup,
)

# Runtime shape validation for ONEX 4-node architecture
# NOTE: RuntimeShapeValidator uses EXECUTION_SHAPE_RULES from validator_execution_shape.py
# (not a separate definition). See the import section of validator_runtime_shape.py.
from omnibase_infra.validation.validator_runtime_shape import (
    ExecutionShapeViolationError,
    RuntimeShapeValidator,
    detect_message_category,
    enforce_execution_shape,
)

# Security validation (OMN-1277) - contract-driven validator
from omnibase_infra.validation.validator_security import ValidatorSecurity

# Topic category validation for execution shape enforcement
from omnibase_infra.validation.validator_topic_category import (
    NODE_ARCHETYPE_EXPECTED_CATEGORIES,
    TOPIC_CATEGORY_PATTERNS,
    TOPIC_SUFFIXES,
    TopicCategoryASTVisitor,
    TopicCategoryValidator,
    validate_message_on_topic,
    validate_topic_categories_in_directory,
    validate_topic_categories_in_file,
)

__all__: list[str] = [
    # Constants
    "EXECUTION_SHAPE_RULES",  # Runtime shape validation rules
    "INFRA_MAX_UNIONS",  # Infrastructure max union threshold
    "NODE_ARCHETYPE_EXPECTED_CATEGORIES",  # Node archetype categories
    "TOPIC_CATEGORY_PATTERNS",  # Topic category patterns
    "TOPIC_SUFFIXES",  # Topic suffix constants
    # Topic suffix validation (OMN-1537)
    "ModelTopicSuffixParts",  # Topic suffix parsed parts
    "ModelTopicValidationResult",  # Topic validation result
    "compose_full_topic",  # Compose full topic from parts
    "is_valid_topic_suffix",  # Check if topic suffix is valid
    "parse_topic_suffix",  # Parse topic suffix into parts
    "validate_topic_suffix",  # Validate topic suffix format
    # Errors
    "ChainPropagationError",  # Chain propagation error (OMN-951)
    "ExecutionShapeViolationError",  # Execution shape violation
    "RoutingCoverageError",  # Routing coverage error (OMN-958)
    # Enums
    "EnumContractViolationSeverity",  # Contract violation severity
    # Models
    "ModelAnyTypeValidationResult",  # Any type validation result (OMN-1276)
    "ModelContractLintResult",  # Contract lint result
    "ModelContractViolation",  # Contract violation model
    "ModelDetectedNodeInfo",  # Detected node info
    "ModelExecutionShapeValidationResult",  # Execution shape result
    "ModelLocalHandlerValidationResult",  # LocalHandler validation result (OMN-743)
    "ModelLocalHandlerViolation",  # LocalHandler violation model (OMN-743)
    "ModelModuleImportResult",  # Module import result (from omnibase_core)
    # Validators
    "AnyTypeDetector",  # Any type AST detector (OMN-1276)
    "ChainPropagationValidator",  # Chain propagation validator (OMN-951)
    "CircularImportValidator",  # Circular import validator
    "ContractLinter",  # Contract linter (PR #57)
    "ExecutionShapeValidator",  # AST-based shape validation (OMN-958)
    "RegistrationSecurityValidator",  # Registration-time security (OMN-1098)
    "RoutingCoverageValidator",  # Routing coverage validator (OMN-958)
    "RuntimeShapeValidator",  # Runtime shape validator
    "TopicCategoryASTVisitor",  # Topic category AST visitor
    "TopicCategoryValidator",  # Topic category validator
    "ValidationAggregator",  # Validation error aggregation (OMN-1091)
    "ValidatorSecurity",  # Contract-driven security validator (OMN-1277)
    # Functions
    "check_routing_coverage_ci",  # CI routing coverage check
    "detect_message_category",  # Message category detection
    "discover_message_types",  # Message type discovery
    "discover_registered_routes",  # Route discovery
    "enforce_chain_propagation",  # Chain propagation enforcement
    "enforce_execution_shape",  # Execution shape enforcement
    "get_execution_shape_rules",  # Get shape rules
    "get_validation_summary",  # Get validation summary
    "is_isinstance_union",  # Check if union is in isinstance() call
    "lint_contract_file",  # Lint single contract file
    "lint_contracts_ci",  # CI contract linting
    "lint_contracts_in_directory",  # Directory contract linting
    "validate_all",  # Re-export from omnibase_core
    "validate_any_types",  # Any type validation (OMN-1276)
    "validate_any_types_ci",  # Any type CI validation (OMN-1276)
    "validate_any_types_in_file",  # Any type file validation (OMN-1276)
    "validate_architecture",  # Re-export from omnibase_core
    "validate_contracts",  # Re-export from omnibase_core
    "validate_execution_shapes",  # Execution shape validation
    "validate_execution_shapes_ci",  # CI shape validation
    "validate_handler_registration",  # Handler registration validation (OMN-1098)
    "validate_infra_all",  # Infrastructure validation
    "validate_infra_architecture",  # Infrastructure architecture
    "validate_infra_circular_imports",  # Circular import check
    "validate_infra_contracts",  # Infrastructure contracts
    "validate_infra_patterns",  # Infrastructure patterns
    "validate_infra_union_usage",  # Union usage validation
    "validate_localhandler",  # LocalHandler validation (OMN-743)
    "validate_localhandler_ci",  # LocalHandler CI validation (OMN-743)
    "validate_localhandler_in_file",  # LocalHandler file validation (OMN-743)
    "validate_message_chain",  # Message chain validation
    "validate_message_on_topic",  # Topic message validation
    "validate_patterns",  # Re-export from omnibase_core
    "validate_routing_coverage_on_startup",  # Startup routing check
    "validate_topic_categories_in_directory",  # Directory topic validation
    "validate_topic_categories_in_file",  # File topic validation
    "validate_union_usage",  # Re-export from omnibase_core
]
