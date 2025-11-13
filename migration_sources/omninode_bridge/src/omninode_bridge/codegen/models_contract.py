#!/usr/bin/env python3
"""
Contract Data Models for ONEX v2.0 Code Generation (Phase 3 Enhanced).

Provides Pydantic v2 data models for:
- Mixin declarations
- Advanced features configuration
- Enhanced contract representation
- Phase 3: Template configuration, generation directives, quality gates

These models support:
- v1.0 contracts (backward compatible)
- v2.0 contracts (mixins + advanced_features)
- v2.1 contracts (Phase 3: template hints + generation directives)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ============================================================================
# Phase 3: Enums for Template and Generation Configuration
# ============================================================================


class EnumTemplateVariant(str, Enum):
    """
    Template variant selection (Phase 3).

    Variants determine the template complexity and features:
    - MINIMAL: Bare bones, learning/prototyping
    - STANDARD: Core features, common use cases
    - PRODUCTION: Full features, production-ready
    - CUSTOM: User-defined template path
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    PRODUCTION = "production"
    CUSTOM = "custom"


class EnumLLMTier(str, Enum):
    """
    LLM tier selection for code generation (Phase 3).

    Tiers balance cost, speed, and quality:
    - LOCAL: Fast, no cost, limited quality (Ollama/vLLM)
    - CLOUD_FAST: Fast, low cost, good quality (e.g., GLM-4.5)
    - CLOUD_ACCURATE: Slower, higher cost, best quality (e.g., GPT-4, Claude 3)
    """

    LOCAL = "LOCAL"
    CLOUD_FAST = "CLOUD_FAST"
    CLOUD_ACCURATE = "CLOUD_ACCURATE"


class EnumQualityLevel(str, Enum):
    """
    Quality level for generation and validation (Phase 3).

    Levels determine validation strictness:
    - MINIMAL: Syntax only
    - STANDARD: Syntax + ONEX compliance + imports
    - PRODUCTION: All checks + security + patterns
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    PRODUCTION = "production"


class EnumFallbackStrategy(str, Enum):
    """
    Fallback strategy when generation fails (Phase 3).

    Strategies:
    - FAIL: Abort generation on failure
    - GRACEFUL: Fallback to template with TODO comments
    - MINIMAL: Fallback to minimal template without patterns
    """

    FAIL = "fail"
    GRACEFUL = "graceful"
    MINIMAL = "minimal"


@dataclass
class ModelMixinDeclaration:
    """
    Represents a single mixin declaration in a contract.

    Corresponds to items in the `mixins` array in v2.0 contracts.

    Attributes:
        name: Mixin class name (must follow Mixin* pattern)
        enabled: Whether mixin is enabled (allows conditional inclusion)
        config: Mixin-specific configuration dict
        import_path: Resolved import path (e.g., omnibase_core.mixins.mixin_health_check)
        validation_errors: List of validation error messages
    """

    name: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    import_path: str = ""
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if mixin declaration has no validation errors."""
        return len(self.validation_errors) == 0

    def add_validation_error(self, error: str) -> None:
        """Add validation error message."""
        self.validation_errors.append(error)


@dataclass
class ModelCircuitBreakerConfig:
    """Circuit breaker configuration for advanced_features."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_ms: int = 60000
    half_open_max_calls: int = 3
    services: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelRetryPolicyConfig:
    """Retry policy configuration for advanced_features."""

    enabled: bool = True
    max_attempts: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_multiplier: float = 2.0
    retryable_exceptions: list[str] = field(
        default_factory=lambda: ["TimeoutError", "ConnectionError", "TemporaryFailure"]
    )
    retryable_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )


@dataclass
class ModelDeadLetterQueueConfig:
    """Dead letter queue configuration for advanced_features."""

    enabled: bool = True
    max_retries: int = 3
    topic_suffix: str = ".dlq"
    retry_delay_ms: int = 5000
    alert_threshold: int = 100
    monitoring: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTransactionsConfig:
    """Transaction configuration for advanced_features."""

    enabled: bool = True
    isolation_level: str = "READ_COMMITTED"
    timeout_seconds: int = 30
    rollback_on_error: bool = True
    savepoints: bool = True


@dataclass
class ModelSecurityValidationConfig:
    """Security validation configuration for advanced_features."""

    enabled: bool = True
    sanitize_inputs: bool = True
    sanitize_logs: bool = True
    validate_sql: bool = True
    max_input_length: int = 10000
    forbidden_patterns: list[str] = field(
        default_factory=lambda: [
            r"(?i)(DROP|DELETE|TRUNCATE)\s+TABLE",
            r"(?i)EXEC(UTE)?\s+",
        ]
    )
    redact_fields: list[str] = field(
        default_factory=lambda: ["password", "api_key", "secret", "token"]
    )


@dataclass
class ModelObservabilityConfig:
    """Observability configuration for advanced_features."""

    tracing: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAdvancedFeatures:
    """
    Advanced features configuration for v2.0 contracts.

    Corresponds to the `advanced_features` section in v2.0 contracts.
    These features are built into NodeEffect but configurable via contract.

    Attributes:
        circuit_breaker: Circuit breaker configuration
        retry_policy: Retry policy configuration
        dead_letter_queue: DLQ configuration
        transactions: Transaction configuration
        security_validation: Security validation configuration
        observability: Observability configuration
    """

    circuit_breaker: Optional[ModelCircuitBreakerConfig] = None
    retry_policy: Optional[ModelRetryPolicyConfig] = None
    dead_letter_queue: Optional[ModelDeadLetterQueueConfig] = None
    transactions: Optional[ModelTransactionsConfig] = None
    security_validation: Optional[ModelSecurityValidationConfig] = None
    observability: Optional[ModelObservabilityConfig] = None


@dataclass
class ModelVersionInfo:
    """Semantic version information."""

    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        """String representation (e.g., '1.0.0')."""
        return f"{self.major}.{self.minor}.{self.patch}"


# ============================================================================
# Phase 3: Template Configuration Models
# ============================================================================


@dataclass
class ModelTemplateConfiguration:
    """
    Template configuration for Phase 3 code generation.

    Specifies template variant, requested patterns, and pattern-specific
    configuration for intelligent template selection and code generation.

    Attributes:
        variant: Template variant to use (minimal/standard/production/custom)
        custom_template: Path to custom template (required if variant=custom)
        patterns: List of pattern names to include (e.g., ['circuit_breaker', 'retry_policy'])
        pattern_configuration: Pattern-specific configuration overrides
    """

    variant: EnumTemplateVariant = EnumTemplateVariant.STANDARD
    custom_template: Optional[str] = None
    patterns: list[str] = field(default_factory=list)
    pattern_configuration: dict[str, Any] = field(default_factory=dict)

    def add_pattern(
        self, pattern_name: str, config: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Add a pattern to the configuration.

        Args:
            pattern_name: Name of the pattern (e.g., 'circuit_breaker')
            config: Optional pattern-specific configuration
        """
        if pattern_name not in self.patterns:
            self.patterns.append(pattern_name)

        if config:
            self.pattern_configuration[pattern_name] = config

    def has_pattern(self, pattern_name: str) -> bool:
        """Check if pattern is configured."""
        return pattern_name in self.patterns

    def get_pattern_config(self, pattern_name: str) -> dict[str, Any]:
        """Get configuration for specific pattern."""
        return self.pattern_configuration.get(pattern_name, {})

    @property
    def is_custom(self) -> bool:
        """Check if using custom template."""
        return self.variant == EnumTemplateVariant.CUSTOM

    @property
    def is_production(self) -> bool:
        """Check if using production template."""
        return self.variant == EnumTemplateVariant.PRODUCTION


@dataclass
class ModelGenerationDirectives:
    """
    Generation directives for Phase 3 LLM-enhanced code generation.

    Controls LLM behavior, context building, quality levels, and fallback
    strategies for intelligent code generation.

    Attributes:
        enable_llm: Enable LLM for business logic generation
        llm_tier: LLM tier to use (LOCAL/CLOUD_FAST/CLOUD_ACCURATE)
        quality_level: Target quality level (minimal/standard/production)
        fallback_strategy: What to do when generation fails
        include_patterns: Include pattern examples in LLM context
        include_references: Include similar node references in LLM context
        max_context_size: Maximum tokens for LLM context
        timeout_seconds: LLM generation timeout
        retry_attempts: Number of retry attempts on failure
    """

    enable_llm: bool = True
    llm_tier: EnumLLMTier = EnumLLMTier.CLOUD_FAST
    quality_level: EnumQualityLevel = EnumQualityLevel.STANDARD
    fallback_strategy: EnumFallbackStrategy = EnumFallbackStrategy.GRACEFUL

    # Context enhancement
    include_patterns: bool = True
    include_references: bool = True
    max_context_size: int = 8000

    # Performance tuning
    timeout_seconds: int = 30
    retry_attempts: int = 3

    @property
    def is_llm_enabled(self) -> bool:
        """Check if LLM generation is enabled."""
        return self.enable_llm

    @property
    def is_production_quality(self) -> bool:
        """Check if targeting production quality."""
        return self.quality_level == EnumQualityLevel.PRODUCTION

    @property
    def should_include_context_enhancements(self) -> bool:
        """Check if context enhancements are enabled."""
        return self.include_patterns or self.include_references


@dataclass
class ModelQualityGate:
    """
    Quality gate configuration for Phase 3 validation.

    Defines a single quality gate with validation rules and requirements.

    Attributes:
        name: Gate name (e.g., 'syntax_validation', 'onex_compliance')
        required: Whether this gate must pass
        config: Gate-specific configuration
        description: Human-readable description
    """

    name: str
    required: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def is_required(self) -> bool:
        """Check if this gate is required."""
        return self.required


@dataclass
class ModelQualityGatesConfiguration:
    """
    Quality gates configuration for Phase 3 validation pipeline.

    Defines multiple quality gates that generated code must pass through.

    Attributes:
        gates: List of quality gates
        fail_on_first_error: Stop validation on first failure
        collect_warnings: Collect warnings even if gates pass
    """

    gates: list[ModelQualityGate] = field(default_factory=list)
    fail_on_first_error: bool = True
    collect_warnings: bool = True

    def add_gate(
        self,
        name: str,
        required: bool = True,
        config: Optional[dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        """
        Add a quality gate.

        Args:
            name: Gate name
            required: Whether gate must pass
            config: Gate-specific configuration
            description: Human-readable description
        """
        gate = ModelQualityGate(
            name=name, required=required, config=config or {}, description=description
        )
        self.gates.append(gate)

    def get_gate(self, name: str) -> Optional[ModelQualityGate]:
        """Get quality gate by name."""
        for gate in self.gates:
            if gate.name == name:
                return gate
        return None

    def has_gate(self, name: str) -> bool:
        """Check if gate is configured."""
        return self.get_gate(name) is not None

    def get_required_gates(self) -> list[ModelQualityGate]:
        """Get list of required gates."""
        return [gate for gate in self.gates if gate.required]

    def get_optional_gates(self) -> list[ModelQualityGate]:
        """Get list of optional gates."""
        return [gate for gate in self.gates if not gate.required]


@dataclass
class ModelEnhancedContract:
    """
    Enhanced contract representation supporting v1.0, v2.0, and v2.1 contracts.

    Combines core contract fields with v2.0 mixin/advanced features support
    and v2.1 (Phase 3) template/generation enhancements.
    Backward compatible with v1.0 and v2.0 contracts (all advanced fields optional).

    Attributes:
        # Core fields (required in all versions)
        name: Node class name (follows NodeXxxYyy pattern)
        version: Semantic version (major.minor.patch)
        node_type: Node type (effect, compute, reducer, orchestrator)
        description: Human-readable description

        # Optional v1.0/v2.0 fields
        capabilities: List of node capabilities
        endpoints: Service endpoints configuration
        dependencies: Service dependencies
        configuration: Configuration parameters
        subcontracts: Subcontract references
        performance_targets: Performance SLAs
        health_checks: Health check configuration
        input_state: Input state schema
        output_state: Output state schema
        io_operations: I/O operations (for effect nodes)
        definitions: Model definitions

        # v2.0 fields (optional for backward compatibility)
        schema_version: Contract schema version (e.g., 'v2.0.0', 'v2.1.0')
        mixins: List of mixin declarations
        advanced_features: Advanced features configuration

        # Phase 3 (v2.1) fields (optional for backward compatibility)
        template: Template configuration (variant, patterns, custom template)
        generation: Generation directives (LLM tier, quality level, fallback)
        quality_gates: Quality gates configuration (validation pipeline)

        # Validation state
        has_errors: Whether contract has validation errors
        validation_errors: List of validation error messages
    """

    # Core fields (required)
    name: str
    version: ModelVersionInfo
    node_type: str
    description: str

    # Optional v1.0/v2.0 fields
    capabilities: list[dict[str, str]] = field(default_factory=list)
    endpoints: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    subcontracts: dict[str, Any] = field(default_factory=dict)
    performance_targets: dict[str, Any] = field(default_factory=dict)
    health_checks: dict[str, Any] = field(default_factory=dict)
    input_state: dict[str, Any] = field(default_factory=dict)
    output_state: dict[str, Any] = field(default_factory=dict)
    io_operations: list[dict[str, Any]] = field(default_factory=list)
    definitions: dict[str, Any] = field(default_factory=dict)

    # v2.0 fields
    schema_version: str = "v1.0.0"  # Default to v1.0.0 for backward compatibility
    mixins: list[ModelMixinDeclaration] = field(default_factory=list)
    advanced_features: Optional[ModelAdvancedFeatures] = None

    # Phase 3 (v2.1) fields - Optional for backward compatibility
    template: ModelTemplateConfiguration = field(
        default_factory=ModelTemplateConfiguration
    )
    generation: ModelGenerationDirectives = field(
        default_factory=ModelGenerationDirectives
    )
    quality_gates: ModelQualityGatesConfiguration = field(
        default_factory=ModelQualityGatesConfiguration
    )

    # Deprecated fields (v1.0 -> v2.0 migration)
    error_handling: dict[str, Any] = field(
        default_factory=dict
    )  # DEPRECATED: use advanced_features

    # Validation state
    has_errors: bool = False
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_v2(self) -> bool:
        """Check if contract is v2.0 (has schema_version starting with 'v2')."""
        return self.schema_version.startswith("v2")

    @property
    def is_valid(self) -> bool:
        """Check if contract has no validation errors."""
        return not self.has_errors and len(self.validation_errors) == 0

    def add_validation_error(self, error: str) -> None:
        """Add validation error message."""
        self.validation_errors.append(error)
        self.has_errors = True

    def get_enabled_mixins(self) -> list[ModelMixinDeclaration]:
        """Get list of enabled mixin declarations."""
        return [mixin for mixin in self.mixins if mixin.enabled]

    def get_mixin_names(self) -> list[str]:
        """Get list of enabled mixin names."""
        return [mixin.name for mixin in self.get_enabled_mixins()]

    def has_mixin(self, mixin_name: str) -> bool:
        """Check if contract has specific enabled mixin."""
        return mixin_name in self.get_mixin_names()

    def has_deprecated_error_handling(self) -> bool:
        """Check if contract uses deprecated error_handling field."""
        return bool(self.error_handling)

    # Phase 3 helper methods
    @property
    def is_v2_1(self) -> bool:
        """Check if contract is v2.1 or higher (has Phase 3 enhancements)."""
        return self.schema_version.startswith("v2.1") or self.schema_version.startswith(
            "v2.2"
        )

    @property
    def is_llm_enabled(self) -> bool:
        """Check if LLM generation is enabled."""
        return self.generation.is_llm_enabled

    @property
    def is_production_quality(self) -> bool:
        """Check if targeting production quality."""
        return self.generation.is_production_quality

    def get_template_patterns(self) -> list[str]:
        """Get list of requested template patterns."""
        return self.template.patterns

    def has_pattern(self, pattern_name: str) -> bool:
        """Check if contract requests specific pattern."""
        return self.template.has_pattern(pattern_name)

    def get_required_quality_gates(self) -> list[ModelQualityGate]:
        """Get list of required quality gates."""
        return self.quality_gates.get_required_gates()

    def get_optional_quality_gates(self) -> list[ModelQualityGate]:
        """Get list of optional quality gates."""
        return self.quality_gates.get_optional_gates()

    def has_quality_gate(self, gate_name: str) -> bool:
        """Check if quality gate is configured."""
        return self.quality_gates.has_gate(gate_name)


# Export
__all__ = [
    # Enums
    "EnumTemplateVariant",
    "EnumLLMTier",
    "EnumQualityLevel",
    "EnumFallbackStrategy",
    # Mixin models
    "ModelMixinDeclaration",
    # Advanced features models
    "ModelCircuitBreakerConfig",
    "ModelRetryPolicyConfig",
    "ModelDeadLetterQueueConfig",
    "ModelTransactionsConfig",
    "ModelSecurityValidationConfig",
    "ModelObservabilityConfig",
    "ModelAdvancedFeatures",
    # Phase 3 models
    "ModelTemplateConfiguration",
    "ModelGenerationDirectives",
    "ModelQualityGate",
    "ModelQualityGatesConfiguration",
    # Core models
    "ModelVersionInfo",
    "ModelEnhancedContract",
]
