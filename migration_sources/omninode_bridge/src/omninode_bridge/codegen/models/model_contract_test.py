"""
Test Contract Model - ONEX Standards Compliant.

VERSION: 1.0.0 - INTERFACE LOCKED FOR CODE GENERATION

STABILITY GUARANTEE:
- All fields, methods, and validators are stable interfaces
- New optional fields may be added in minor versions only
- Existing fields cannot be removed or have types/constraints changed

Specialized contract model for test generation providing:
- Test coverage requirements and targets
- Test type specifications (unit, integration, contract, performance)
- Mock configuration and dependencies
- Test target specifications with scenarios
- Assertion requirements and validation

ZERO TOLERANCE: No Any types allowed in implementation.

NOTE: This is a prototype implementation in omninode_bridge that will be
migrated to omnibase_core once the pattern is validated.
"""

from typing import ClassVar, cast
from uuid import UUID, uuid4

from omnibase_core.errors.error_codes import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.common.model_error_context import ModelErrorContext
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.primitives.model_semver import ModelSemVer

# Type aliases for structured data - ZERO TOLERANCE for Any types
from omnibase_core.types.constraints import PrimitiveValueType
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .enum_test_type import EnumTestType
from .model_mock_requirements import ModelMockRequirements
from .model_test_configuration import ModelTestConfiguration
from .model_test_target import ModelTestTarget

type ParameterValue = PrimitiveValueType
type StructuredData = dict[str, ParameterValue]
type StructuredDataList = list[StructuredData]


class ModelContractTest(BaseModel):
    """
    Contract model for test generation - Clean Architecture Pattern.

    Specialized contract for test generation following the same patterns
    as ModelContractEffect but tailored for test specifications.

    ZERO TOLERANCE: No Any types allowed in implementation.
    """

    # Interface version for code generation stability
    INTERFACE_VERSION: ClassVar[ModelSemVer] = ModelSemVer(major=1, minor=0, patch=0)

    # === CORE TEST CONTRACT IDENTIFICATION ===

    # Contract identification
    name: str = Field(
        ...,
        description="Unique test contract name for identification",
        min_length=1,
    )

    version: ModelSemVer = Field(
        ...,
        description="Semantic version following SemVer specification",
    )

    description: str = Field(
        ...,
        description="Human-readable test contract description",
        min_length=1,
    )

    # UUID correlation tracking for ONEX compliance
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for correlation tracking across system boundaries",
    )

    execution_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for tracking individual test execution instances",
    )

    # === TEST TARGET IDENTIFICATION ===

    # Target node information
    target_node: str = Field(
        ...,
        description="Name of the node being tested",
        min_length=1,
    )

    target_version: str = Field(
        ...,
        description="Version of the target node",
        min_length=1,
    )

    target_node_type: str = Field(
        ...,
        description="Type of target node (effect, compute, reducer, orchestrator)",
        min_length=1,
    )

    # Test suite information
    test_suite_name: str = Field(
        ...,
        description="Name of the test suite to generate",
        min_length=1,
    )

    test_file_path: str = Field(
        default="tests/",
        description="Path where test file should be generated",
    )

    # === COVERAGE REQUIREMENTS ===

    # Coverage targets
    coverage_minimum: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Minimum acceptable coverage percentage",
    )

    coverage_target: int = Field(
        default=95,
        ge=0,
        le=100,
        description="Target coverage percentage",
    )

    coverage_by_file: bool = Field(
        default=True,
        description="Whether to enforce coverage per file",
    )

    coverage_by_function: bool = Field(
        default=True,
        description="Whether to enforce coverage per function",
    )

    # === TEST TYPE SPECIFICATIONS ===

    # Test types to generate
    test_types: list[EnumTestType] = Field(
        default_factory=lambda: [EnumTestType.UNIT],
        description="Types of tests to generate",
        min_length=1,
    )

    # === TEST TARGETS AND SCENARIOS ===

    # Test targets (methods/functions to test)
    test_targets: list[ModelTestTarget] = Field(
        default_factory=list,
        description="Specific methods/functions to test with scenarios",
    )

    # === MOCK REQUIREMENTS ===

    # Mock configuration
    mock_requirements: ModelMockRequirements = Field(
        default_factory=ModelMockRequirements,
        description="Mock requirements and configuration",
    )

    # === ASSERTION REQUIREMENTS ===

    # Assertion configuration
    assertion_types: list[str] = Field(
        default_factory=list,
        description="Types of assertions to include (equality, type, exception, etc.)",
    )

    custom_assertions: list[str] = Field(
        default_factory=list,
        description="Custom assertion functions to include",
    )

    # === TEST CONFIGURATION ===

    # Test execution configuration
    test_configuration: ModelTestConfiguration = Field(
        default_factory=ModelTestConfiguration,
        description="Pytest and execution configuration",
    )

    # === TEST GENERATION OPTIONS ===

    # Generation options
    include_docstrings: bool = Field(
        default=True,
        description="Whether to include docstrings in generated tests",
    )

    include_type_hints: bool = Field(
        default=True,
        description="Whether to include type hints in generated tests",
    )

    use_async_tests: bool = Field(
        default=False,
        description="Whether to generate async test functions",
    )

    parametrize_tests: bool = Field(
        default=True,
        description="Whether to use pytest.mark.parametrize for test variations",
    )

    # === QUALITY GATES ===

    # Quality requirements
    enforce_test_naming: bool = Field(
        default=True,
        description="Whether to enforce test naming conventions",
    )

    enforce_test_isolation: bool = Field(
        default=True,
        description="Whether to enforce test isolation (no shared state)",
    )

    enforce_deterministic_tests: bool = Field(
        default=True,
        description="Whether to enforce deterministic test results",
    )

    # === METADATA ===

    # Metadata and documentation
    author: str | None = Field(
        default=None,
        description="Test contract author information",
    )

    documentation_url: str | None = Field(
        default=None,
        description="URL to detailed test documentation",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Test contract classification tags",
    )

    # === FIELD VALIDATORS ===

    @field_validator("coverage_target")
    @classmethod
    def validate_coverage_target(cls, v: int, info: ValidationInfo) -> int:
        """Validate coverage target is >= coverage minimum."""
        # Note: In Pydantic v2, we need to access other fields via info.data
        if info.data and "coverage_minimum" in info.data:
            minimum = info.data["coverage_minimum"]
            if v < minimum:
                msg = f"coverage_target ({v}) must be >= coverage_minimum ({minimum})"
                raise ModelOnexError(
                    message=msg,
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    details=ModelErrorContext.with_context(
                        {
                            "error_type": ModelSchemaValue.from_value("valueerror"),
                            "validation_context": ModelSchemaValue.from_value(
                                "model_validation",
                            ),
                        },
                    ),
                )
        return v

    @field_validator("target_node_type")
    @classmethod
    def validate_target_node_type(cls, v: str) -> str:
        """Validate target node type is a valid ONEX node type."""
        valid_types = ["effect", "compute", "reducer", "orchestrator"]
        if v.lower() not in valid_types:
            msg = f"target_node_type must be one of {valid_types}, got: {v}"
            raise ModelOnexError(
                message=msg,
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                details=ModelErrorContext.with_context(
                    {
                        "error_type": ModelSchemaValue.from_value("valueerror"),
                        "validation_context": ModelSchemaValue.from_value(
                            "model_validation",
                        ),
                    },
                ),
            )
        return v.lower()

    @field_validator("test_types")
    @classmethod
    def validate_test_types_consistency(
        cls,
        v: list[EnumTestType],
    ) -> list[EnumTestType]:
        """Validate test types are consistent and appropriate."""
        if not v:
            msg = "At least one test type must be specified"
            raise ModelOnexError(
                message=msg,
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                details=ModelErrorContext.with_context(
                    {
                        "error_type": ModelSchemaValue.from_value("valueerror"),
                        "validation_context": ModelSchemaValue.from_value(
                            "model_validation",
                        ),
                    },
                ),
            )

        # Check for conflicting test types
        if EnumTestType.E2E in v and EnumTestType.UNIT in v:
            # This is allowed but should be noted
            pass

        return v

    def model_post_init(self, __context: object) -> None:
        """Post-initialization validation."""
        # Validate test contract specific requirements
        self._validate_test_contract_requirements()

    def _validate_test_contract_requirements(self) -> None:
        """Validate test contract specific requirements."""
        # Validate test targets exist if coverage is high
        if self.coverage_target > 90 and not self.test_targets:
            msg = "High coverage targets (>90%) require explicit test_targets specification"
            raise ModelOnexError(
                message=msg,
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                details=ModelErrorContext.with_context(
                    {
                        "error_type": ModelSchemaValue.from_value("valueerror"),
                        "validation_context": ModelSchemaValue.from_value(
                            "model_validation",
                        ),
                    },
                ),
            )

        # Validate mock requirements are specified for integration/e2e tests
        integration_types = [
            EnumTestType.INTEGRATION,
            EnumTestType.E2E,
            EnumTestType.LOAD,
            EnumTestType.STRESS,
        ]
        if any(t in self.test_types for t in integration_types):
            if (
                not self.mock_requirements.mock_dependencies
                and not self.mock_requirements.mock_external_services
            ):
                # This is a warning case, not an error
                pass

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields from YAML contracts
        use_enum_values=False,  # Keep enum objects, don't convert to strings
        validate_assignment=True,
    )

    def to_yaml(self) -> str:
        """
        Export contract model to YAML format.

        Returns:
            str: YAML representation of the contract
        """
        from omnibase_core.utils.safe_yaml_loader import (
            serialize_pydantic_model_to_yaml,
        )

        return cast(
            str,
            serialize_pydantic_model_to_yaml(
                self,
                default_flow_style=False,
                sort_keys=False,
            ),
        )

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "ModelContractTest":
        """
        Create contract model from YAML content with proper enum handling.

        Args:
            yaml_content: YAML string representation

        Returns:
            ModelContractTest: Validated contract model instance
        """
        import yaml
        from pydantic import ValidationError

        try:
            # Parse YAML directly without recursion
            yaml_data = yaml.safe_load(yaml_content)
            if yaml_data is None:
                yaml_data = {}

            # Validate with Pydantic model directly - avoids from_yaml recursion
            return cls.model_validate(yaml_data)

        except ValidationError as e:
            raise ModelOnexError(
                message=f"Test contract validation failed: {e}",
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                details=ModelErrorContext.with_context(
                    {
                        "error_type": ModelSchemaValue.from_value("valueerror"),
                        "validation_context": ModelSchemaValue.from_value(
                            "model_validation",
                        ),
                    },
                ),
            ) from e
        except yaml.YAMLError as e:
            raise ModelOnexError(
                message=f"YAML parsing error: {e}",
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                details=ModelErrorContext.with_context(
                    {
                        "error_type": ModelSchemaValue.from_value("valueerror"),
                        "validation_context": ModelSchemaValue.from_value(
                            "model_validation",
                        ),
                    },
                ),
            ) from e
        except Exception as e:
            raise ModelOnexError(
                message=f"Failed to load test contract YAML: {e}",
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                details=ModelErrorContext.with_context(
                    {
                        "error_type": ModelSchemaValue.from_value("valueerror"),
                        "validation_context": ModelSchemaValue.from_value(
                            "model_validation",
                        ),
                    },
                ),
            ) from e
