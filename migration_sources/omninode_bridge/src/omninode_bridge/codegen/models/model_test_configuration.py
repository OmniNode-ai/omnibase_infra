"""
Test Configuration Model - ONEX Standards Compliant.

VERSION: 1.0.0 - INTERFACE LOCKED FOR CODE GENERATION

STABILITY GUARANTEE:
- All fields and methods are stable interfaces
- New optional fields may be added in minor versions only
- Existing fields cannot be removed or have types/constraints changed

Defines pytest and test execution configuration for generated tests.

ZERO TOLERANCE: No Any types allowed in implementation.
"""

from typing import ClassVar

from omnibase_core.primitives.model_semver import ModelSemVer

# Type aliases for structured data - ZERO TOLERANCE for Any types
from omnibase_core.types.constraints import PrimitiveValueType
from pydantic import BaseModel, ConfigDict, Field

type ParameterValue = PrimitiveValueType
type StructuredData = dict[str, ParameterValue]


class ModelTestConfiguration(BaseModel):
    """
    Test configuration specification for pytest and test execution.

    Defines pytest settings, markers, fixtures, and execution parameters
    for generated tests.

    ZERO TOLERANCE: No Any types allowed in implementation.
    """

    # Interface version for code generation stability
    INTERFACE_VERSION: ClassVar[ModelSemVer] = ModelSemVer(major=1, minor=0, patch=0)

    # Pytest configuration
    pytest_markers: list[str] = Field(
        default_factory=list,
        description="Pytest markers to apply to tests (e.g., 'slow', 'integration')",
    )

    pytest_plugins: list[str] = Field(
        default_factory=list,
        description="Pytest plugins to enable",
    )

    pytest_options: StructuredData = Field(
        default_factory=dict,
        description="Additional pytest command-line options",
    )

    # Test execution settings
    parallel_execution: bool = Field(
        default=False,
        description="Whether to enable parallel test execution",
    )

    parallel_workers: int = Field(
        default=1,
        ge=1,
        description="Number of parallel workers for test execution",
    )

    timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Test execution timeout in seconds",
    )

    # Coverage configuration
    coverage_enabled: bool = Field(
        default=True,
        description="Whether to enable coverage reporting",
    )

    coverage_threshold: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Minimum coverage threshold percentage",
    )

    coverage_fail_under: bool = Field(
        default=True,
        description="Whether to fail if coverage is below threshold",
    )

    # Fixtures
    required_fixtures: list[str] = Field(
        default_factory=list,
        description="Required pytest fixtures",
    )

    fixture_configurations: StructuredData = Field(
        default_factory=dict,
        description="Fixture configuration overrides",
    )

    # Test data
    test_data_directory: str = Field(
        default="tests/data",
        description="Directory containing test data files (documentation only - templates use Path(__file__).parent / 'data')",
    )

    use_test_database: bool = Field(
        default=False,
        description="Whether to use a test database",
    )

    test_database_config: StructuredData = Field(
        default_factory=dict,
        description="Test database configuration",
    )

    # Output configuration
    verbose_output: bool = Field(
        default=True,
        description="Whether to enable verbose test output",
    )

    generate_html_report: bool = Field(
        default=True,
        description="Whether to generate HTML coverage report",
    )

    report_output_directory: str = Field(
        default="htmlcov",
        description="Directory for test reports",
    )

    # Retry configuration
    retry_failed_tests: bool = Field(
        default=False,
        description="Whether to retry failed tests",
    )

    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for failed tests",
    )

    # Environment setup
    setup_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run before tests",
    )

    teardown_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run after tests",
    )

    environment_variables: StructuredData = Field(
        default_factory=dict,
        description="Environment variables to set for tests",
    )

    # Quality gates
    enforce_type_checking: bool = Field(
        default=True,
        description="Whether to enforce type checking in tests",
    )

    enforce_linting: bool = Field(
        default=True,
        description="Whether to enforce linting in tests",
    )

    allowed_warnings: list[str] = Field(
        default_factory=list,
        description="Warning patterns that are allowed",
    )

    model_config = ConfigDict(
        extra="forbid",  # Strict validation
        use_enum_values=False,
        validate_assignment=True,
    )
