"""
Mock Requirements Model - ONEX Standards Compliant.

VERSION: 1.0.0 - INTERFACE LOCKED FOR CODE GENERATION

STABILITY GUARANTEE:
- All fields and methods are stable interfaces
- New optional fields may be added in minor versions only
- Existing fields cannot be removed or have types/constraints changed

Defines mock requirements for test generation including dependencies to mock
and their configurations.

ZERO TOLERANCE: No Any types allowed in implementation.
"""

from typing import ClassVar

from omnibase_core.primitives.model_semver import ModelSemVer

# Type aliases for structured data - ZERO TOLERANCE for Any types
from omnibase_core.types.constraints import PrimitiveValueType
from pydantic import BaseModel, ConfigDict, Field

type ParameterValue = PrimitiveValueType
type StructuredData = dict[str, ParameterValue]
type StructuredDataList = list[StructuredData]


class ModelMockRequirements(BaseModel):
    """
    Mock requirements specification for test generation.

    Defines which dependencies should be mocked, their configurations,
    and mock behaviors.

    ZERO TOLERANCE: No Any types allowed in implementation.
    """

    # Interface version for code generation stability
    INTERFACE_VERSION: ClassVar[ModelSemVer] = ModelSemVer(major=1, minor=0, patch=0)

    # Dependencies to mock
    mock_dependencies: list[str] = Field(
        default_factory=list,
        description="List of dependencies to mock (module paths or class names)",
    )

    # External services to mock
    mock_external_services: list[str] = Field(
        default_factory=list,
        description="External services that should be mocked (APIs, databases, etc.)",
    )

    # Database mocking
    mock_database: bool = Field(
        default=False,
        description="Whether to mock database operations",
    )

    database_mock_config: StructuredData = Field(
        default_factory=dict,
        description="Database mock configuration",
    )

    # HTTP/API mocking
    mock_http_clients: bool = Field(
        default=False,
        description="Whether to mock HTTP clients",
    )

    http_mock_responses: StructuredDataList = Field(
        default_factory=list,
        description="HTTP mock response configurations",
    )

    # Kafka/Event mocking
    mock_kafka_producer: bool = Field(
        default=False,
        description="Whether to mock Kafka producer",
    )

    mock_kafka_consumer: bool = Field(
        default=False,
        description="Whether to mock Kafka consumer",
    )

    kafka_mock_config: StructuredData = Field(
        default_factory=dict,
        description="Kafka mock configuration",
    )

    # File system mocking
    mock_filesystem: bool = Field(
        default=False,
        description="Whether to mock filesystem operations",
    )

    filesystem_mock_config: StructuredData = Field(
        default_factory=dict,
        description="Filesystem mock configuration",
    )

    # Time/Clock mocking
    mock_datetime: bool = Field(
        default=False,
        description="Whether to mock datetime operations",
    )

    datetime_mock_config: StructuredData = Field(
        default_factory=dict,
        description="Datetime mock configuration (fixed time, etc.)",
    )

    # Environment variables
    mock_environment: bool = Field(
        default=False,
        description="Whether to mock environment variables",
    )

    environment_mock_config: StructuredData = Field(
        default_factory=dict,
        description="Environment variable mock configuration",
    )

    # Custom mocks
    custom_mocks: StructuredDataList = Field(
        default_factory=list,
        description="Custom mock specifications",
    )

    # Mock behavior configuration
    mock_return_values: StructuredData = Field(
        default_factory=dict,
        description="Mock return value specifications",
    )

    mock_side_effects: list[str] = Field(
        default_factory=list,
        description="Mock side effect specifications",
    )

    mock_exceptions: list[str] = Field(
        default_factory=list,
        description="Exceptions that mocks should raise for testing error handling",
    )

    # Fixture configuration
    use_fixtures: bool = Field(
        default=True,
        description="Whether to use pytest fixtures for mocks",
    )

    fixture_scope: str = Field(
        default="function",
        description="Pytest fixture scope (function, class, module, session)",
    )

    model_config = ConfigDict(
        extra="forbid",  # Strict validation
        use_enum_values=False,
        validate_assignment=True,
    )
