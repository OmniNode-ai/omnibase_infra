"""
Test Target Model - ONEX Standards Compliant.

VERSION: 1.0.0 - INTERFACE LOCKED FOR CODE GENERATION

STABILITY GUARANTEE:
- All fields and methods are stable interfaces
- New optional fields may be added in minor versions only
- Existing fields cannot be removed or have types/constraints changed

Defines a specific method or component to be tested with scenarios.

ZERO TOLERANCE: No Any types allowed in implementation.
"""

from typing import ClassVar

from omnibase_core.primitives.model_semver import ModelSemVer

# Type aliases for structured data - ZERO TOLERANCE for Any types
from omnibase_core.types.constraints import PrimitiveValueType
from pydantic import BaseModel, ConfigDict, Field

type ParameterValue = PrimitiveValueType
type StructuredData = dict[str, ParameterValue]


class ModelTestTarget(BaseModel):
    """
    Test target specification for individual test cases.

    Defines what to test (method/component), test scenarios,
    expected behaviors, and assertions.

    ZERO TOLERANCE: No Any types allowed in implementation.
    """

    # Interface version for code generation stability
    INTERFACE_VERSION: ClassVar[ModelSemVer] = ModelSemVer(major=1, minor=0, patch=0)

    # Target identification
    target_name: str = Field(
        ...,
        description="Name of the method, function, or component being tested",
        min_length=1,
    )

    target_type: str = Field(
        default="method",
        description="Type of target (method, function, class, module)",
    )

    # Test scenarios
    test_scenarios: list[str] = Field(
        default_factory=list,
        description="List of test scenarios to validate",
    )

    # Expected behaviors
    expected_behaviors: list[str] = Field(
        default_factory=list,
        description="List of expected behaviors to verify",
    )

    # Input/output specifications
    input_parameters: StructuredData = Field(
        default_factory=dict,
        description="Input parameter specifications for test cases",
    )

    expected_outputs: StructuredData = Field(
        default_factory=dict,
        description="Expected output specifications",
    )

    # Edge cases and error conditions
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Edge cases to test",
    )

    error_conditions: list[str] = Field(
        default_factory=list,
        description="Error conditions to validate",
    )

    # Assertions
    assertions: list[str] = Field(
        default_factory=list,
        description="Specific assertions to include in tests",
    )

    # Test configuration
    test_priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Test priority (1=critical, 5=optional)",
    )

    skip_test: bool = Field(
        default=False,
        description="Whether to skip this test target",
    )

    skip_reason: str | None = Field(
        default=None,
        description="Reason for skipping this test target",
    )

    model_config = ConfigDict(
        extra="forbid",  # Strict validation
        use_enum_values=False,
        validate_assignment=True,
    )
