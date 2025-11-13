"""
Test Type Enumeration - ONEX Standards Compliant.

VERSION: 1.0.0 - INTERFACE LOCKED FOR CODE GENERATION

STABILITY GUARANTEE:
- All enum values are stable interfaces
- New values may be added in minor versions only
- Existing values cannot be removed or renamed

Defines the types of tests that can be generated for nodes.
"""

from enum import Enum


class EnumTestType(str, Enum):
    """
    Test type classification for test generation.

    Values represent different categories of automated tests
    that can be generated for ONEX nodes.
    """

    # Core test types
    UNIT = "unit"
    INTEGRATION = "integration"
    CONTRACT = "contract"
    PERFORMANCE = "performance"
    E2E = "e2e"

    # Specialized test types
    LOAD = "load"
    STRESS = "stress"
    SECURITY = "security"
    REGRESSION = "regression"
    SMOKE = "smoke"

    def __str__(self) -> str:
        """String representation of test type."""
        return self.value
