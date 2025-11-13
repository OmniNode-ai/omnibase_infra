"""Analysis type enumeration for code generation requests.

This enum defines the types of analysis that can be performed
on PRD content during the code generation workflow.
"""

from enum import Enum


class EnumAnalysisType(str, Enum):
    """
    Types of analysis for PRD processing.

    Inherits from str for JSON serialization compatibility with Pydantic v2.

    Values:
        FULL: Comprehensive analysis including requirements, architecture, and dependencies
        PARTIAL: Targeted analysis of specific sections or requirements
        QUICK: Fast analysis for basic validation and overview
    """

    FULL = "full"
    PARTIAL = "partial"
    QUICK = "quick"
