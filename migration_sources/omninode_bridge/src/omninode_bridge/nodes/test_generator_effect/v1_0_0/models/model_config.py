"""
Test Generator Configuration Model - ONEX v2.0 Compliant.

Configuration model for test generation node.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelTestGeneratorConfig(BaseModel):
    """
    Configuration model for Test Generator Effect Node.

    Controls template rendering, file writing, and test generation behavior.
    """

    # === TEMPLATE CONFIGURATION ===

    template_directory: Path = Field(
        default=Path("src/omninode_bridge/codegen/templates/test_templates"),
        description="Directory containing Jinja2 test templates",
    )

    template_autoescape: bool = Field(
        default=False,
        description="Enable Jinja2 autoescape (should be False for code generation)",
    )

    # === FILE WRITING CONFIGURATION ===

    enable_fixtures: bool = Field(
        default=True,
        description="Whether to generate pytest fixtures in conftest.py",
    )

    overwrite_existing: bool = Field(
        default=False,
        description="Whether to overwrite existing test files",
    )

    create_directories: bool = Field(
        default=True,
        description="Whether to create missing directories automatically",
    )

    # === CODE GENERATION OPTIONS ===

    include_docstrings: bool = Field(
        default=True,
        description="Whether to include docstrings in generated tests",
    )

    include_type_hints: bool = Field(
        default=True,
        description="Whether to include type hints in generated tests",
    )

    use_async_tests: bool = Field(
        default=True,
        description="Whether to generate async test functions by default",
    )

    parametrize_tests: bool = Field(
        default=True,
        description="Whether to use pytest.mark.parametrize for test variations",
    )

    # === PERFORMANCE CONFIGURATION ===

    max_template_render_time_ms: int = Field(
        default=2000,
        description="Maximum time allowed for template rendering (ms)",
        ge=100,
    )

    max_file_write_time_ms: int = Field(
        default=1000,
        description="Maximum time allowed for file writing (ms)",
        ge=100,
    )

    # === VALIDATION CONFIGURATION ===

    validate_generated_code: bool = Field(
        default=True,
        description="Whether to validate generated Python code syntax",
    )

    enforce_test_naming: bool = Field(
        default=True,
        description="Whether to enforce test naming conventions (test_*)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "template_directory": "src/omninode_bridge/codegen/templates/test_templates",
                "enable_fixtures": True,
                "overwrite_existing": False,
                "include_docstrings": True,
                "use_async_tests": True,
                "max_template_render_time_ms": 2000,
                "validate_generated_code": True,
            }
        }
    )


__all__ = ["ModelTestGeneratorConfig"]
