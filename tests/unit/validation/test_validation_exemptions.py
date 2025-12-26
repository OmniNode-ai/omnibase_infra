"""
Tests for validation_exemptions.yaml regex pattern validity.

This module provides pre-commit validation that all regex patterns in the
validation_exemptions.yaml file are syntactically valid. Invalid regex patterns
would cause runtime errors during validation, so catching them early is critical.

The tests validate:
1. All regex patterns compile without re.error exceptions
2. YAML schema version is present and valid
3. All exemption sections are properly structured
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

# -----------------------------------------------------------------------------
# Module-level constants for path resolution and configuration
# -----------------------------------------------------------------------------

# Compute project root from test file location for robust path resolution
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent.parent  # tests/unit/validation -> root

# Relative paths from project root
_EXEMPTIONS_YAML_REL_PATH = "src/omnibase_infra/validation/validation_exemptions.yaml"
_SOURCE_DIR_REL_PATH = "src/omnibase_infra"

# Exemption sections that should be present in the YAML file
EXEMPTION_SECTIONS = (
    "pattern_exemptions",
    "architecture_exemptions",
    "union_exemptions",
)

# Pattern field names to extract from exemptions
PATTERN_FIELDS = (
    "file_pattern",
    "class_pattern",
    "method_pattern",
    "violation_pattern",
)

# Required fields for each exemption entry
REQUIRED_EXEMPTION_FIELDS = frozenset({"file_pattern", "violation_pattern", "reason"})

# Key files that have exemptions defined - used for existence validation
KEY_EXEMPTION_FILES = (
    "src/omnibase_infra/event_bus/kafka_event_bus.py",
    "src/omnibase_infra/runtime/runtime_host_process.py",
    "src/omnibase_infra/runtime/message_dispatch_engine.py",
    "src/omnibase_infra/mixins/mixin_node_introspection.py",
    "src/omnibase_infra/validation/execution_shape_validator.py",
)


# -----------------------------------------------------------------------------
# Module-level fixtures for shared use across test classes
# -----------------------------------------------------------------------------


@pytest.fixture
def project_root() -> Path:
    """Return the project root path for consistent path resolution."""
    return _PROJECT_ROOT


@pytest.fixture
def exemptions_yaml_path() -> Path:
    """Return the absolute path to the exemptions YAML file."""
    return _PROJECT_ROOT / _EXEMPTIONS_YAML_REL_PATH


@pytest.fixture
def exemptions_yaml(exemptions_yaml_path: Path) -> dict[str, Any]:
    """Load the exemptions YAML file."""
    with open(exemptions_yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def source_files() -> list[str]:
    """Get list of all Python source files in the codebase."""
    src_path = _PROJECT_ROOT / _SOURCE_DIR_REL_PATH
    return [str(f.relative_to(_PROJECT_ROOT)) for f in src_path.rglob("*.py")]


@pytest.fixture
def all_regex_patterns(exemptions_yaml: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Extract all regex patterns from the exemptions YAML file.

    Returns:
        List of tuples: (section_name, pattern_field, pattern_value)
    """
    patterns: list[tuple[str, str, str]] = []

    for section in EXEMPTION_SECTIONS:
        for exemption in exemptions_yaml.get(section, []):
            for field in PATTERN_FIELDS:
                if field in exemption:
                    patterns.append((section, field, exemption[field]))

    return patterns


class TestValidationExemptionsRegex:
    """Tests for validation_exemptions.yaml regex pattern validity."""

    def test_yaml_file_exists(self, exemptions_yaml_path: Path) -> None:
        """Verify the exemptions YAML file exists."""
        assert exemptions_yaml_path.exists(), (
            f"Exemptions YAML file not found at {exemptions_yaml_path}"
        )

    def test_yaml_file_is_valid_yaml(self, exemptions_yaml_path: Path) -> None:
        """Verify the exemptions file is valid YAML."""
        with open(exemptions_yaml_path) as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in {exemptions_yaml_path}: {e}")

    def test_schema_version_present(self, exemptions_yaml: dict[str, Any]) -> None:
        """Verify schema_version is present in the YAML file."""
        assert "schema_version" in exemptions_yaml, (
            "schema_version field is required in validation_exemptions.yaml"
        )
        assert exemptions_yaml["schema_version"] == "1.0.0", (
            f"Expected schema_version '1.0.0', got '{exemptions_yaml['schema_version']}'"
        )

    def test_all_regex_patterns_are_valid(
        self, all_regex_patterns: list[tuple[str, str, str]]
    ) -> None:
        """Verify all regex patterns compile without errors.

        This is the critical pre-commit test that catches invalid regex patterns
        before they reach production and cause runtime errors.
        """
        invalid_patterns: list[tuple[str, str, str, str]] = []

        for section, field, pattern in all_regex_patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                invalid_patterns.append((section, field, pattern, str(e)))

        if invalid_patterns:
            error_messages = [
                f"  [{section}] {field}: '{pattern}' - {error}"
                for section, field, pattern, error in invalid_patterns
            ]
            pytest.fail(
                f"Found {len(invalid_patterns)} invalid regex pattern(s):\n"
                + "\n".join(error_messages)
            )

    @pytest.mark.parametrize("section", EXEMPTION_SECTIONS)
    def test_exemption_section_exists_and_is_list(
        self, exemptions_yaml: dict[str, Any], section: str
    ) -> None:
        """Verify each exemption section exists and is a list."""
        assert section in exemptions_yaml, f"{section} section is required"
        assert isinstance(exemptions_yaml[section], list), f"{section} must be a list"

    def test_all_exemptions_have_required_fields(
        self, exemptions_yaml: dict[str, Any]
    ) -> None:
        """Verify all exemptions have required fields: file_pattern, violation_pattern, reason."""
        missing_fields_errors: list[str] = []

        for section in EXEMPTION_SECTIONS:
            for idx, exemption in enumerate(exemptions_yaml.get(section, [])):
                missing = REQUIRED_EXEMPTION_FIELDS - set(exemption.keys())
                if missing:
                    missing_fields_errors.append(
                        f"  [{section}][{idx}]: Missing fields: {missing}"
                    )

        if missing_fields_errors:
            pytest.fail(
                "Found exemptions with missing required fields:\n"
                + "\n".join(missing_fields_errors)
            )

    def test_pattern_count_sanity_check(
        self, all_regex_patterns: list[tuple[str, str, str]]
    ) -> None:
        """Verify a reasonable number of patterns exist (sanity check)."""
        # At the time of writing, there are many patterns. This test ensures
        # the extraction is working and we have a reasonable number.
        assert len(all_regex_patterns) >= 50, (
            f"Expected at least 50 patterns, found {len(all_regex_patterns)}. "
            "This may indicate a problem with pattern extraction."
        )


class TestExemptionPatternsMatchFiles:
    """Optional tests to verify patterns can match expected files."""

    def test_file_patterns_match_at_least_one_file(
        self, exemptions_yaml: dict[str, Any], source_files: list[str]
    ) -> None:
        """Verify each file_pattern matches at least one file in the codebase.

        This catches patterns that reference files that have been renamed or deleted.
        """
        orphaned_patterns: list[tuple[str, str]] = []

        for section in EXEMPTION_SECTIONS:
            for exemption in exemptions_yaml.get(section, []):
                file_pattern = exemption.get("file_pattern")
                if file_pattern:
                    pattern = re.compile(file_pattern)
                    matches_any = any(pattern.search(f) for f in source_files)
                    if not matches_any:
                        orphaned_patterns.append((section, file_pattern))

        if orphaned_patterns:
            warnings = [
                f"  [{section}] file_pattern: '{pattern}' - matches no files"
                for section, pattern in orphaned_patterns
            ]
            pytest.fail(
                f"Found {len(orphaned_patterns)} file_pattern(s) that match no files "
                "(files may have been renamed or deleted):\n" + "\n".join(warnings)
            )

    def test_known_exemption_files_exist(self, project_root: Path) -> None:
        """Verify key files that have exemptions actually exist."""
        for file_path in KEY_EXEMPTION_FILES:
            full_path = project_root / file_path
            assert full_path.exists(), (
                f"Expected file {file_path} to exist (has exemptions defined)"
            )
