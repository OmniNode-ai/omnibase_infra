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


class TestValidationExemptionsRegex:
    """Tests for validation_exemptions.yaml regex pattern validity."""

    @pytest.fixture
    def exemptions_yaml_path(self) -> Path:
        """Return the path to the exemptions YAML file."""
        return Path("src/omnibase_infra/validation/validation_exemptions.yaml")

    @pytest.fixture
    def exemptions_yaml(self, exemptions_yaml_path: Path) -> dict[str, Any]:
        """Load the exemptions YAML file."""
        with open(exemptions_yaml_path) as f:
            return yaml.safe_load(f)

    def _extract_all_patterns(
        self, exemptions_yaml: dict[str, Any]
    ) -> list[tuple[str, str, str]]:
        """Extract all regex patterns from the YAML file.

        Returns:
            List of tuples: (section_name, pattern_field, pattern_value)
        """
        patterns: list[tuple[str, str, str]] = []
        pattern_fields = [
            "file_pattern",
            "class_pattern",
            "method_pattern",
            "violation_pattern",
        ]

        # Extract from pattern_exemptions
        for exemption in exemptions_yaml.get("pattern_exemptions", []):
            for field in pattern_fields:
                if field in exemption:
                    patterns.append(("pattern_exemptions", field, exemption[field]))

        # Extract from architecture_exemptions
        for exemption in exemptions_yaml.get("architecture_exemptions", []):
            for field in pattern_fields:
                if field in exemption:
                    patterns.append(
                        ("architecture_exemptions", field, exemption[field])
                    )

        # Extract from union_exemptions
        for exemption in exemptions_yaml.get("union_exemptions", []):
            for field in pattern_fields:
                if field in exemption:
                    patterns.append(("union_exemptions", field, exemption[field]))

        return patterns

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
        self, exemptions_yaml: dict[str, Any]
    ) -> None:
        """Verify all regex patterns compile without errors.

        This is the critical pre-commit test that catches invalid regex patterns
        before they reach production and cause runtime errors.
        """
        patterns = self._extract_all_patterns(exemptions_yaml)
        invalid_patterns: list[tuple[str, str, str, str]] = []

        for section, field, pattern in patterns:
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

    def test_pattern_exemptions_section_exists(
        self, exemptions_yaml: dict[str, Any]
    ) -> None:
        """Verify pattern_exemptions section exists and is a list."""
        assert "pattern_exemptions" in exemptions_yaml, (
            "pattern_exemptions section is required"
        )
        assert isinstance(exemptions_yaml["pattern_exemptions"], list), (
            "pattern_exemptions must be a list"
        )

    def test_architecture_exemptions_section_exists(
        self, exemptions_yaml: dict[str, Any]
    ) -> None:
        """Verify architecture_exemptions section exists and is a list."""
        assert "architecture_exemptions" in exemptions_yaml, (
            "architecture_exemptions section is required"
        )
        assert isinstance(exemptions_yaml["architecture_exemptions"], list), (
            "architecture_exemptions must be a list"
        )

    def test_union_exemptions_section_exists(
        self, exemptions_yaml: dict[str, Any]
    ) -> None:
        """Verify union_exemptions section exists and is a list."""
        assert "union_exemptions" in exemptions_yaml, (
            "union_exemptions section is required"
        )
        assert isinstance(exemptions_yaml["union_exemptions"], list), (
            "union_exemptions must be a list"
        )

    def test_all_exemptions_have_required_fields(
        self, exemptions_yaml: dict[str, Any]
    ) -> None:
        """Verify all exemptions have required fields: file_pattern, violation_pattern, reason."""
        required_fields = {"file_pattern", "violation_pattern", "reason"}
        missing_fields_errors: list[str] = []

        for section in [
            "pattern_exemptions",
            "architecture_exemptions",
            "union_exemptions",
        ]:
            for idx, exemption in enumerate(exemptions_yaml.get(section, [])):
                missing = required_fields - set(exemption.keys())
                if missing:
                    missing_fields_errors.append(
                        f"  [{section}][{idx}]: Missing fields: {missing}"
                    )

        if missing_fields_errors:
            pytest.fail(
                "Found exemptions with missing required fields:\n"
                + "\n".join(missing_fields_errors)
            )

    def test_pattern_count_sanity_check(self, exemptions_yaml: dict[str, Any]) -> None:
        """Verify a reasonable number of patterns exist (sanity check)."""
        patterns = self._extract_all_patterns(exemptions_yaml)
        # At the time of writing, there are many patterns. This test ensures
        # the extraction is working and we have a reasonable number.
        assert len(patterns) >= 50, (
            f"Expected at least 50 patterns, found {len(patterns)}. "
            "This may indicate a problem with pattern extraction."
        )


class TestExemptionPatternsMatchFiles:
    """Optional tests to verify patterns can match expected files."""

    @pytest.fixture
    def exemptions_yaml(self) -> dict[str, Any]:
        """Load the exemptions YAML file."""
        path = Path("src/omnibase_infra/validation/validation_exemptions.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def source_files(self) -> list[str]:
        """Get list of all Python source files in the codebase."""
        src_path = Path("src/omnibase_infra")
        return [str(f) for f in src_path.rglob("*.py")]

    def test_file_patterns_match_at_least_one_file(
        self, exemptions_yaml: dict[str, Any], source_files: list[str]
    ) -> None:
        """Verify each file_pattern matches at least one file in the codebase.

        This catches patterns that reference files that have been renamed or deleted.
        """
        orphaned_patterns: list[tuple[str, str]] = []

        for section in [
            "pattern_exemptions",
            "architecture_exemptions",
            "union_exemptions",
        ]:
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

    def test_known_exemption_files_exist(self) -> None:
        """Verify key files that have exemptions actually exist."""
        key_files = [
            "src/omnibase_infra/event_bus/kafka_event_bus.py",
            "src/omnibase_infra/runtime/runtime_host_process.py",
            "src/omnibase_infra/runtime/message_dispatch_engine.py",
            "src/omnibase_infra/mixins/mixin_node_introspection.py",
            "src/omnibase_infra/validation/execution_shape_validator.py",
        ]
        for file_path in key_files:
            assert Path(file_path).exists(), (
                f"Expected file {file_path} to exist (has exemptions defined)"
            )
