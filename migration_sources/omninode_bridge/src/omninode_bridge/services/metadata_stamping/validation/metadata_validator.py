"""O.N.E. v0.1 metadata header validation.

This module provides comprehensive validation tools for ensuring metadata headers
are compliant, parseable, and functional according to O.N.E. v0.1 standards.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional, TypedDict

import yaml

logger = logging.getLogger(__name__)


# Type-safe metadata structures
class RuntimeConstraints(TypedDict, total=False):
    """Runtime constraints for tool execution."""

    sandboxed: bool
    privileged: bool
    requires_network: bool
    requires_gpu: bool


class DependencySpec(TypedDict, total=False):
    """Dependency specification."""

    name: str
    version: str
    optional: bool


class EnvironmentVariable(TypedDict):
    """Environment variable specification."""

    name: str
    required: bool
    description: str


class ONEMetadataDict(TypedDict, total=False):
    """Type-safe structure for O.N.E. v0.1 metadata.

    This TypedDict provides compile-time type checking for metadata structures
    while maintaining compatibility with dynamic YAML parsing.
    """

    # Required fields
    metadata_version: str
    name: str
    namespace: str
    version: str
    entrypoint: str
    protocols_supported: list[str]

    # Optional fields
    title: str
    category: str
    type: str
    role: str
    description: str
    tags: list[str]
    author: str
    license: str
    runtime_constraints: RuntimeConstraints
    dependencies: list[DependencySpec]
    environment: list[EnvironmentVariable]


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    level: ValidationLevel
    field: Optional[str]
    message: str
    line_number: Optional[int] = None


@dataclass
class FileValidationReport:
    """Complete validation report for a single file."""

    file_path: str
    is_valid: bool
    metadata_found: bool
    yaml_parseable: bool
    results: list[ValidationResult]
    extracted_metadata: Optional[ONEMetadataDict] = None

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.WARNING)


class ONEMetadataValidator:
    """Validates O.N.E. v0.1 tool metadata headers."""

    # Required fields according to O.N.E. v0.1 specification
    REQUIRED_FIELDS: ClassVar[list[str]] = [
        "metadata_version",
        "name",
        "namespace",
        "version",
        "entrypoint",
        "protocols_supported",
    ]

    # Expected field types for validation
    FIELD_TYPES: ClassVar[dict[str, type]] = {
        "metadata_version": str,
        "name": str,
        "title": str,
        "version": str,
        "namespace": str,
        "category": str,
        "type": str,
        "role": str,
        "description": str,
        "tags": list,
        "author": str,
        "license": str,
        "entrypoint": str,
        "protocols_supported": list,
        "runtime_constraints": dict,
        "dependencies": list,
        "environment": list,
    }

    # Valid namespace pattern for omninode services
    VALID_NAMESPACE_PATTERN = r"^omninode\.services\.[a-z][a-z0-9_]*$"

    # Expected metadata version
    EXPECTED_METADATA_VERSION = "0.1"

    def __init__(self, service_root: Optional[str] = None):
        """Initialize the validator.

        Args:
            service_root: Root directory of the service to validate.
                        If None, will try to auto-detect.
        """
        if service_root:
            self.service_root = Path(service_root)
        else:
            # Auto-detect service root
            current_file = Path(__file__)
            self.service_root = current_file.parent.parent

        logger.info(
            f"Initialized ONEMetadataValidator for service at: {self.service_root}"
        )

    def extract_metadata_header(
        self, file_path: str
    ) -> tuple[Optional[ONEMetadataDict], list[ValidationResult]]:
        """Extract and parse metadata header from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (extracted_metadata, validation_results)
        """
        results = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            results.append(
                ValidationResult(
                    level=ValidationLevel.ERROR,
                    field=None,
                    message=f"Could not read file: {e}",
                )
            )
            return None, results

        # Look for metadata header delimiters
        start_pattern = r"# === OmniNode:Tool_Metadata ==="
        end_pattern = r"# === /OmniNode:Tool_Metadata ==="

        start_match = re.search(start_pattern, content)
        end_match = re.search(end_pattern, content)

        if not start_match:
            results.append(
                ValidationResult(
                    level=ValidationLevel.ERROR,
                    field=None,
                    message="Missing metadata header start delimiter",
                )
            )
            return None, results

        if not end_match:
            results.append(
                ValidationResult(
                    level=ValidationLevel.ERROR,
                    field=None,
                    message="Missing metadata header end delimiter",
                )
            )
            return None, results

        # Extract metadata section
        header_content = content[start_match.end() : end_match.start()]

        # Remove comment prefixes and clean up
        yaml_content = ""
        for line in header_content.strip().split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith("#"):
                # Preserve indentation after removing # prefix
                yaml_line = stripped_line[1:]
                # Only strip leading space if there is one (to handle # vs #<space>)
                if yaml_line.startswith(" "):
                    yaml_line = yaml_line[1:]
                yaml_content += yaml_line + "\n"
            elif stripped_line == "":
                yaml_content += "\n"

        # Parse YAML
        try:
            metadata = yaml.safe_load(yaml_content.strip())
            if not isinstance(metadata, dict):
                results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        field=None,
                        message="Metadata is not a valid YAML dictionary",
                    )
                )
                return None, results
        except yaml.YAMLError as e:
            results.append(
                ValidationResult(
                    level=ValidationLevel.ERROR,
                    field=None,
                    message=f"Invalid YAML in metadata header: {e}",
                )
            )
            return None, results

        return metadata, results

    def validate_metadata_content(
        self, metadata: ONEMetadataDict
    ) -> list[ValidationResult]:
        """Validate the content of extracted metadata.

        Args:
            metadata: Extracted metadata dictionary (type-safe structure)

        Returns:
            List of validation results
        """
        results = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in metadata:
                results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        field=field,
                        message=f"Required field '{field}' is missing",
                    )
                )
            elif not metadata[field]:
                results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        field=field,
                        message=f"Required field '{field}' is empty",
                    )
                )

        # Validate field types
        for field, expected_type in self.FIELD_TYPES.items():
            if field in metadata:
                actual_value = metadata[field]
                # Special handling for metadata_version - allow float that converts to valid string
                if field == "metadata_version" and isinstance(
                    actual_value, int | float
                ):
                    continue  # This will be validated in the specific validation section
                elif not isinstance(actual_value, expected_type):
                    results.append(
                        ValidationResult(
                            level=ValidationLevel.ERROR,
                            field=field,
                            message=f"Field '{field}' should be {expected_type.__name__}, got {type(actual_value).__name__}",
                        )
                    )

        # Validate specific field values

        # Metadata version
        if "metadata_version" in metadata:
            # Convert to string if it's a number (YAML might parse 0.1 as float)
            version_str = str(metadata["metadata_version"])
            if version_str != self.EXPECTED_METADATA_VERSION:
                results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        field="metadata_version",
                        message=f"Expected metadata_version '{self.EXPECTED_METADATA_VERSION}', got '{version_str}'",
                    )
                )

        # Namespace validation
        if "namespace" in metadata:
            namespace = metadata["namespace"]
            if not re.match(self.VALID_NAMESPACE_PATTERN, namespace):
                results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        field="namespace",
                        message=f"Invalid namespace format: '{namespace}'. Expected pattern: 'omninode.services.*'",
                    )
                )

        # Protocols supported
        if "protocols_supported" in metadata:
            protocols = metadata["protocols_supported"]
            if isinstance(protocols, list):
                if "O.N.E. v0.1" not in protocols:
                    results.append(
                        ValidationResult(
                            level=ValidationLevel.ERROR,
                            field="protocols_supported",
                            message="Must include 'O.N.E. v0.1' in protocols_supported",
                        )
                    )
            else:
                results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        field="protocols_supported",
                        message="protocols_supported must be a list",
                    )
                )

        # Version format validation (semantic versioning)
        if "version" in metadata:
            version = metadata["version"]
            version_pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$"
            if not re.match(version_pattern, version):
                results.append(
                    ValidationResult(
                        level=ValidationLevel.WARNING,
                        field="version",
                        message=f"Version '{version}' does not follow semantic versioning (x.y.z)",
                    )
                )

        # Runtime constraints validation
        if "runtime_constraints" in metadata:
            constraints = metadata["runtime_constraints"]
            if isinstance(constraints, dict):
                expected_keys = {
                    "sandboxed",
                    "privileged",
                    "requires_network",
                    "requires_gpu",
                }
                for key in expected_keys:
                    if key not in constraints:
                        results.append(
                            ValidationResult(
                                level=ValidationLevel.WARNING,
                                field="runtime_constraints",
                                message=f"Missing recommended runtime constraint: '{key}'",
                            )
                        )
                    elif not isinstance(constraints[key], bool):
                        results.append(
                            ValidationResult(
                                level=ValidationLevel.ERROR,
                                field="runtime_constraints",
                                message=f"Runtime constraint '{key}' must be boolean",
                            )
                        )

        # Dependencies validation
        if "dependencies" in metadata:
            dependencies = metadata["dependencies"]
            if isinstance(dependencies, list):
                for i, dep in enumerate(dependencies):
                    if isinstance(dep, dict):
                        if "name" not in dep:
                            results.append(
                                ValidationResult(
                                    level=ValidationLevel.ERROR,
                                    field="dependencies",
                                    message=f"Dependency {i} missing 'name' field",
                                )
                            )
                        if "version" not in dep:
                            results.append(
                                ValidationResult(
                                    level=ValidationLevel.WARNING,
                                    field="dependencies",
                                    message=f"Dependency {i} missing 'version' field",
                                )
                            )

        return results

    def validate_file(self, file_path: str) -> FileValidationReport:
        """Validate metadata header in a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Complete validation report for the file
        """
        file_path = str(file_path)
        results = []

        # Check if file exists and is Python file
        if not Path(file_path).exists():
            results.append(
                ValidationResult(
                    level=ValidationLevel.ERROR,
                    field=None,
                    message="File does not exist",
                )
            )
            return FileValidationReport(
                file_path=file_path,
                is_valid=False,
                metadata_found=False,
                yaml_parseable=False,
                results=results,
            )

        if not file_path.endswith(".py"):
            results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    field=None,
                    message="File is not a Python file",
                )
            )

        # Extract metadata
        metadata, extraction_results = self.extract_metadata_header(file_path)
        results.extend(extraction_results)

        metadata_found = metadata is not None
        yaml_parseable = metadata is not None

        # Validate metadata content if successfully extracted
        if metadata:
            content_results = self.validate_metadata_content(metadata)
            results.extend(content_results)

        # Determine overall validity
        is_valid = all(r.level != ValidationLevel.ERROR for r in results)

        return FileValidationReport(
            file_path=file_path,
            is_valid=is_valid,
            metadata_found=metadata_found,
            yaml_parseable=yaml_parseable,
            results=results,
            extracted_metadata=metadata,
        )

    def get_service_python_files(self) -> list[str]:
        """Get list of core Python files in the service that should have metadata.

        Returns:
            List of file paths that should contain metadata headers
        """
        core_files = [
            "main.py",
            "service.py",
            "api/router.py",
            "engine/stamping_engine.py",
            "engine/hash_generator.py",
        ]

        service_files = []
        for file_pattern in core_files:
            file_path = self.service_root / file_pattern
            if file_path.exists():
                service_files.append(str(file_path))

        return service_files

    def validate_all_service_files(self) -> dict[str, Any]:
        """Validate all service files for metadata compliance.

        Returns:
            Complete validation report for all service files
        """
        logger.info("Starting validation of all service files")

        service_files = self.get_service_python_files()
        file_reports = {}

        for file_path in service_files:
            logger.debug(f"Validating: {file_path}")
            report = self.validate_file(file_path)
            file_reports[file_path] = report

        # Generate summary
        total_files = len(file_reports)
        valid_files = sum(1 for r in file_reports.values() if r.is_valid)
        files_with_metadata = sum(1 for r in file_reports.values() if r.metadata_found)
        total_errors = sum(r.error_count for r in file_reports.values())
        total_warnings = sum(r.warning_count for r in file_reports.values())

        # Check namespace consistency
        namespaces = set()
        for report in file_reports.values():
            if report.extracted_metadata and "namespace" in report.extracted_metadata:
                namespaces.add(report.extracted_metadata["namespace"])

        namespace_consistent = len(namespaces) <= 1

        # Check version consistency
        versions = set()
        for report in file_reports.values():
            if report.extracted_metadata and "version" in report.extracted_metadata:
                versions.add(report.extracted_metadata["version"])

        version_consistent = len(versions) <= 1

        summary = {
            "overall_valid": total_errors == 0,
            "total_files": total_files,
            "valid_files": valid_files,
            "files_with_metadata": files_with_metadata,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "namespace_consistent": namespace_consistent,
            "version_consistent": version_consistent,
            "namespaces_found": list(namespaces),
            "versions_found": list(versions),
            "compliance_percentage": (
                (valid_files / total_files * 100) if total_files > 0 else 0
            ),
            "metadata_coverage": (
                (files_with_metadata / total_files * 100) if total_files > 0 else 0
            ),
        }

        return {
            "summary": summary,
            "file_reports": file_reports,
            "validation_timestamp": str(Path(__file__).stat().st_mtime),
        }

    def generate_report_text(self, validation_report: dict[str, Any]) -> str:
        """Generate a human-readable text report.

        Args:
            validation_report: Report from validate_all_service_files()

        Returns:
            Formatted text report
        """
        summary = validation_report["summary"]
        file_reports = validation_report["file_reports"]

        report_lines = [
            "O.N.E. v0.1 Metadata Validation Report",
            "=" * 40,
            "",
            f"Overall Status: {'✅ PASSED' if summary['overall_valid'] else '❌ FAILED'}",
            f"Compliance: {summary['compliance_percentage']:.1f}%",
            f"Metadata Coverage: {summary['metadata_coverage']:.1f}%",
            "",
            "Summary:",
            f"  Total Files: {summary['total_files']}",
            f"  Valid Files: {summary['valid_files']}",
            f"  Files with Metadata: {summary['files_with_metadata']}",
            f"  Total Errors: {summary['total_errors']}",
            f"  Total Warnings: {summary['total_warnings']}",
            "",
            f"Namespace Consistency: {'✅' if summary['namespace_consistent'] else '❌'}",
            f"Version Consistency: {'✅' if summary['version_consistent'] else '❌'}",
            "",
        ]

        if summary["namespaces_found"]:
            report_lines.append(f"Namespaces: {', '.join(summary['namespaces_found'])}")

        if summary["versions_found"]:
            report_lines.append(f"Versions: {', '.join(summary['versions_found'])}")

        report_lines.extend(["", "File Details:", "-" * 20])

        for file_path, report in file_reports.items():
            relative_path = Path(file_path).relative_to(self.service_root)
            status = "✅" if report.is_valid else "❌"
            report_lines.append(f"{status} {relative_path}")

            if report.error_count > 0:
                report_lines.append(f"   Errors: {report.error_count}")
                for result in report.results:
                    if result.level == ValidationLevel.ERROR:
                        field_info = f" ({result.field})" if result.field else ""
                        report_lines.append(f"     - {result.message}{field_info}")

            if report.warning_count > 0:
                report_lines.append(f"   Warnings: {report.warning_count}")
                for result in report.results:
                    if result.level == ValidationLevel.WARNING:
                        field_info = f" ({result.field})" if result.field else ""
                        report_lines.append(f"     - {result.message}{field_info}")

            report_lines.append("")

        return "\n".join(report_lines)
