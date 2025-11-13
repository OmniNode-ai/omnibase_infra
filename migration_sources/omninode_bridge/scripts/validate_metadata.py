#!/usr/bin/env python3
"""CLI tool for validating O.N.E. v0.1 metadata compliance.

This script provides command-line validation of metadata headers in service files
for compliance with O.N.E. v0.1 standards.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the validator directly to avoid dependency issues
validator_path = (
    src_path / "omninode_bridge" / "services" / "metadata_stamping" / "validation"
)
sys.path.insert(0, str(validator_path))

from metadata_validator import ONEMetadataValidator, ValidationLevel


@click.command()
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="Validate specific file",
)
@click.option("--all", "-a", is_flag=True, help="Validate all service files")
@click.option(
    "--service-root",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    help="Root directory of the service (auto-detected if not specified)",
)
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format for results",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat warnings as errors (exit with non-zero code on warnings)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Only show errors and warnings, suppress info messages",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed validation information"
)
def validate_metadata(
    file: Optional[Path],
    all: bool,
    service_root: Optional[Path],
    output_format: str,
    strict: bool,
    quiet: bool,
    verbose: bool,
):
    """Validate O.N.E. v0.1 metadata headers in Python files.

    This tool validates metadata headers in MetadataStampingService files
    to ensure compliance with O.N.E. v0.1 standards.

    Examples:
    \b
        # Validate all service files
        python validate_metadata.py --all

        # Validate specific file
        python validate_metadata.py --file main.py

        # Validate with JSON output
        python validate_metadata.py --all --output-format json

        # Strict mode (treat warnings as errors)
        python validate_metadata.py --all --strict
    """
    if not file and not all:
        click.echo("Error: Must specify either --file or --all", err=True)
        sys.exit(1)

    if file and all:
        click.echo("Error: Cannot specify both --file and --all", err=True)
        sys.exit(1)

    try:
        # Initialize validator
        if service_root:
            validator = ONEMetadataValidator(str(service_root))
        else:
            # Auto-detect service root
            default_service_root = (
                Path(__file__).parent.parent
                / "src"
                / "omninode_bridge"
                / "services"
                / "metadata_stamping"
            )
            if default_service_root.exists():
                validator = ONEMetadataValidator(str(default_service_root))
            else:
                click.echo(
                    "Error: Could not auto-detect service root. Please specify --service-root",
                    err=True,
                )
                sys.exit(1)

        if not quiet:
            click.echo(f"🔍 Using service root: {validator.service_root}")

        # Perform validation
        if file:
            # Validate single file
            if not quiet:
                click.echo(f"📁 Validating file: {file}")

            report = validator.validate_file(str(file))

            if output_format == "json":
                output_single_file_json(report)
            else:
                output_single_file_text(report, verbose, quiet)

            # Exit with appropriate code
            exit_code = determine_exit_code(report.results, strict)
            sys.exit(exit_code)

        else:
            # Validate all service files
            if not quiet:
                click.echo("📊 Validating all service files...")

            validation_report = validator.validate_all_service_files()

            if output_format == "json":
                output_all_files_json(validation_report)
            else:
                output_all_files_text(validation_report, validator, verbose, quiet)

            # Exit with appropriate code
            summary = validation_report["summary"]
            has_errors = summary["total_errors"] > 0
            has_warnings = summary["total_warnings"] > 0

            if has_errors or (strict and has_warnings):
                sys.exit(1)
            else:
                sys.exit(0)

    except Exception as e:
        click.echo(f"❌ Validation failed with error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def output_single_file_text(report, verbose: bool, quiet: bool):
    """Output single file validation results in text format."""
    if not quiet:
        click.echo("\n" + "=" * 50)
        click.echo(f"📄 File: {Path(report.file_path).name}")
        click.echo("=" * 50)

    # Status summary
    status_icon = "✅" if report.is_valid else "❌"
    status_text = "VALID" if report.is_valid else "INVALID"

    click.echo(f"{status_icon} Status: {status_text}")

    if not quiet:
        click.echo(f"📋 Metadata Found: {'✅' if report.metadata_found else '❌'}")
        click.echo(f"📝 YAML Parseable: {'✅' if report.yaml_parseable else '❌'}")
        click.echo(f"🚨 Errors: {report.error_count}")
        click.echo(f"⚠️  Warnings: {report.warning_count}")

    # Show results
    if report.results:
        if not quiet:
            click.echo("\n📋 Validation Results:")

        for result in report.results:
            icon = get_result_icon(result.level)
            field_info = f" [{result.field}]" if result.field else ""

            if result.level == ValidationLevel.ERROR or verbose or not quiet:
                click.echo(f"  {icon} {result.message}{field_info}")

    # Show extracted metadata in verbose mode
    if verbose and report.extracted_metadata:
        click.echo("\n📊 Extracted Metadata:")
        for key, value in report.extracted_metadata.items():
            click.echo(f"  {key}: {value}")


def output_all_files_text(validation_report, validator, verbose: bool, quiet: bool):
    """Output all files validation results in text format."""
    summary = validation_report["summary"]
    file_reports = validation_report["file_reports"]

    if not quiet:
        # Generate and display the comprehensive report
        report_text = validator.generate_report_text(validation_report)
        click.echo(report_text)
    else:
        # Minimal output for quiet mode
        status_icon = "✅" if summary["overall_valid"] else "❌"
        status_text = "PASSED" if summary["overall_valid"] else "FAILED"
        click.echo(f"{status_icon} Overall Status: {status_text}")

        if summary["total_errors"] > 0:
            click.echo(f"🚨 Total Errors: {summary['total_errors']}")

        if summary["total_warnings"] > 0:
            click.echo(f"⚠️  Total Warnings: {summary['total_warnings']}")

        # Show error details even in quiet mode
        for file_path, report in file_reports.items():
            if report.error_count > 0:
                relative_path = Path(file_path).name
                click.echo(f"❌ {relative_path}: {report.error_count} errors")

                for result in report.results:
                    if result.level == ValidationLevel.ERROR:
                        field_info = f" [{result.field}]" if result.field else ""
                        click.echo(f"   - {result.message}{field_info}")


def output_single_file_json(report):
    """Output single file validation results in JSON format."""
    json_data = {
        "file_path": report.file_path,
        "is_valid": report.is_valid,
        "metadata_found": report.metadata_found,
        "yaml_parseable": report.yaml_parseable,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "results": [
            {
                "level": result.level.value,
                "field": result.field,
                "message": result.message,
                "line_number": result.line_number,
            }
            for result in report.results
        ],
        "extracted_metadata": report.extracted_metadata,
    }

    click.echo(json.dumps(json_data, indent=2))


def output_all_files_json(validation_report):
    """Output all files validation results in JSON format."""
    # Convert file reports to JSON-serializable format
    file_reports_json = {}
    for file_path, report in validation_report["file_reports"].items():
        file_reports_json[file_path] = {
            "file_path": report.file_path,
            "is_valid": report.is_valid,
            "metadata_found": report.metadata_found,
            "yaml_parseable": report.yaml_parseable,
            "error_count": report.error_count,
            "warning_count": report.warning_count,
            "results": [
                {
                    "level": result.level.value,
                    "field": result.field,
                    "message": result.message,
                    "line_number": result.line_number,
                }
                for result in report.results
            ],
            "extracted_metadata": report.extracted_metadata,
        }

    json_data = {
        "summary": validation_report["summary"],
        "file_reports": file_reports_json,
        "validation_timestamp": validation_report["validation_timestamp"],
    }

    click.echo(json.dumps(json_data, indent=2))


def get_result_icon(level: ValidationLevel) -> str:
    """Get icon for validation result level."""
    icons = {
        ValidationLevel.ERROR: "🚨",
        ValidationLevel.WARNING: "⚠️",
        ValidationLevel.INFO: "INFO",
    }
    return icons.get(level, "❓")


def determine_exit_code(results, strict: bool) -> int:
    """Determine appropriate exit code based on validation results."""
    has_errors = any(r.level == ValidationLevel.ERROR for r in results)
    has_warnings = any(r.level == ValidationLevel.WARNING for r in results)

    if has_errors or (strict and has_warnings):
        return 1
    else:
        return 0


if __name__ == "__main__":
    validate_metadata()
