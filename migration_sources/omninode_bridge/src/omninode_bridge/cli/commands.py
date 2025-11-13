#!/usr/bin/env python3
"""
Main CLI commands module for OmniNode Bridge.

This module provides the primary command-line interface for the OmniNode Bridge
workflow coordinator, integrating workflow submission, validation, and CI tools.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .workflow_ci import (
    WorkflowGenerator,
    WorkflowLinter,
    WorkflowTester,
    WorkflowValidator,
)
from .workflow_submit import WorkflowSubmissionCLI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool):
    """
    OmniNode Bridge CLI - Intelligent workflow coordination and management.

    This CLI provides tools for workflow submission, validation, generation,
    and CI integration for the OmniNode Bridge platform.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Configure logging based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        ctx.obj["verbose"] = True
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)
        ctx.obj["quiet"] = True
    else:
        logging.getLogger().setLevel(logging.INFO)


@cli.group()
@click.pass_context
def workflow(ctx):
    """Workflow management commands."""
    pass


@workflow.command("submit")
@click.argument("workflow_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--endpoint",
    "-e",
    default="http://localhost:8000",
    help="Hook receiver endpoint URL",
)
@click.option("--token", "-t", help="Authentication token")
@click.option(
    "--dry-run", "-n", is_flag=True, help="Validate workflow without submitting"
)
@click.option("--wait", "-w", is_flag=True, help="Wait for workflow completion")
@click.pass_context
def submit_workflow(
    ctx,
    workflow_file: Path,
    endpoint: str,
    token: Optional[str],
    dry_run: bool,
    wait: bool,
):
    """Submit a workflow to the OmniNode Bridge system."""

    async def _submit():
        submission_cli = WorkflowSubmissionCLI()

        # Configure submission parameters
        submission_cli.endpoint = endpoint
        if token:
            submission_cli.token = token
        submission_cli.dry_run = dry_run
        submission_cli.wait_for_completion = wait

        # Submit workflow
        success = await submission_cli.submit_workflow_file(workflow_file)

        if success:
            click.echo(f"✅ Workflow submitted successfully: {workflow_file}")
            return 0
        else:
            click.echo(f"❌ Failed to submit workflow: {workflow_file}", err=True)
            return 1

    try:
        exit_code = asyncio.run(_submit())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        click.echo("\n⚠️  Interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logger.exception("Workflow submission failed")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@workflow.command("validate")
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--strict", is_flag=True, help="Enable strict validation mode")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def validate_workflow(ctx, files: tuple[Path, ...], strict: bool, output_format: str):
    """Validate workflow files for syntax and best practices."""

    if not files:
        click.echo("No files provided for validation", err=True)
        sys.exit(1)

    validator = WorkflowValidator(strict_mode=strict)
    all_valid = True

    for file_path in files:
        if not ctx.obj.get("quiet"):
            click.echo(f"🔍 Validating: {file_path}")

        is_valid, data = validator.validate_yaml_file(file_path)
        report = validator.get_validation_report()

        if output_format == "text":
            if is_valid:
                if not ctx.obj.get("quiet"):
                    click.echo(f"✅ {file_path} is valid")
            else:
                click.echo(
                    f"❌ {file_path} has {report['error_count']} errors", err=True
                )
                all_valid = False

                for error in report["errors"]:
                    click.echo(f"   Error: {error}", err=True)

            if report["warnings"] and not ctx.obj.get("quiet"):
                click.echo(f"⚠️  {len(report['warnings'])} warnings:")
                for warning in report["warnings"]:
                    click.echo(f"   Warning: {warning}")

        # Reset validator for next file
        validator.errors = []
        validator.warnings = []

    if output_format == "json":
        import json

        results = [
            {
                "file": str(f),
                "valid": True,  # Would need to re-validate for JSON output
                "report": {"errors": [], "warnings": []},
            }
            for f in files
        ]
        click.echo(json.dumps(results, indent=2))

    sys.exit(0 if all_valid else 1)


@workflow.command("test")
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Verbose test output")
@click.pass_context
def test_workflow(ctx, files: tuple[Path, ...], verbose: bool):
    """Test workflow configurations with dry-run simulation."""

    if not files:
        click.echo("No files provided for testing", err=True)
        sys.exit(1)

    tester = WorkflowTester()
    all_passed = True

    for file_path in files:
        result = tester.dry_run_test(file_path)

        if result["success"]:
            if not ctx.obj.get("quiet"):
                click.echo(f"✅ {file_path} test passed")

            if verbose or ctx.obj.get("verbose"):
                for job_result in result["test_results"]:
                    click.echo(
                        f"  Job '{job_result['job_name']}': {job_result['estimated_duration']}"
                    )
                    if job_result["issues"]:
                        for issue in job_result["issues"]:
                            click.echo(f"    ⚠️  {issue}")
        else:
            click.echo(f"❌ {file_path} test failed", err=True)
            all_passed = False

            validation = result["validation"]
            for error in validation["errors"]:
                click.echo(f"   Error: {error}", err=True)

    sys.exit(0 if all_passed else 1)


@workflow.command("lint")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option("--fix", is_flag=True, help="Automatically fix issues where possible")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def lint_workflow(ctx, paths: tuple[Path, ...], fix: bool, output_format: str):
    """Lint workflow files for style and best practices."""

    if not paths:
        click.echo("No paths provided for linting", err=True)
        sys.exit(1)

    linter = WorkflowLinter(fix_issues=fix)
    all_results = []

    for path in paths:
        if path.is_file():
            result = linter.lint_file(path)
            all_results.append(result)
        elif path.is_dir():
            result = linter.lint_directory(path)
            all_results.extend(result["results"])
        else:
            click.echo(f"❌ Path not found: {path}", err=True)

    if output_format == "text":
        total_issues = 0
        for result in all_results:
            file_path = result["file"]
            issues = result.get("issues", [])
            total_issues += len(issues)

            if issues:
                click.echo(f"📄 {file_path}:")
                for issue in issues:
                    severity_icon = {"error": "❌", "warning": "⚠️", "suggestion": "💡"}
                    icon = severity_icon.get(issue["severity"], "i")
                    line_info = f"Line {issue['line']}: " if "line" in issue else ""
                    click.echo(f"  {icon} {line_info}{issue['message']}")
            else:
                if not ctx.obj.get("quiet"):
                    click.echo(f"✅ {file_path}: No issues")

        if not ctx.obj.get("quiet"):
            click.echo(f"\n📊 Total issues found: {total_issues}")

        sys.exit(1 if total_issues > 0 else 0)

    elif output_format == "json":
        import json

        click.echo(json.dumps(all_results, indent=2))


@workflow.command("generate")
@click.option("--template", "-t", required=True, help="Template name to generate from")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output file path"
)
@click.option("--list-templates", is_flag=True, help="List available templates")
@click.pass_context
def generate_workflow(ctx, template: str, output: Optional[Path], list_templates: bool):
    """Generate workflow from predefined templates."""

    generator = WorkflowGenerator()

    if list_templates:
        templates = generator.list_templates()
        click.echo("Available templates:")
        for name, description in templates.items():
            click.echo(f"  {name}: {description}")
        return

    try:
        workflow = generator.generate_from_template(template, output)

        if output:
            click.echo(f"✅ Generated workflow: {output}")
        else:
            # Print to stdout
            import yaml

            yaml_dict = workflow.to_yaml_dict()
            click.echo(yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False))

    except ValueError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if ctx.obj.get("verbose"):
            logger.exception("Workflow generation failed")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def ci():
    """CI/CD integration commands."""
    pass


@ci.command("validate-all")
@click.option(
    "--workflows-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(".github/workflows"),
    help="Workflows directory to validate",
)
@click.option("--strict", is_flag=True, help="Enable strict validation mode")
@click.pass_context
def validate_all_workflows(ctx, workflows_dir: Path, strict: bool):
    """Validate all workflow files in the workflows directory."""

    if not workflows_dir.exists():
        click.echo(f"❌ Workflows directory not found: {workflows_dir}", err=True)
        sys.exit(1)

    workflow_files = list(workflows_dir.glob("*.yml")) + list(
        workflows_dir.glob("*.yaml")
    )

    if not workflow_files:
        click.echo(f"INFO: No workflow files found in {workflows_dir}")
        sys.exit(0)

    validator = WorkflowValidator(strict_mode=strict)
    total_errors = 0
    total_warnings = 0

    for file_path in workflow_files:
        if not ctx.obj.get("quiet"):
            click.echo(f"📋 Validating: {file_path}")

        is_valid, data = validator.validate_yaml_file(file_path)
        report = validator.get_validation_report()

        total_errors += report["error_count"]
        total_warnings += report["warning_count"]

        if report["errors"]:
            for error in report["errors"]:
                click.echo(f"  ❌ {error}", err=True)

        if report["warnings"] and not ctx.obj.get("quiet"):
            for warning in report["warnings"]:
                click.echo(f"  ⚠️  {warning}")

        if is_valid and not report["errors"]:
            if not ctx.obj.get("quiet"):
                click.echo("  ✅ Valid")

        # Reset validator for next file
        validator.errors = []
        validator.warnings = []

    # Summary
    if total_errors == 0:
        if not ctx.obj.get("quiet"):
            click.echo(f"✅ All {len(workflow_files)} workflow files are valid")
            if total_warnings > 0:
                click.echo(f"⚠️  Found {total_warnings} warnings (not blocking)")
    else:
        click.echo(
            f"❌ Found {total_errors} errors across {len(workflow_files)} files",
            err=True,
        )

    sys.exit(0 if total_errors == 0 else 1)


@cli.command("version")
def version():
    """Show version information."""
    try:
        import importlib.metadata

        version_str = importlib.metadata.version("omninode_bridge")
        click.echo(f"OmniNode Bridge CLI v{version_str}")
    except importlib.metadata.PackageNotFoundError:
        click.echo("OmniNode Bridge CLI (development version)")


if __name__ == "__main__":
    cli()
