#!/usr/bin/env python3
"""
CI-specific CLI for OmniNode Bridge.

This module provides specialized command-line tools for CI/CD environments,
including the Pydantic CI workflow validation system and pre-commit hooks.
"""

import asyncio
import sys
from pathlib import Path

import click

# Import the workflow CI tools
from ...cli.workflow_ci import main as workflow_ci_main


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool):
    """
    OmniNode Bridge CI Tools - Specialized tools for CI/CD environments.

    This CLI provides validation, linting, and testing tools specifically
    designed for use in continuous integration environments.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store options for sub-commands
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command("validate")
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
def validate(ctx, files: tuple[Path, ...], strict: bool, output_format: str):
    """Validate workflow files for syntax and best practices."""

    # Prepare arguments for the workflow_ci main function
    args = ["validate"]
    for file_path in files:
        args.append(str(file_path))

    if strict:
        args.append("--strict")

    if output_format != "text":
        args.extend(["--format", output_format])

    # Override sys.argv to pass arguments to workflow_ci main
    original_argv = sys.argv
    try:
        sys.argv = ["workflow_ci"] + args
        asyncio.run(workflow_ci_main())
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        sys.argv = original_argv


@cli.command("test")
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Verbose test output")
@click.pass_context
def test(ctx, files: tuple[Path, ...], verbose: bool):
    """Test workflow configurations with dry-run simulation."""

    # Prepare arguments for the workflow_ci main function
    args = ["test"]
    for file_path in files:
        args.append(str(file_path))

    if verbose or ctx.obj.get("verbose"):
        args.append("--verbose")

    # Override sys.argv to pass arguments to workflow_ci main
    original_argv = sys.argv
    try:
        sys.argv = ["workflow_ci"] + args
        asyncio.run(workflow_ci_main())
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        sys.argv = original_argv


@cli.command("lint")
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
def lint(ctx, paths: tuple[Path, ...], fix: bool, output_format: str):
    """Lint workflow files for style and best practices."""

    # Prepare arguments for the workflow_ci main function
    args = ["lint"]
    for path in paths:
        args.append(str(path))

    if fix:
        args.append("--fix")

    if output_format != "text":
        args.extend(["--format", output_format])

    # Override sys.argv to pass arguments to workflow_ci main
    original_argv = sys.argv
    try:
        sys.argv = ["workflow_ci"] + args
        asyncio.run(workflow_ci_main())
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        sys.argv = original_argv


@cli.command("generate")
@click.option("--template", "-t", required=True, help="Template name to generate from")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output file path"
)
@click.option("--list-templates", is_flag=True, help="List available templates")
@click.pass_context
def generate(ctx, template: str, output: Path, list_templates: bool):
    """Generate workflow from predefined templates."""

    # Prepare arguments for the workflow_ci main function
    args = ["generate", "--template", template]

    if output:
        args.extend(["--output", str(output)])

    if list_templates:
        args.append("--list-templates")

    # Override sys.argv to pass arguments to workflow_ci main
    original_argv = sys.argv
    try:
        sys.argv = ["workflow_ci"] + args
        asyncio.run(workflow_ci_main())
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        sys.argv = original_argv


@cli.command("templates")
@click.pass_context
def templates(ctx):
    """List available workflow templates."""

    # Prepare arguments for the workflow_ci main function
    args = ["templates"]

    # Override sys.argv to pass arguments to workflow_ci main
    original_argv = sys.argv
    try:
        sys.argv = ["workflow_ci"] + args
        asyncio.run(workflow_ci_main())
    except SystemExit as e:
        sys.exit(e.code)
    finally:
        sys.argv = original_argv


@cli.command("pre-commit-validate")
@click.argument("files", nargs=-1, type=click.Path())
@click.option("--strict", is_flag=True, help="Enable strict validation mode")
@click.option("--no-lint", is_flag=True, help="Disable linting checks")
@click.option(
    "--all-workflows",
    is_flag=True,
    help="Validate all workflows in .github/workflows directory",
)
@click.pass_context
def pre_commit_validate(
    ctx, files: tuple[str, ...], strict: bool, no_lint: bool, all_workflows: bool
):
    """
    Pre-commit hook for workflow validation.

    This command is designed to be used as a pre-commit hook to validate
    GitHub Actions workflow files before they are committed.
    """

    # Secure import: Use proper Python package structure instead of path manipulation
    # Scripts should be properly packaged or accessible via PYTHONPATH

    try:
        from pre_commit_workflow_validation import PreCommitWorkflowValidator
    except ImportError as e:
        click.echo(f"❌ Failed to import workflow validation tools: {e}", err=True)
        click.echo(
            "Make sure you're running this from the repository root with dependencies installed.",
            err=True,
        )
        sys.exit(2)

    try:
        # Initialize validator with options
        validator = PreCommitWorkflowValidator(
            strict_mode=strict, enable_linting=not no_lint
        )

        # Validate workflows
        if all_workflows:
            success = validator.validate_workflow_directory()
        elif files:
            success = validator.validate_files(list(files))
        else:
            # Default: validate all workflows in standard directory
            success = validator.validate_workflow_directory()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        click.echo("\n⚠️  Interrupted by user", err=True)
        sys.exit(2)
    except Exception as e:
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        click.echo(f"❌ Script error: {e}", err=True)
        sys.exit(2)


@cli.command("version")
def version():
    """Show version information."""
    try:
        import importlib.metadata

        version_str = importlib.metadata.version("omninode_bridge")
        click.echo(f"OmniNode Bridge CI Tools v{version_str}")
    except importlib.metadata.PackageNotFoundError:
        click.echo("OmniNode Bridge CI Tools (development version)")


if __name__ == "__main__":
    cli()
