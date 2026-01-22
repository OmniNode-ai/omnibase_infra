"""
ONEX Infrastructure CLI Commands.

Provides CLI interface for infrastructure management and validation.
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli() -> None:
    """ONEX Infrastructure CLI."""


@cli.group()
def validate() -> None:
    """Validation commands for infrastructure code."""


@validate.command("architecture")
@click.argument("directory", default="src/omnibase_infra/")
@click.option(
    "--max-violations",
    default=None,
    help="Maximum allowed violations (default: INFRA_MAX_VIOLATIONS)",
)
def validate_architecture_cmd(directory: str, max_violations: int | None) -> None:
    """Validate architecture (one-model-per-file)."""
    from omnibase_infra.validation.infra_validators import (
        INFRA_MAX_VIOLATIONS,
        validate_infra_architecture,
    )

    console.print(f"[bold blue]Validating architecture in {directory}...[/bold blue]")
    # Use INFRA_MAX_VIOLATIONS constant if no override provided
    effective_max_violations = (
        max_violations if max_violations is not None else INFRA_MAX_VIOLATIONS
    )
    result = validate_infra_architecture(directory, effective_max_violations)
    _print_result("Architecture", result)
    raise SystemExit(0 if result.is_valid else 1)


@validate.command("contracts")
@click.argument("directory", default="src/omnibase_infra/nodes/")
def validate_contracts_cmd(directory: str) -> None:
    """Validate YAML contracts."""
    from omnibase_infra.validation.infra_validators import validate_infra_contracts

    console.print(f"[bold blue]Validating contracts in {directory}...[/bold blue]")
    result = validate_infra_contracts(directory)
    _print_result("Contracts", result)
    raise SystemExit(0 if result.is_valid else 1)


@validate.command("patterns")
@click.argument("directory", default="src/omnibase_infra/")
@click.option(
    "--strict/--no-strict",
    default=None,
    help="Enable strict mode (default: INFRA_PATTERNS_STRICT)",
)
def validate_patterns_cmd(directory: str, strict: bool | None) -> None:
    """Validate code patterns and naming conventions."""
    from omnibase_infra.validation.infra_validators import (
        INFRA_PATTERNS_STRICT,
        validate_infra_patterns,
    )

    console.print(f"[bold blue]Validating patterns in {directory}...[/bold blue]")
    # Use INFRA_PATTERNS_STRICT constant if no override provided
    effective_strict = strict if strict is not None else INFRA_PATTERNS_STRICT
    result = validate_infra_patterns(directory, effective_strict)
    _print_result("Patterns", result)
    raise SystemExit(0 if result.is_valid else 1)


@validate.command("unions")
@click.argument("directory", default="src/omnibase_infra/")
@click.option(
    "--max-unions",
    default=None,
    help="Maximum allowed union count (default: INFRA_MAX_UNIONS)",
)
@click.option(
    "--strict/--no-strict",
    default=None,
    help="Enable strict mode (default: INFRA_UNIONS_STRICT)",
)
def validate_unions_cmd(
    directory: str, max_unions: int | None, strict: bool | None
) -> None:
    """Validate Union type usage.

    Counts total unions in the codebase.
    Valid `X | None` patterns are counted but not flagged as violations.
    """
    from omnibase_infra.validation.infra_validators import (
        INFRA_MAX_UNIONS,
        INFRA_UNIONS_STRICT,
        validate_infra_union_usage,
    )

    console.print(f"[bold blue]Validating union usage in {directory}...[/bold blue]")
    # Use constants if no override provided
    effective_max_unions = max_unions if max_unions is not None else INFRA_MAX_UNIONS
    effective_strict = strict if strict is not None else INFRA_UNIONS_STRICT
    result = validate_infra_union_usage(
        directory, effective_max_unions, effective_strict
    )
    _print_result("Union Usage", result)
    raise SystemExit(0 if result.is_valid else 1)


@validate.command("imports")
@click.argument("directory", default="src/omnibase_infra/")
def validate_imports_cmd(directory: str) -> None:
    """Check for circular imports."""
    from omnibase_infra.validation.infra_validators import (
        validate_infra_circular_imports,
    )

    console.print(f"[bold blue]Checking circular imports in {directory}...[/bold blue]")
    result = validate_infra_circular_imports(directory)

    # ModelImportValidationResult uses has_circular_imports property (plural)
    if not result.has_circular_imports:
        console.print("[bold green]Circular Imports: PASS[/bold green]")
        raise SystemExit(0)
    console.print("[bold red]Circular Imports: FAIL[/bold red]")
    if hasattr(result, "cycles") and result.cycles:
        for cycle in result.cycles:
            console.print(f"  [red]Cycle: {cycle}[/red]")
    if hasattr(result, "errors") and result.errors:
        for error in result.errors:
            console.print(f"  [red]{error}[/red]")
    raise SystemExit(1)


@validate.command("all")
@click.argument("directory", default="src/omnibase_infra/")
@click.option(
    "--nodes-dir", default="src/omnibase_infra/nodes/", help="Nodes directory"
)
def validate_all_cmd(directory: str, nodes_dir: str) -> None:
    """Run all validations."""
    from omnibase_infra.validation.infra_validators import (
        get_validation_summary,
        validate_infra_all,
    )

    console.print(f"[bold blue]Running all validations on {directory}...[/bold blue]\n")
    results = validate_infra_all(directory, nodes_dir)
    summary = get_validation_summary(results)

    # Create summary table
    table = Table(title="Validation Results")
    table.add_column("Validator", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Errors", style="red")

    for name, result in results.items():
        is_valid = _is_result_valid(result)
        error_count = _get_error_count(result)
        status = "[green]PASS[/green]" if is_valid else "[red]FAIL[/red]"
        table.add_row(name.replace("_", " ").title(), status, str(error_count))

    console.print(table)

    # Print summary
    passed = summary.get("passed", 0)
    total = summary.get("total_validators", 0)
    console.print(f"\n[bold]Summary: {passed}/{total} passed[/bold]")

    all_valid = summary.get("failed", 0) == 0
    raise SystemExit(0 if all_valid else 1)


def _is_result_valid(result: object) -> bool:
    """Check if a validation result is valid."""
    if hasattr(result, "has_circular_imports"):
        return not bool(result.has_circular_imports)
    if hasattr(result, "is_valid"):
        return bool(result.is_valid)
    return False


def _get_error_count(result: object) -> int:
    """Get the error count from a validation result."""
    if hasattr(result, "has_circular_imports"):
        if hasattr(result, "cycles"):
            return len(result.cycles)
        return 1 if result.has_circular_imports else 0
    if hasattr(result, "errors"):
        return len(result.errors)
    return 0


def _print_result(name: str, result: object) -> None:
    """Print validation result with rich formatting."""
    if hasattr(result, "is_valid"):
        if result.is_valid:
            console.print(f"[bold green]{name}: PASS[/bold green]")
        else:
            console.print(f"[bold red]{name}: FAIL[/bold red]")
            if hasattr(result, "errors") and result.errors:
                for error in result.errors:
                    console.print(f"  [red]{error}[/red]")


if __name__ == "__main__":
    cli()
