"""
ONEX Infrastructure CLI Commands.

Provides CLI interface for infrastructure management, validation, and
registry queries.
"""

from __future__ import annotations

import asyncio
import os
import re
from urllib.parse import quote_plus
from uuid import uuid4

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


# =============================================================================
# Registry Query Commands
# =============================================================================


@cli.group()
def registry() -> None:
    """Registry discovery and node query commands.

    Query the registration projection database directly without needing
    the full FastAPI server running.
    """


def _get_db_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Resolution order:
        1. ``OMNIBASE_INFRA_DB_URL`` -- full DSN, returned as-is.
        2. Individual ``POSTGRES_*`` variables assembled into a DSN.

    Raises:
        click.ClickException: If no database name is configured.
    """
    # Prefer explicit DSN when available (OMN-2146)
    db_url = os.environ.get("OMNIBASE_INFRA_DB_URL", "")
    if db_url:
        if not db_url.startswith(("postgresql://", "postgres://")):
            click.echo(
                "Error: OMNIBASE_INFRA_DB_URL must start with postgresql:// or postgres://",
                err=True,
            )
            raise SystemExit(1)
        return db_url

    host = os.environ.get("POSTGRES_HOST", "192.168.86.200")
    port = os.environ.get("POSTGRES_PORT", "5436")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    database = os.environ.get("POSTGRES_DATABASE", "")
    if not database:
        raise click.ClickException(
            "No database configured. Set OMNIBASE_INFRA_DB_URL or POSTGRES_DATABASE."
        )
    if not password:
        click.echo(
            "Warning: POSTGRES_PASSWORD not set. Set it via environment or .env file.",
            err=True,
        )
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{quote_plus(database)}"


def _sanitize_dsn(dsn: str) -> str:
    """Mask the password portion of a DSN to prevent credential leaks."""
    return re.sub(r"://([^:]+):([^@]*)@", r"://\1:****@", dsn)


async def _run_list_nodes(
    state: str | None,
    node_type: str | None,
    limit: int,
) -> None:
    """Async implementation for list-nodes command."""
    import asyncpg

    from omnibase_infra.enums import EnumRegistrationState
    from omnibase_infra.models.projection.model_registration_projection import (
        ModelRegistrationProjection,
    )
    from omnibase_infra.projectors import ProjectionReaderRegistration

    dsn = _get_db_dsn()
    correlation_id = uuid4()
    try:
        pool = await asyncio.wait_for(
            asyncpg.create_pool(dsn, min_size=1, max_size=2),
            timeout=10.0,
        )
    except TimeoutError:
        sanitized = _sanitize_dsn(dsn)
        console.print(
            f"[red]Connection timed out to {sanitized} (correlation_id={correlation_id})[/red]"
        )
        raise SystemExit(1)
    except Exception as e:
        sanitized = _sanitize_dsn(dsn)
        console.print(
            f"[red]Failed to connect to {sanitized}: {type(e).__name__} "
            f"(correlation_id={correlation_id})[/red]"
        )
        raise SystemExit(1)
    try:
        reader = ProjectionReaderRegistration(pool)

        state_filter = None
        if state:
            try:
                state_filter = EnumRegistrationState(state.lower())
            except ValueError:
                console.print(f"[red]Invalid state: {state}[/red]")
                console.print(
                    f"Valid states: {', '.join(s.value for s in EnumRegistrationState)}"
                )
                raise SystemExit(1)

        projections: list[ModelRegistrationProjection] = []
        if state_filter is not None:
            raw = await reader.get_by_state(
                state=state_filter, limit=limit, correlation_id=correlation_id
            )
            if node_type:
                nt_upper = node_type.upper()
                projections = [p for p in raw if p.node_type.value.upper() == nt_upper][
                    :limit
                ]
            else:
                projections = raw
        else:
            remaining = limit
            for query_state in [
                EnumRegistrationState.ACTIVE,
                EnumRegistrationState.ACCEPTED,
                EnumRegistrationState.AWAITING_ACK,
                EnumRegistrationState.ACK_RECEIVED,
                EnumRegistrationState.PENDING_REGISTRATION,
            ]:
                if remaining <= 0:
                    break
                state_projections = await reader.get_by_state(
                    state=query_state,
                    limit=remaining,
                    correlation_id=correlation_id,
                )
                if node_type:
                    nt_upper = node_type.upper()
                    state_projections = [
                        p
                        for p in state_projections
                        if p.node_type.value.upper() == nt_upper
                    ]
                projections.extend(state_projections)
                remaining = limit - len(projections)

        if not projections:
            console.print("[yellow]No nodes found matching criteria[/yellow]")
            raise SystemExit(0)

        table = Table(title=f"Registered Nodes ({len(projections)})")
        table.add_column("Node ID", style="cyan", max_width=12)
        table.add_column("Type", style="bold")
        table.add_column("State", style="green")
        table.add_column("Version", style="dim")
        table.add_column("Last Heartbeat", style="dim")
        table.add_column("Registered At", style="dim")

        for proj in projections:
            hb = (
                proj.last_heartbeat_at.strftime("%Y-%m-%d %H:%M:%S")
                if proj.last_heartbeat_at
                else "-"
            )
            reg = (
                proj.registered_at.strftime("%Y-%m-%d %H:%M:%S")
                if proj.registered_at
                else "-"
            )
            table.add_row(
                str(proj.entity_id)[:12],
                proj.node_type.value.upper(),
                proj.current_state.value,
                str(proj.node_version) if proj.node_version else "-",
                hb,
                reg,
            )

        console.print(table)
    finally:
        await pool.close()


async def _run_get_node(node_id_str: str) -> None:
    """Async implementation for get-node command."""
    from uuid import UUID

    import asyncpg

    from omnibase_infra.projectors import ProjectionReaderRegistration

    try:
        node_id = UUID(node_id_str)
    except ValueError:
        console.print(f"[red]Invalid UUID: {node_id_str}[/red]")
        raise SystemExit(1)

    dsn = _get_db_dsn()
    correlation_id = uuid4()
    try:
        pool = await asyncio.wait_for(
            asyncpg.create_pool(dsn, min_size=1, max_size=2),
            timeout=10.0,
        )
    except TimeoutError:
        sanitized = _sanitize_dsn(dsn)
        console.print(
            f"[red]Connection timed out to {sanitized} (correlation_id={correlation_id})[/red]"
        )
        raise SystemExit(1)
    except Exception as e:
        sanitized = _sanitize_dsn(dsn)
        console.print(
            f"[red]Failed to connect to {sanitized}: {type(e).__name__} "
            f"(correlation_id={correlation_id})[/red]"
        )
        raise SystemExit(1)
    try:
        reader = ProjectionReaderRegistration(pool)
        proj = await reader.get_entity_state(
            entity_id=node_id, correlation_id=correlation_id
        )

        if proj is None:
            console.print(f"[yellow]Node not found: {node_id}[/yellow]")
            raise SystemExit(1)

        console.print(f"[bold cyan]Node: {proj.entity_id}[/bold cyan]")
        console.print(f"  Type:          {proj.node_type.value.upper()}")
        console.print(f"  State:         {proj.current_state.value}")
        console.print(f"  Version:       {proj.node_version or '-'}")
        console.print(f"  Domain:        {proj.domain}")
        if proj.registered_at:
            console.print(
                f"  Registered:    {proj.registered_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if proj.last_heartbeat_at:
            console.print(
                f"  Last Heartbeat:{proj.last_heartbeat_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if proj.liveness_deadline:
            console.print(
                f"  Liveness Until:{proj.liveness_deadline.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if proj.capability_tags:
            console.print(f"  Capabilities:  {', '.join(proj.capability_tags)}")
    finally:
        await pool.close()


def _run_list_topics() -> None:
    """Implementation for list-topics command."""
    from omnibase_infra.topics import ALL_PLATFORM_SUFFIXES

    table = Table(title="ONEX Platform Topics")
    table.add_column("Topic Suffix", style="cyan")
    table.add_column("Kind", style="bold")
    table.add_column("Description", style="dim")

    for suffix in ALL_PLATFORM_SUFFIXES:
        # Parse kind from suffix: onex.<kind>.<producer>.<event-name>.v<version>
        parts = suffix.split(".")
        kind = parts[1] if len(parts) > 1 else "unknown"
        kind_labels = {
            "evt": "Event",
            "cmd": "Command",
            "intent": "Intent",
            "snapshot": "Snapshot",
            "dlq": "DLQ",
        }
        kind_label = kind_labels.get(kind, kind)

        # Build description from parts
        desc = ".".join(parts[2:-1]) if len(parts) > 3 else suffix

        table.add_row(suffix, kind_label, desc)

    console.print(table)


@registry.command("list-nodes")
@click.option("--state", default=None, help="Filter by registration state")
@click.option(
    "--node-type",
    default=None,
    help="Filter by node type (effect, compute, reducer, orchestrator)",
)
@click.option(
    "--limit", default=100, type=click.IntRange(min=1), help="Maximum number of results"
)
def registry_list_nodes(state: str | None, node_type: str | None, limit: int) -> None:
    """List registered nodes from the projection database."""
    try:
        asyncio.run(_run_list_nodes(state, node_type, limit))
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {type(e).__name__}[/red]")
        raise SystemExit(1)


@registry.command("get-node")
@click.argument("node_id")
def registry_get_node(node_id: str) -> None:
    """Show details for a specific node by UUID."""
    try:
        asyncio.run(_run_get_node(node_id))
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {type(e).__name__}[/red]")
        raise SystemExit(1)


@registry.command("list-topics")
def registry_list_topics() -> None:
    """List all ONEX platform topics."""
    try:
        _run_list_topics()
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {type(e).__name__}[/red]")
        raise SystemExit(1)


# =============================================================================
# Utility Functions
# =============================================================================


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
