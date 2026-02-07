# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Environment management commands (``env up`` / ``env down``).

Wraps Docker Compose to manage the E2E testing infrastructure stack.
"""

from __future__ import annotations

import subprocess
import time

import click
from rich.console import Console

console = Console()


def _run_compose(
    compose_file: str,
    project_name: str,
    args: list[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a ``docker compose`` command.

    Args:
        compose_file: Path to the Docker Compose file.
        project_name: Docker Compose project name.
        args: Additional arguments to ``docker compose``.
        capture: If True, capture stdout/stderr instead of streaming.

    Returns:
        CompletedProcess result.

    Raises:
        SystemExit: If the command fails.
    """
    cmd = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "-p",
        project_name,
        *args,
    ]
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if capture and result.stderr:
            console.print(f"[red]{result.stderr.strip()}[/red]")
        raise SystemExit(result.returncode)
    return result


def _wait_for_healthy(
    compose_file: str,
    project_name: str,
    timeout_seconds: int = 120,
) -> bool:
    """Poll ``docker compose ps`` until all services are healthy.

    Args:
        compose_file: Path to the Docker Compose file.
        project_name: Docker Compose project name.
        timeout_seconds: Maximum time to wait.

    Returns:
        True if all services healthy, False on timeout.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                compose_file,
                "-p",
                project_name,
                "ps",
                "--format",
                "{{.Service}}\t{{.Health}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            time.sleep(2)
            continue

        lines = [
            line.strip() for line in result.stdout.strip().splitlines() if line.strip()
        ]
        if not lines:
            time.sleep(2)
            continue

        # Check health status for each service
        all_healthy = True
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            service, health = parts[0].strip(), parts[1].strip().lower()
            # Services without healthchecks show empty or "-"
            if health and health not in ("healthy", "-", ""):
                all_healthy = False
                break

        if all_healthy:
            return True

        time.sleep(2)

    return False


@click.group()
def env() -> None:
    """Manage the E2E testing environment."""


@env.command("up")
@click.option(
    "--profile",
    multiple=True,
    help="Docker Compose profiles to activate (e.g. --profile runtime).",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait for all services to become healthy.",
    show_default=True,
)
@click.option(
    "--timeout",
    default=120,
    help="Timeout in seconds for health checks.",
    show_default=True,
)
@click.pass_context
def env_up(
    ctx: click.Context,
    profile: tuple[str, ...],
    wait: bool,
    timeout: int,
) -> None:
    """Start local infrastructure for E2E testing.

    Starts PostgreSQL, Redpanda, Consul, Valkey, and Infisical via
    Docker Compose. Optionally activates the ``runtime`` profile to
    include the ONEX runtime container.
    """
    compose_file = ctx.obj["compose_file"]
    project_name = ctx.obj["project_name"]

    console.print("[bold blue]Starting E2E infrastructure...[/bold blue]")

    args = ["up", "-d"]
    for p in profile:
        args = ["--profile", p, *args]

    _run_compose(compose_file, project_name, args)

    if wait:
        console.print(
            f"[yellow]Waiting for services to become healthy (timeout={timeout}s)...[/yellow]"
        )
        if _wait_for_healthy(compose_file, project_name, timeout):
            console.print("[bold green]All services healthy.[/bold green]")
        else:
            console.print("[bold red]Timeout: not all services are healthy.[/bold red]")
            # Show current status for debugging
            _run_compose(compose_file, project_name, ["ps"])
            raise SystemExit(1)

    # Show running services
    _run_compose(compose_file, project_name, ["ps"])


@env.command("down")
@click.option(
    "--volumes/--no-volumes",
    "-v",
    default=True,
    help="Remove volumes (clean state).",
    show_default=True,
)
@click.pass_context
def env_down(ctx: click.Context, volumes: bool) -> None:
    """Stop and remove E2E infrastructure.

    Tears down all Docker Compose services. By default also removes
    volumes so the next ``env up`` starts with a clean state.
    """
    compose_file = ctx.obj["compose_file"]
    project_name = ctx.obj["project_name"]

    console.print("[bold blue]Stopping E2E infrastructure...[/bold blue]")

    args = ["down"]
    if volumes:
        args.append("-v")

    _run_compose(compose_file, project_name, args)
    console.print("[bold green]Infrastructure stopped.[/bold green]")
