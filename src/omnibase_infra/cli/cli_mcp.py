# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP CLI Commands - Entry point for MCP server.

Provides CLI commands for running the MCP server:
- `onex mcp serve`: Start the MCP server with Consul discovery and Kafka hot reload
- `onex mcp serve --dev`: Start in dev mode with local contract scanning

Usage:
    ```bash
    # Production: uses Consul + Kafka
    onex mcp serve

    # Custom port
    onex mcp serve --port 9000

    # Dev mode: bypasses Consul, scans local contracts
    onex mcp serve --dev --contracts-dir ./nodes

    # Custom Consul configuration
    onex mcp serve --consul-host consul.local --consul-port 8500
    ```
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

import click
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


@click.group(name="mcp")
def mcp_cli() -> None:
    """MCP (Model Context Protocol) commands.

    Manage the MCP server for exposing ONEX orchestrators as AI agent tools.
    """


@mcp_cli.command(name="serve")
@click.option(
    "--port", type=int, default=8090, help="MCP HTTP server port (default: 8090)"
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",  # noqa: S104 - Intentional bind-all for server CLI
    help="MCP HTTP server host (default: 0.0.0.0)",
)
@click.option(
    "--consul-host", type=str, default=None, help="Consul host (default: from env)"
)
@click.option(
    "--consul-port", type=int, default=None, help="Consul port (default: from env)"
)
@click.option("--no-kafka", is_flag=True, help="Disable Kafka hot reload")
@click.option(
    "--dev", is_flag=True, help="Dev mode: bypass Consul, scan local contracts"
)
@click.option(
    "--contracts-dir", type=str, default=None, help="Contracts directory for dev mode"
)
@click.option(
    "--timeout", type=float, default=30.0, help="Tool execution timeout (default: 30.0)"
)
@click.pass_context
def serve(ctx: click.Context, **kwargs: object) -> None:
    """Start the MCP server for exposing ONEX orchestrators as tools.

    The MCP server enables AI agents (Claude, etc.) to discover and invoke
    ONEX orchestrator nodes as MCP tools.

    \b
    Production Mode (default):
        - Discovers tools from Consul (services with mcp-enabled tag)
        - Subscribes to Kafka for hot reload of tool registry
        - Requires Consul and optionally Kafka to be running

    \b
    Dev Mode (--dev):
        - Scans local contracts directory for MCP-enabled orchestrators
        - No Consul or Kafka required
        - Useful for local development and testing

    \b
    Examples:
        # Start with defaults
        onex mcp serve

        # Custom port
        onex mcp serve --port 9000

        # Dev mode with local contracts
        onex mcp serve --dev --contracts-dir ./src/omnibase_infra/nodes

        # Disable Kafka hot reload
        onex mcp serve --no-kafka
    """
    # Extract parameters from kwargs with proper type handling
    # Click guarantees types for decorated options, but kwargs typing is generic
    port_val = kwargs.get("port")
    port: int = port_val if isinstance(port_val, int) else 8090
    host_val = kwargs.get("host")
    host: str = host_val if isinstance(host_val, str) else "0.0.0.0"  # noqa: S104
    consul_host_val = kwargs.get("consul_host")
    consul_host: str | None = (
        consul_host_val if isinstance(consul_host_val, str) else None
    )
    consul_port_val = kwargs.get("consul_port")
    consul_port: int | None = (
        consul_port_val if isinstance(consul_port_val, int) else None
    )
    no_kafka: bool = bool(kwargs.get("no_kafka", False))
    dev: bool = bool(kwargs.get("dev", False))
    contracts_dir_val = kwargs.get("contracts_dir")
    contracts_dir: str | None = (
        contracts_dir_val if isinstance(contracts_dir_val, str) else None
    )
    timeout_val = kwargs.get("timeout")
    timeout: float = timeout_val if isinstance(timeout_val, (int, float)) else 30.0

    # Resolve Consul configuration from CLI > env > standard defaults
    # Use explicit type narrowing for mypy
    effective_consul_host: str = (
        consul_host
        if consul_host is not None
        else os.getenv("CONSUL_HOST", "localhost")
    )
    if consul_port is not None:
        effective_consul_port: int = consul_port
    else:
        port_env = os.getenv("CONSUL_PORT", "8500")
        try:
            effective_consul_port = int(port_env)
        except ValueError:
            console.print(
                f"[red]Invalid CONSUL_PORT: '{port_env}' is not a number[/red]"
            )
            sys.exit(1)

    console.print("[bold blue]Starting MCP Server[/bold blue]")
    console.print(f"  Host: {host}:{port}")

    if dev:
        console.print("  Mode: [yellow]Development[/yellow] (local contracts)")
        if contracts_dir:
            console.print(f"  Contracts: {contracts_dir}")
    else:
        console.print("  Mode: [green]Production[/green] (Consul + Kafka)")
        console.print(f"  Consul: {effective_consul_host}:{effective_consul_port}")
        console.print(f"  Kafka Hot Reload: {'disabled' if no_kafka else 'enabled'}")

    console.print(f"  Timeout: {timeout}s")
    console.print()

    # Run the async server
    try:
        asyncio.run(
            _run_server(
                port=port,
                host=host,
                consul_host=effective_consul_host,
                consul_port=effective_consul_port,
                kafka_enabled=not no_kafka,
                dev_mode=dev,
                contracts_dir=contracts_dir,
                timeout=timeout,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        logger.exception("Server error")
        sys.exit(1)


async def _run_server(
    port: int,
    host: str,
    consul_host: str,
    consul_port: int,
    kafka_enabled: bool,
    dev_mode: bool,
    contracts_dir: str | None,
    timeout: float,
) -> None:
    """Run the MCP server with the given configuration.

    Args:
        port: HTTP port to bind.
        host: Host to bind.
        consul_host: Consul server host.
        consul_port: Consul server port.
        kafka_enabled: Whether to enable Kafka hot reload.
        dev_mode: Whether to run in dev mode.
        contracts_dir: Directory for contract scanning in dev mode.
        timeout: Default tool execution timeout.
    """
    from omnibase_infra.services.mcp import (
        MCPServerLifecycle,
        ModelMCPServerConfig,
    )

    # Build configuration
    config = ModelMCPServerConfig(
        consul_host=consul_host,
        consul_port=consul_port,
        kafka_enabled=kafka_enabled and not dev_mode,
        http_host=host,
        http_port=port,
        default_timeout=timeout,
        dev_mode=dev_mode,
        contracts_dir=contracts_dir,
    )

    # Create Kafka bus if enabled
    bus = None
    if kafka_enabled and not dev_mode:
        try:
            from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

            bus = EventBusKafka.default()
            await bus.start()
            console.print("[green]Kafka connected[/green]")
        except Exception as e:
            console.print(f"[yellow]Kafka unavailable: {e}[/yellow]")
            console.print("[yellow]Continuing without hot reload[/yellow]")
            bus = None

    # Create and start lifecycle
    lifecycle = MCPServerLifecycle(config=config, bus=bus)
    await lifecycle.start()

    tool_count = lifecycle.registry.tool_count if lifecycle.registry else 0
    console.print(f"[green]MCP server started with {tool_count} tools[/green]")
    console.print(f"[green]Listening on http://{host}:{port}/mcp[/green]")
    console.print()
    console.print("Press Ctrl+C to stop")

    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Signal handlers not supported on Windows
        pass

    try:
        # Run uvicorn server
        # Note: In production, you'd integrate with the existing MCP transport
        # For now, we'll just wait for shutdown signal
        # The actual HTTP serving would be handled by uvicorn with the MCP transport

        # Import and run uvicorn if available
        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.responses import JSONResponse
            from starlette.routing import Route

            async def health(_request: object) -> JSONResponse:
                return JSONResponse(
                    {
                        "status": "healthy",
                        "tool_count": lifecycle.registry.tool_count
                        if lifecycle.registry
                        else 0,
                        "kafka_connected": bus is not None and bus._started
                        if bus
                        else False,
                    }
                )

            async def tools_list(_request: object) -> JSONResponse:
                if lifecycle.registry:
                    tools = await lifecycle.registry.list_tools()
                    return JSONResponse(
                        {
                            "tools": [
                                {
                                    "name": t.name,
                                    "description": t.description,
                                    "endpoint": t.endpoint,
                                }
                                for t in tools
                            ]
                        }
                    )
                return JSONResponse({"tools": []})

            app = Starlette(
                routes=[
                    Route("/health", health, methods=["GET"]),
                    Route("/mcp/tools", tools_list, methods=["GET"]),
                ],
            )

            config_uvicorn = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(config_uvicorn)

            # Run server in background
            server_task = asyncio.create_task(server.serve())

            # Wait for shutdown signal
            await shutdown_event.wait()

            # Stop server
            server.should_exit = True
            await server_task

        except ImportError:
            console.print(
                "[yellow]uvicorn not available, running in wait mode[/yellow]"
            )
            # Just wait for shutdown signal
            await shutdown_event.wait()

    finally:
        # Cleanup
        console.print("\n[yellow]Shutting down...[/yellow]")
        await lifecycle.shutdown()

        if bus is not None:
            await bus.close()

        console.print("[green]Shutdown complete[/green]")


# Export for CLI registration
__all__ = ["mcp_cli"]
