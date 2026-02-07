# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Introspection trigger command.

Publishes a node introspection event to the event bus, simulating a node
announcing itself to the cluster. This is the canonical trigger for the
registration orchestrator workflow.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from uuid import uuid4

import click
from rich.console import Console

from omnibase_infra.cli.infra_test._helpers import get_broker

console = Console()

# Default ONEX topic for introspection events (5-segment format)
DEFAULT_INTROSPECTION_TOPIC = "onex.evt.platform.node-introspection.v1"


def _build_introspection_payload(
    node_id: str | None = None,
    node_type: str = "EFFECT",
    version: str = "1.0.0",
) -> dict[str, object]:
    """Build a minimal introspection event payload.

    Args:
        node_id: Optional node UUID. Auto-generated if not provided.
        node_type: ONEX node type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
        version: Node version string.

    Returns:
        Dict representing a ModelNodeIntrospectionEvent.
    """
    nid = node_id or str(uuid4())
    cid = str(uuid4())
    now = datetime.now(UTC).isoformat()

    parts = version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        raise click.BadParameter(
            f"Invalid version '{version}'. Expected semver format (e.g. 1.0.0).",
            param_hint="'--version'",
        ) from None

    return {
        "node_id": nid,
        "node_type": node_type,
        "node_version": {"major": major, "minor": minor, "patch": patch},
        "declared_capabilities": {},
        "discovered_capabilities": {},
        "contract_capabilities": None,
        "endpoints": {"health": f"http://localhost:8080/{nid}/health"},
        "reason": "STARTUP",
        "correlation_id": cid,
        "timestamp": now,
        "metadata": {},
    }


@click.command()
@click.option("--node-id", default=None, help="Node UUID (auto-generated if omitted).")
@click.option(
    "--node-type",
    type=click.Choice(["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"]),
    default="EFFECT",
    help="ONEX node type.",
    show_default=True,
)
@click.option("--version", default="1.0.0", help="Node version.", show_default=True)
@click.option(
    "--topic",
    default=DEFAULT_INTROSPECTION_TOPIC,
    envvar="ONEX_INTROSPECTION_TOPIC",
    help="Kafka topic for introspection events.",
    show_default=True,
)
@click.option(
    "--broker",
    default=None,
    envvar="KAFKA_BOOTSTRAP_SERVERS",
    help="Kafka bootstrap server (defaults to localhost:29092).",
)
def introspect(
    node_id: str | None,
    node_type: str,
    version: str,
    topic: str,
    broker: str | None,
) -> None:
    """Publish a node introspection event to the event bus.

    Simulates a node announcing itself to the cluster by publishing an
    introspection event to Kafka via ``rpk``.
    """
    # Click resolves envvar=KAFKA_BOOTSTRAP_SERVERS for --broker; apply
    # the final fallback only when neither the flag nor the envvar was set.
    resolved_broker: str = broker or get_broker()

    payload = _build_introspection_payload(node_id, node_type, version)
    payload_json = json.dumps(payload)

    nid = payload["node_id"]
    cid = payload["correlation_id"]

    console.print("[bold blue]Publishing introspection event...[/bold blue]")
    console.print(f"  node_id:        {nid}")
    console.print(f"  correlation_id: {cid}")
    console.print(f"  topic:          {topic}")
    console.print(f"  broker:         {resolved_broker}")

    # Use rpk to publish (available in Redpanda container and locally)
    result = subprocess.run(
        [
            "rpk",
            "topic",
            "produce",
            topic,
            "--brokers",
            resolved_broker,
            "-k",
            str(nid),
        ],
        input=payload_json,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    if result.returncode != 0:
        console.print(
            f"[bold red]Failed to publish:[/bold red] {result.stderr.strip()}"
        )
        raise SystemExit(1)

    console.print("[bold green]Introspection event published.[/bold green]")
