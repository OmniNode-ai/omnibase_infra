# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Verification commands for infrastructure state.

Provides commands to verify the state of Consul, PostgreSQL, Kafka topics,
snapshot topics, and idempotency of the registration pipeline.
"""

from __future__ import annotations

import json
import re
import subprocess

import click
from rich.console import Console
from rich.table import Table

from omnibase_infra.cli.infra_test._helpers import (
    get_broker,
    get_consul_addr,
    get_postgres_dsn,
)

console = Console()

# ONEX 5-segment topic naming: onex.<kind>.<producer>.<event-name>.v<version>
# kind: evt, cmd, intent, snapshot, dlq
ONEX_TOPIC_PATTERN = re.compile(
    r"^onex\.(evt|cmd|intent|snapshot|dlq)\.[a-z][a-z0-9-]*\.[a-z][a-z0-9-]*\.v[0-9]+$"
)


@click.group()
def verify() -> None:
    """Verify infrastructure state."""


@verify.command("registry")
@click.option("--node-id", default=None, help="Filter by specific node UUID.")
def verify_registry(node_id: str | None) -> None:
    """Check registration state in Consul and PostgreSQL.

    Queries both Consul's KV store and PostgreSQL's registration_projections
    table to verify that node registrations are persisted correctly.
    """
    consul_ok = _verify_consul_registry(node_id)
    postgres_ok = _verify_postgres_registry(node_id)

    if consul_ok and postgres_ok:
        console.print("[bold green]Registry verification: PASS[/bold green]")
    else:
        console.print("[bold red]Registry verification: FAIL[/bold red]")
        raise SystemExit(1)


@verify.command("topics")
def verify_topics() -> None:
    """Check ONEX topic naming compliance.

    Lists all Kafka topics and validates they follow the ONEX 5-segment
    naming convention: ``onex.<kind>.<producer>.<event-name>.v<version>``.

    Non-ONEX topics (internal Redpanda topics like ``_schemas``) are
    reported but not flagged as violations.
    """
    broker = get_broker()

    console.print(
        f"[bold blue]Checking topic naming compliance ({broker})...[/bold blue]"
    )

    result = subprocess.run(
        ["rpk", "topic", "list", "--brokers", broker, "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    if result.returncode != 0:
        console.print(
            f"[bold red]Failed to list topics:[/bold red] {result.stderr.strip()}"
        )
        raise SystemExit(1)

    try:
        topics = json.loads(result.stdout)
    except json.JSONDecodeError:
        console.print("[bold red]Failed to parse topic list JSON.[/bold red]")
        raise SystemExit(1)

    table = Table(title="Topic Naming Compliance")
    table.add_column("Topic", style="cyan")
    table.add_column("ONEX Compliant", style="bold")
    table.add_column("Notes")

    violations = 0
    for topic_info in topics:
        name = (
            topic_info.get("name", topic_info)
            if isinstance(topic_info, dict)
            else str(topic_info)
        )

        # Skip internal Redpanda/Kafka topics
        if name.startswith("_"):
            table.add_row(name, "[dim]-[/dim]", "Internal topic (skipped)")
            continue

        if ONEX_TOPIC_PATTERN.match(name):
            table.add_row(name, "[green]YES[/green]", "")
        else:
            violations += 1
            table.add_row(
                name,
                "[red]NO[/red]",
                "Does not match onex.<kind>.<producer>.<name>.v<N>",
            )

    console.print(table)

    if violations > 0:
        console.print(
            f"[bold red]{violations} topic(s) violate ONEX naming convention.[/bold red]"
        )
        raise SystemExit(1)

    console.print("[bold green]All ONEX topics are compliant.[/bold green]")


@verify.command("snapshots")
@click.option(
    "--topic",
    default="onex.snapshot.platform.registration-snapshots.v1",
    help="Snapshot topic to verify.",
    show_default=True,
)
def verify_snapshots(topic: str) -> None:
    """Check snapshot topic state.

    Verifies that the compacted snapshot topic exists and contains at least
    one record, indicating the snapshot publisher has written data.
    """
    broker = get_broker()

    console.print(f"[bold blue]Verifying snapshot topic: {topic}[/bold blue]")

    # Check topic exists
    result = subprocess.run(
        ["rpk", "topic", "describe", topic, "--brokers", broker],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    if result.returncode != 0:
        console.print(f"[bold red]Snapshot topic not found:[/bold red] {topic}")
        raise SystemExit(1)

    # Check for compaction config
    if "cleanup.policy" in result.stdout and "compact" in result.stdout:
        console.print("  cleanup.policy: [green]compact[/green]")
    else:
        console.print(
            "  [yellow]Warning: cleanup.policy may not be set to compact[/yellow]"
        )

    # Consume one message to verify data exists
    consume_result = subprocess.run(
        [
            "rpk",
            "topic",
            "consume",
            topic,
            "--brokers",
            broker,
            "-n",
            "1",
            "--format",
            "%v\\n",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    if consume_result.returncode != 0 or not consume_result.stdout.strip():
        console.print("[yellow]Snapshot topic is empty (no records yet).[/yellow]")
        console.print("[bold yellow]Snapshots verification: WARN (empty)[/bold yellow]")
        return

    console.print("  [green]Snapshot data present.[/green]")
    console.print("[bold green]Snapshots verification: PASS[/bold green]")


@verify.command("idempotency")
@click.option(
    "--topic",
    default="onex.evt.platform.node-introspection.v1",
    help="Introspection topic.",
    show_default=True,
)
@click.option(
    "--repetitions",
    default=3,
    help="Number of identical events to publish.",
    show_default=True,
)
def verify_idempotency(topic: str, repetitions: int) -> None:
    """Verify idempotent registration (no duplicates).

    Publishes the same introspection event multiple times and verifies that
    only one registration record exists in PostgreSQL.

    This tests the orchestrator's idempotency guard: duplicate introspections
    for an already-processing or active node should be no-ops.
    """
    import time
    from uuid import uuid4

    from omnibase_infra.cli.infra_test.introspect import _build_introspection_payload

    broker = get_broker()
    node_id = str(uuid4())

    console.print(
        f"[bold blue]Testing idempotency: publishing {repetitions} identical events...[/bold blue]"
    )
    console.print(f"  node_id: {node_id}")

    # Publish same event N times
    payload = _build_introspection_payload(node_id=node_id)
    payload_json = json.dumps(payload)

    for i in range(repetitions):
        result = subprocess.run(
            [
                "rpk",
                "topic",
                "produce",
                topic,
                "--brokers",
                broker,
                "-k",
                node_id,
            ],
            input=payload_json,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            console.print(
                f"[red]Failed to publish event {i + 1}: {result.stderr.strip()}[/red]"
            )
            raise SystemExit(1)
        console.print(f"  Published event {i + 1}/{repetitions}")

    # Wait for processing
    console.print("[yellow]Waiting for registration processing (5s)...[/yellow]")
    time.sleep(5)

    # Check PostgreSQL for duplicate records
    dsn = get_postgres_dsn()
    count = _count_postgres_registrations(dsn, node_id)

    if count is None:
        console.print("[bold red]Failed to query PostgreSQL.[/bold red]")
        raise SystemExit(1)

    console.print(f"  Registration records found: {count}")

    if count <= 1:
        console.print("[bold green]Idempotency verification: PASS[/bold green]")
    else:
        console.print(
            f"[bold red]Idempotency verification: FAIL "
            f"(expected <= 1 record, found {count})[/bold red]"
        )
        raise SystemExit(1)


# =============================================================================
# Internal helpers
# =============================================================================


def _verify_consul_registry(node_id: str | None) -> bool:
    """Verify registrations in Consul KV store."""
    consul_addr = get_consul_addr()

    console.print(f"[bold blue]Checking Consul registry ({consul_addr})...[/bold blue]")

    import httpx

    try:
        url = f"{consul_addr}/v1/kv/onex/services/?keys"
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        keys = resp.json()
    except httpx.HTTPError as e:
        # KV not populated or unreachable -- fall back to service catalog
        console.print(
            f"  [dim]KV lookup failed ({type(e).__name__}), trying service catalog...[/dim]"
        )
        try:
            url = f"{consul_addr}/v1/agent/services"
            resp = httpx.get(url, timeout=5.0)
            resp.raise_for_status()
            services = resp.json()

            onex_services = {k: v for k, v in services.items() if "onex" in k.lower()}

            if not onex_services:
                if node_id:
                    console.print(
                        f"  [red]Node {node_id} not found in Consul services.[/red]"
                    )
                    return False
                console.print("  [yellow]No ONEX services found in Consul.[/yellow]")
                return True  # Not a failure if nothing registered yet

            table = Table(title="Consul ONEX Services")
            table.add_column("Service ID", style="cyan")
            table.add_column("Service Name")
            table.add_column("Tags")

            matched = 0
            for sid, svc in onex_services.items():
                if node_id and node_id not in sid:
                    continue
                matched += 1
                table.add_row(
                    sid, svc.get("Service", ""), ", ".join(svc.get("Tags", []))
                )

            console.print(table)
            if node_id and matched == 0:
                console.print(
                    f"  [red]Node {node_id} not found in Consul services.[/red]"
                )
                return False
            return True
        except httpx.HTTPError as e:
            console.print(
                f"  [red]Cannot reach Consul at {consul_addr}: {type(e).__name__}: {e}[/red]"
            )
            return False

    if not keys:
        if node_id:
            console.print(f"  [red]Node {node_id} not found in Consul KV.[/red]")
            return False
        console.print("  [yellow]No ONEX service keys in Consul KV.[/yellow]")
        return True

    table = Table(title="Consul KV Service Keys")
    table.add_column("Key", style="cyan")

    matched = 0
    for key in keys:
        if node_id and node_id not in key:
            continue
        matched += 1
        table.add_row(key)

    console.print(table)
    if node_id and matched == 0:
        console.print(f"  [red]Node {node_id} not found in Consul KV.[/red]")
        return False
    console.print(f"  [green]{len(keys)} service key(s) found.[/green]")
    return True


def _verify_postgres_registry(node_id: str | None) -> bool:
    """Verify registrations in PostgreSQL."""
    dsn = get_postgres_dsn()

    console.print("[bold blue]Checking PostgreSQL registry...[/bold blue]")

    import psycopg2

    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                if node_id:
                    cur.execute(
                        "SELECT entity_id, current_state, node_type, updated_at "
                        "FROM registration_projections WHERE entity_id = %s",
                        (node_id,),
                    )
                else:
                    cur.execute(
                        "SELECT entity_id, current_state, node_type, updated_at "
                        "FROM registration_projections ORDER BY updated_at DESC LIMIT 20"
                    )

                rows = cur.fetchall()
        finally:
            conn.close()
    except psycopg2.Error as e:
        console.print(f"  [red]Cannot connect to PostgreSQL: {type(e).__name__}[/red]")
        return False

    if not rows:
        if node_id:
            console.print(f"  [red]Node {node_id} not found in projections.[/red]")
            return False
        console.print("  [yellow]No registration projections found.[/yellow]")
        return True

    table = Table(title="Registration Projections")
    table.add_column("Entity ID", style="cyan")
    table.add_column("State", style="bold")
    table.add_column("Node Type")
    table.add_column("Updated")

    for row in rows:
        entity_id, state, ntype, updated = row
        state_style = "[green]" if state == "ACTIVE" else "[yellow]"
        table.add_row(
            str(entity_id),
            f"{state_style}{state}[/{state_style.strip('[')}",
            str(ntype or ""),
            str(updated or ""),
        )

    console.print(table)
    console.print(f"  [green]{len(rows)} projection(s) found.[/green]")
    return True


def _count_postgres_registrations(dsn: str, node_id: str) -> int | None:
    """Count registration records for a node in PostgreSQL."""
    import psycopg2

    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM registration_projections WHERE entity_id = %s",
                    (node_id,),
                )
                row = cur.fetchone()
                count: int = row[0] if row else 0
            return count
        finally:
            conn.close()
    except psycopg2.Error as e:
        console.print(f"  [red]PostgreSQL error: {type(e).__name__}: {e}[/red]")
        return None
