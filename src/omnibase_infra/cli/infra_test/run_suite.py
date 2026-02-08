# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test suite runner (``run --suite smoke|failure``).

Orchestrates multi-step integration test suites by invoking the
individual CLI commands in sequence.
"""

from __future__ import annotations

import json
import subprocess
import time
from uuid import uuid4

import click
from rich.console import Console
from rich.panel import Panel

from omnibase_infra.cli.infra_test._helpers import get_broker, get_postgres_dsn

console = Console()


def _get_compose_args(ctx: click.Context) -> tuple[str, str]:
    """Extract compose file and project name from context."""
    compose_file = ctx.obj["compose_file"]
    project_name = ctx.obj["project_name"]
    return compose_file, project_name


def _step(name: str) -> None:
    """Print a suite step header."""
    console.print(f"\n[bold cyan]--- {name} ---[/bold cyan]")


def _run_cli_command(args: list[str]) -> int:
    """Run a CLI subcommand via subprocess.

    Args:
        args: Command arguments (e.g. ["verify", "registry"]).

    Returns:
        Exit code.
    """
    cmd = ["onex-infra-test", *args]
    result = subprocess.run(cmd, check=False, timeout=120)
    return result.returncode


def _verify_env_running(compose_file: str, project_name: str) -> None:
    """Verify the E2E docker-compose environment is running.

    Raises:
        SystemExit: If no running containers are found.
    """
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            compose_file,
            "-p",
            project_name,
            "ps",
            "--status",
            "running",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        console.print(
            "[bold red]E2E environment is not running. Run 'env up' first.[/bold red]"
        )
        raise SystemExit(1)
    console.print("  [green]Environment running.[/green]")


def _publish_introspection(
    broker: str,
    topic: str,
    node_id: str,
    payload_json: str | None = None,
) -> bool:
    """Publish an introspection event via rpk.

    Args:
        broker: Kafka bootstrap server.
        topic: Target topic.
        node_id: Node UUID string.
        payload_json: Pre-serialised JSON payload. When *None* a fresh
            payload is built (new ``correlation_id`` and ``timestamp``).
            Pass an explicit value to guarantee byte-identical publishes
            (e.g. idempotency tests).

    Returns:
        True if published successfully.
    """
    if payload_json is None:
        from omnibase_infra.cli.infra_test.introspect import (
            _build_introspection_payload,
        )

        payload = _build_introspection_payload(node_id=node_id)
        payload_json = json.dumps(payload)

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
    return result.returncode == 0


def _wait_for_registration(dsn: str, node_id: str, timeout: int = 30) -> bool:
    """Wait for a registration projection to appear in PostgreSQL.

    Args:
        dsn: PostgreSQL connection string.
        node_id: Node UUID to look for.
        timeout: Seconds to wait.

    Returns:
        True if found within timeout.
    """
    import psycopg2

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            conn = psycopg2.connect(dsn, connect_timeout=5)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT current_state FROM registration_projections WHERE entity_id = %s",
                        (node_id,),
                    )
                    row = cur.fetchone()
            finally:
                conn.close()
            if row is not None:
                console.print(f"  Registration state: [green]{row[0]}[/green]")
                return True
        except psycopg2.Error as e:
            # Transient database errors during polling -- log and retry
            console.print(f"  [dim]Poll error: {type(e).__name__}: {e}[/dim]")
        time.sleep(2)

    return False


@click.command("run")
@click.option(
    "--suite",
    type=click.Choice(["smoke", "idempotency", "failure"]),
    required=True,
    help="Test suite to execute.",
)
@click.pass_context
def run_suite(ctx: click.Context, suite: str) -> None:
    """Execute an integration test suite.

    Available suites:

    \b
    smoke       - Happy path: env up, introspect, verify all, env down
    idempotency - Duplicate event handling: publish N times, assert exactly 1 record
    failure     - Kill runtime, restart, verify recovery with no data loss
    """
    compose_file, project_name = _get_compose_args(ctx)

    console.print(
        Panel(
            f"[bold]Running suite: {suite}[/bold]",
            title="onex-infra-test",
            border_style="blue",
        )
    )

    if suite == "smoke":
        _run_smoke_suite(compose_file, project_name)
    elif suite == "idempotency":
        _run_idempotency_suite(compose_file, project_name)
    elif suite == "failure":
        _run_failure_suite(compose_file, project_name)


def _run_smoke_suite(compose_file: str, project_name: str) -> None:
    """Smoke suite: happy-path registration verification.

    Steps:
        1. Verify environment is running
        2. Publish introspection event
        3. Wait for registration to complete
        4. Verify registry (Consul + PostgreSQL)
        5. Verify topic naming compliance
        6. Verify snapshot topic
    """
    broker = get_broker()
    dsn = get_postgres_dsn()
    topic = "onex.evt.platform.node-introspection.v1"
    node_id = str(uuid4())
    failures: list[str] = []

    _step("1. Verify environment is running")
    _verify_env_running(compose_file, project_name)

    _step("2. Publish introspection event")
    console.print(f"  node_id: {node_id}")
    if not _publish_introspection(broker, topic, node_id):
        console.print("[bold red]Failed to publish introspection event.[/bold red]")
        raise SystemExit(1)
    console.print("  [green]Published.[/green]")

    _step("3. Wait for registration")
    if not _wait_for_registration(dsn, node_id, timeout=30):
        console.print("[bold red]Registration not found within 30s.[/bold red]")
        failures.append("Registration not found in PostgreSQL")

    _step("4. Verify registry state")
    rc = _run_cli_command(["verify", "registry", "--node-id", node_id])
    if rc != 0:
        failures.append("Registry verification failed")

    _step("5. Verify topic naming")
    rc = _run_cli_command(["verify", "topics"])
    if rc != 0:
        failures.append("Topic naming verification failed")

    _step("6. Verify snapshots")
    rc = _run_cli_command(["verify", "snapshots"])
    # Snapshots may be empty in a fresh environment, so only warn
    if rc != 0:
        console.print(
            "  [yellow]Snapshot verification returned non-zero (may be empty).[/yellow]"
        )

    # Summary
    console.print()
    if failures:
        console.print("[bold red]Smoke suite: FAIL[/bold red]")
        for f in failures:
            console.print(f"  [red]- {f}[/red]")
        raise SystemExit(1)

    console.print("[bold green]Smoke suite: PASS[/bold green]")


def _run_idempotency_suite(compose_file: str, project_name: str) -> None:
    """Idempotency suite: duplicate event handling.

    Steps:
        1. Verify environment is running
        2. Publish same introspection event 3x rapidly (byte-identical payloads)
        3. Wait for processing
        4. Assert exactly 1 registration record in PostgreSQL
    """
    from omnibase_infra.cli.infra_test.introspect import _build_introspection_payload

    broker = get_broker()
    dsn = get_postgres_dsn()
    topic = "onex.evt.platform.node-introspection.v1"
    node_id = str(uuid4())
    repetitions = 3

    _step("1. Verify environment")
    _verify_env_running(compose_file, project_name)

    _step(f"2. Publish {repetitions} identical events")
    console.print(f"  node_id: {node_id}")

    # Build the payload once so every publish is byte-identical
    # (same correlation_id + timestamp).
    payload = _build_introspection_payload(node_id=node_id)
    frozen_payload_json = json.dumps(payload)

    for i in range(repetitions):
        if not _publish_introspection(
            broker, topic, node_id, payload_json=frozen_payload_json
        ):
            console.print(f"[red]Failed on event {i + 1}[/red]")
            raise SystemExit(1)
        console.print(f"  Published {i + 1}/{repetitions}")

    # Allow time for Kafka consumer lag to clear after rapid-fire publishes.
    idempotency_settle_seconds = 5
    _step(f"3. Wait for processing ({idempotency_settle_seconds}s)")
    time.sleep(idempotency_settle_seconds)

    _step("4. Check for duplicates")
    import psycopg2

    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM registration_projections WHERE entity_id = %s",
                    (node_id,),
                )
                count = cur.fetchone()[0]
        finally:
            conn.close()
    except psycopg2.Error as e:
        console.print(f"[bold red]PostgreSQL error: {type(e).__name__}: {e}[/bold red]")
        raise SystemExit(1)

    console.print(f"  Records found: {count}")

    if count == 0:
        console.print(
            "[bold red]Idempotency suite: FAIL "
            "(no registrations found -- events were not processed)[/bold red]"
        )
        raise SystemExit(1)
    if count == 1:
        console.print("[bold green]Idempotency suite: PASS[/bold green]")
    else:
        console.print(
            f"[bold red]Idempotency suite: FAIL (expected 1 record, found {count})[/bold red]"
        )
        raise SystemExit(1)


def _run_failure_suite(compose_file: str, project_name: str) -> None:
    """Failure suite: runtime kill and recovery.

    Tests that the system recovers after the runtime process is killed.
    The registration orchestrator runs INSIDE the runtime-main process,
    not as a separate container.

    Steps:
        1. Verify environment + runtime are running
        2. Publish introspection event for node A, verify registration
        3. Kill runtime-main process
        4. Publish introspection event for node B (goes to Kafka)
        5. Restart runtime
        6. Verify node B registration completes (backfill from Kafka)
        7. Verify node A registration still exists (no data loss)
    """
    broker = get_broker()
    dsn = get_postgres_dsn()
    topic = "onex.evt.platform.node-introspection.v1"
    node_a = str(uuid4())
    node_b = str(uuid4())

    _step("1. Verify environment with runtime profile")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            compose_file,
            "-p",
            project_name,
            "ps",
            "--services",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    services = result.stdout.strip().splitlines()
    if "runtime" not in services:
        console.print(
            "[bold red]Runtime service not found. "
            "Start with: onex-infra-test env up --profile runtime[/bold red]"
        )
        raise SystemExit(1)
    console.print("  [green]Runtime service present.[/green]")

    _step("2. Register node A")
    console.print(f"  node_a: {node_a}")
    if not _publish_introspection(broker, topic, node_a):
        console.print("[bold red]Failed to publish for node A.[/bold red]")
        raise SystemExit(1)

    if not _wait_for_registration(dsn, node_a, timeout=30):
        console.print("[bold red]Node A registration not found.[/bold red]")
        raise SystemExit(1)
    console.print("  [green]Node A registered.[/green]")

    _step("3. Kill runtime process")
    kill_result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            compose_file,
            "-p",
            project_name,
            "kill",
            "runtime",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if kill_result.returncode != 0:
        console.print(
            f"[bold red]Failed to kill runtime: {kill_result.stderr.strip()}[/bold red]"
        )
        raise SystemExit(1)
    console.print("  [green]Runtime killed.[/green]")

    _step("4. Publish introspection for node B (while runtime is down)")
    console.print(f"  node_b: {node_b}")
    if not _publish_introspection(broker, topic, node_b):
        console.print("[bold red]Failed to publish for node B.[/bold red]")
        raise SystemExit(1)
    console.print("  [green]Event published to Kafka (buffered).[/green]")

    _step("5. Restart runtime")
    restart_result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            compose_file,
            "-p",
            project_name,
            "start",
            "runtime",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if restart_result.returncode != 0:
        console.print(
            f"[bold red]Failed to restart runtime: {restart_result.stderr.strip()}[/bold red]"
        )
        raise SystemExit(1)

    # Wait for runtime to become healthy
    console.print("  [yellow]Waiting for runtime health (10s)...[/yellow]")
    time.sleep(10)  # Give it time to start
    console.print("  [green]Runtime restarted.[/green]")

    _step("6. Verify node B registration (backfill)")
    if not _wait_for_registration(dsn, node_b, timeout=60):
        console.print(
            "[bold red]Node B registration not found after recovery.[/bold red]"
        )
        console.print("[bold red]Failure suite: FAIL (backfill failed)[/bold red]")
        raise SystemExit(1)
    console.print("  [green]Node B registered (backfill succeeded).[/green]")

    _step("7. Verify node A still exists (no data loss)")
    import psycopg2

    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT current_state FROM registration_projections WHERE entity_id = %s",
                    (node_a,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
    except psycopg2.Error as e:
        console.print(f"[bold red]PostgreSQL error: {type(e).__name__}: {e}[/bold red]")
        raise SystemExit(1)

    if row is None:
        console.print("[bold red]Node A data lost after restart![/bold red]")
        console.print("[bold red]Failure suite: FAIL (data loss)[/bold red]")
        raise SystemExit(1)

    console.print(f"  Node A state: [green]{row[0]}[/green]")

    console.print("[bold green]Failure suite: PASS[/bold green]")
