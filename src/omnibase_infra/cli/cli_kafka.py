# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""onex kafka — local Kafka command topic publisher (OMN-8435).

Eliminates the SSH+rpk ritual for publishing ONEX Kafka command topics from a
dev machine. Registered into the onex CLI via the onex.cli entry-point group.

Usage:
    onex kafka produce <topic> --payload '<json>' [--dry-run] [--envelope]

V1 limitation: PLAINTEXT auth only. Redpanda on LAN (.201) is unauthenticated
from dev machines. SASL/TLS support is deferred to V2.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from uuid import uuid4

import click
from aiokafka import AIOKafkaProducer


def _build_envelope(
    payload_dict: dict[str, object], requested_by: str
) -> dict[str, object]:
    envelope: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "requested_by": requested_by,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    envelope.update(payload_dict)
    return envelope


@click.group()
def kafka() -> None:  # stub-ok: click group, subcommands via @kafka.command()
    """Kafka utilities — publish to command topics without SSH."""


@kafka.command()
@click.argument("topic")
@click.option(
    "--payload",
    "-p",
    default=None,
    help="JSON payload string. Reads from stdin if omitted.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print topic and payload without connecting to Kafka.",
)
@click.option(
    "--envelope",
    is_flag=True,
    default=False,
    help="Wrap payload in ONEX command envelope (correlation_id, requested_by, timestamp).",
)
@click.option(
    "--requested-by",
    default="onex-cli",
    show_default=True,
    help="Value for the requested_by field when --envelope is set.",
)
def produce(
    topic: str,
    payload: str | None,
    dry_run: bool,
    envelope: bool,
    requested_by: str,
) -> None:
    """Publish a JSON payload to a Kafka topic.

    TOPIC is the full Kafka topic name.

    \b
    Examples:
        onex kafka produce onex.cmd.deploy.rebuild-requested.v1 \\
            --payload '{"scope":"full","services":[],"git_ref":"origin/main"}' \\
            --envelope

        onex kafka produce onex.cmd.test.v1 --payload '{}' --dry-run
    """
    if payload is None:
        if sys.stdin.isatty():
            raise click.UsageError("--payload is required when stdin is a terminal.")
        payload = sys.stdin.read().strip()

    try:
        payload_dict: dict[str, object] = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON payload: {exc}") from exc

    final_payload = (
        _build_envelope(payload_dict, requested_by) if envelope else payload_dict
    )
    final_json = json.dumps(final_payload, indent=2)

    click.echo(f"Topic:   {topic}")
    click.echo(f"Payload: {final_json}")

    if dry_run:
        click.echo("(dry-run: skipping Kafka publish)")
        return

    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap_servers:
        raise click.ClickException(
            "KAFKA_BOOTSTRAP_SERVERS is not set. "
            "Set it to your Redpanda address, e.g. 192.168.86.201:19092"  # kafka-fallback-ok: example in error message, not used as default
        )

    async def _publish() -> None:
        producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers, acks="all")
        await producer.start()
        try:
            await producer.send_and_wait(topic, final_json.encode())
        finally:
            await producer.stop()

    try:
        asyncio.run(_publish())
    except Exception as exc:
        raise click.ClickException(f"Kafka publish failed: {exc}") from exc

    click.echo(f"Published to {topic}")
