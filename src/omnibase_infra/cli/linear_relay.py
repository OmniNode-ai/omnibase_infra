# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""onex-linear-relay CLI — relay Linear workspace snapshots to Kafka.

Reads a Linear snapshot JSON file and publishes a ``ModelLinearSnapshotEvent``
to ``onex.evt.linear.snapshot.v1``.

Usage
-----
    onex-linear-relay emit --snapshot-file /tmp/linear-snapshot.json

Non-Blocking Design
-------------------
If Kafka is unreachable within 2 seconds, the CLI exits 0 and spools the
event to ``~/.onex/spool/linear-snapshots.jsonl`` for deferred delivery.

Kafka-First Design
------------------
This CLI is the **primary** ingress path for Linear snapshot data into the
ONEX event bus. The ``POST /api/linear/snapshot`` endpoint in omnidash is
debug-only and MUST NOT become a primary ingress path.

Topic
-----
``onex.evt.linear.snapshot.v1``
Partition key: ``snapshot_id`` (UUID4, generated per invocation)

Related Tickets:
    - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import click

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Topic constant (mirrors omnibase_core TOPIC_LINEAR_SNAPSHOT_EVENT from PR#531)
TOPIC_LINEAR_SNAPSHOT_EVENT = "onex.evt.linear.snapshot.v1"

# Spool file location for non-blocking fallback
_SPOOL_DIR = Path.home() / ".onex" / "spool"
_SPOOL_FILE = _SPOOL_DIR / "linear-snapshots.jsonl"

# Kafka publish timeout in seconds
_KAFKA_TIMEOUT_SECONDS = 2.0


def _spool_event(event: JsonType) -> None:
    """Write event to spool file for deferred delivery."""
    try:
        _SPOOL_DIR.mkdir(parents=True, exist_ok=True)
        with _SPOOL_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        logger.info("Linear snapshot event spooled to %s", _SPOOL_FILE)
    except OSError as exc:
        logger.warning("Failed to spool linear snapshot event: %s", exc)


async def _publish_event(
    event: JsonType,
    partition_key: str,
    bootstrap_servers: str,
) -> bool:
    """Publish event to Kafka with a 2-second timeout.

    Returns True on success, False if Kafka is unreachable or times out.
    """
    try:
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        bus = EventBusKafka.default()
        await asyncio.wait_for(bus.start(), timeout=_KAFKA_TIMEOUT_SECONDS)
        try:
            success = await asyncio.wait_for(
                bus.publish(
                    topic=TOPIC_LINEAR_SNAPSHOT_EVENT,
                    message=json.dumps(event).encode("utf-8"),
                    key=partition_key.encode("utf-8"),
                ),
                timeout=_KAFKA_TIMEOUT_SECONDS,
            )
            return bool(success)
        finally:
            await bus.close()
    except TimeoutError:
        logger.warning(
            "Kafka publish timed out after %.1fs — spooling event",
            _KAFKA_TIMEOUT_SECONDS,
        )
        return False
    except Exception as exc:
        logger.warning("Kafka publish failed: %s — spooling event", exc)
        return False


def _build_event(
    snapshot_data: dict[str, JsonType],
    snapshot_id: str,
) -> JsonType:
    """Build the event payload dict matching ModelLinearSnapshotEvent schema.

    The ``workstreams`` field is extracted from the snapshot data if present.
    All other snapshot data is embedded under the ``snapshot`` key for
    downstream consumers.

    Args:
        snapshot_data: Parsed Linear snapshot JSON.
        snapshot_id: UUID4 string used as partition key and event identifier.

    Returns:
        JsonType event payload dict.
    """
    raw_workstreams = snapshot_data.get("workstreams", [])
    workstreams: list[str] = (
        [str(w) for w in raw_workstreams] if isinstance(raw_workstreams, list) else []
    )

    return {
        "event_type": TOPIC_LINEAR_SNAPSHOT_EVENT,
        "snapshot_id": snapshot_id,
        "workstreams": workstreams,
        "snapshot": snapshot_data,
        "emitted_at": datetime.now(tz=UTC).isoformat(),
    }


@click.group()
@click.option(
    "--bootstrap-servers",
    default=lambda: os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
    show_default=True,
    help="Kafka bootstrap servers (overrides KAFKA_BOOTSTRAP_SERVERS env var).",
)
@click.pass_context
def cli(ctx: click.Context, bootstrap_servers: str) -> None:
    """onex-linear-relay — relay Linear workspace snapshots to Kafka.

    Non-blocking: if Kafka is unreachable within 2s, events are spooled to
    ~/.onex/spool/linear-snapshots.jsonl and the CLI exits 0.

    This is the primary ingress path for Linear data into the ONEX event bus.
    Do NOT use the omnidash /api/linear/snapshot endpoint for production data.
    """
    ctx.ensure_object(dict)
    ctx.obj["bootstrap_servers"] = bootstrap_servers


@cli.command("emit")
@click.option(
    "--snapshot-file",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=Path),
    help="Path to Linear snapshot JSON file.",
)
@click.option(
    "--snapshot-id",
    default=None,
    help="Optional explicit snapshot UUID4 (auto-generated if omitted).",
)
@click.pass_context
def emit(
    ctx: click.Context,
    snapshot_file: Path,
    snapshot_id: str | None,
) -> None:
    """Emit a Linear snapshot event to Kafka.

    Reads the snapshot JSON from --snapshot-file and publishes
    ModelLinearSnapshotEvent to onex.evt.linear.snapshot.v1.

    The snapshot_id is used as the Kafka partition key.
    """
    # Parse snapshot file
    try:
        raw = snapshot_file.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            click.echo("ERROR: snapshot-file must contain a JSON object.", err=True)
            sys.exit(1)
        snapshot_data: dict[str, JsonType] = parsed
    except (json.JSONDecodeError, OSError) as exc:
        click.echo(f"ERROR: Failed to read --snapshot-file: {exc}", err=True)
        sys.exit(1)

    resolved_id = snapshot_id or str(uuid4())
    event = _build_event(snapshot_data, resolved_id)
    partition_key = resolved_id
    bootstrap_servers: str = ctx.obj["bootstrap_servers"]

    success = asyncio.run(_publish_event(event, partition_key, bootstrap_servers))

    if not success:
        _spool_event(event)
        click.echo(
            f"[onex-linear-relay] Kafka unavailable — event spooled to {_SPOOL_FILE}",
            err=True,
        )
    else:
        raw_workstreams = snapshot_data.get("workstreams", [])
        count = len(raw_workstreams) if isinstance(raw_workstreams, list) else 0
        click.echo(
            f"[onex-linear-relay] Published Linear snapshot {resolved_id} "
            f"({count} workstreams)"
        )

    # Always exit 0 — relay must not block callers
    sys.exit(0)


def main() -> None:
    """Entry point for onex-linear-relay CLI."""
    logging.basicConfig(level=logging.WARNING)
    cli()


__all__ = ["cli", "main"]
