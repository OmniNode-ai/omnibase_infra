# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""CLI command for manual artifact reconciliation.

Publishes a ModelManualReconcileCommand to ``onex.cmd.artifact.reconcile.v1``
via confluent-kafka.  Consumed by HandlerManualTrigger in the
node_artifact_change_detector_effect node.

Usage:
    omni-infra artifact-reconcile --repo omnibase_infra
    omni-infra artifact-reconcile --repo omnibase_infra --files src/foo.py --files src/bar.py
    omni-infra artifact-reconcile --repo omnibase_infra --reason "Post-migration check"
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import socket
from uuid import uuid4

import click
from rich.console import Console

from omnibase_infra.enums.generated.enum_artifact_topic import EnumArtifactTopic

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_BOOTSTRAP_SERVERS = "localhost:19092"


def _get_bootstrap_servers() -> str:
    """Resolve Kafka bootstrap servers from environment."""
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS)


def _resolve_actor() -> str:
    """Best-effort resolve a human-readable actor identity."""
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    return f"{user}@{host}"


@click.command("artifact-reconcile")
@click.option(
    "--repo",
    required=True,
    help="Repository to reconcile, e.g. 'omnibase_infra'.",
)
@click.option(
    "--files",
    multiple=True,
    help=(
        "File path(s) to restrict reconciliation scope. "
        "May be specified multiple times. "
        "Omit to trigger a full-repo reconciliation."
    ),
)
@click.option(
    "--reason",
    default="",
    show_default=False,
    help="Human-readable reason for the manual trigger.",
)
def artifact_reconcile_cmd(
    repo: str,
    files: tuple[str, ...],
    reason: str,
) -> None:
    """Trigger a manual artifact reconciliation.

    Publishes a ModelManualReconcileCommand to the Kafka topic
    ``onex.cmd.artifact.reconcile.v1``.  The Change Detector EFFECT node
    consumes the command and kicks off the reconciliation pipeline.

    When no --files are provided the downstream COMPUTE node treats this as a
    full-repo reconciliation and matches all registered artifacts.

    \b
    Examples:
      omni-infra artifact-reconcile --repo omnibase_infra
      omni-infra artifact-reconcile --repo omnibase_infra --files src/foo.py
      omni-infra artifact-reconcile --repo omnibase_infra --reason "Post-migration check"
    """
    from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_manual_reconcile_command import (
        ModelManualReconcileCommand,
    )

    command_id = uuid4()
    actor = _resolve_actor()
    changed_files = list(files)

    command = ModelManualReconcileCommand(
        command_id=command_id,
        source_repo=repo,
        changed_files=changed_files,
        actor=actor,
        reason=reason,
    )

    bootstrap_servers = _get_bootstrap_servers()

    console.print(
        f"[bold blue]Publishing artifact-reconcile command[/bold blue] "
        f"(id={command_id}, repo={repo}, files={len(changed_files)}, "
        f"bus={bootstrap_servers})"
    )

    _publish_command(command, bootstrap_servers)

    console.print(
        f"[bold green]Published to {EnumArtifactTopic.CMD_RECONCILE_V1.value}[/bold green] — "
        f"command_id={command_id}"
    )


def _publish_command(
    command: object,
    bootstrap_servers: str,
) -> None:
    """Publish command to Kafka synchronously.

    Args:
        command: ModelManualReconcileCommand instance (serialised to JSON).
        bootstrap_servers: Kafka bootstrap servers string.

    Raises:
        click.ClickException: On producer creation or delivery failure.
    """
    try:
        from confluent_kafka import Producer
    except ImportError as exc:
        raise click.ClickException(
            "confluent-kafka is required. Install it with: pip install confluent-kafka"
        ) from exc

    from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_manual_reconcile_command import (
        ModelManualReconcileCommand,
    )

    assert isinstance(command, ModelManualReconcileCommand)

    payload = json.dumps(command.model_dump(mode="json")).encode("utf-8")
    key = str(command.command_id).encode("utf-8")

    delivery_error: list[Exception] = []

    def _on_delivery(err: object, msg: object) -> None:
        if err is not None:
            delivery_error.append(click.ClickException(f"Kafka delivery failed: {err}"))

    try:
        producer = Producer({"bootstrap.servers": bootstrap_servers})
    except Exception as exc:
        raise click.ClickException(
            f"Failed to create Kafka producer for {bootstrap_servers}: {exc}"
        ) from exc

    try:
        producer.produce(
            topic=EnumArtifactTopic.CMD_RECONCILE_V1.value,
            key=key,
            value=payload,
            on_delivery=_on_delivery,
        )
        producer.flush(timeout=10.0)
    except Exception as exc:
        raise click.ClickException(f"Failed to publish command: {exc}") from exc

    if delivery_error:
        raise delivery_error[0]
