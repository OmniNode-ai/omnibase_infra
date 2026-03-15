#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# publish_pr_webhook_event.py
#
# CLI publisher for GitHub PR webhook events.
#
# Constructs a ModelPRWebhookEvent from CLI arguments, validates the payload
# via Pydantic, then publishes the JSON-serialised event to the Kafka topic
# ``onex.evt.github.pr-webhook.v1`` using SASL_SSL authentication.
#
# Ticket: OMN-3946
#
# Invoked by the ``artifact-reconciliation-webhook`` GitHub Actions workflow.
# Can also be run locally for testing (use --dry-run to skip Kafka publish).
#
# Required environment variables (when not --dry-run):
#   KAFKA_BOOTSTRAP_SERVERS  — broker address(es), e.g. host:9092
#   KAFKA_SASL_USERNAME      — SASL username / API key
#   KAFKA_SASL_PASSWORD      — SASL password / API secret
#
# Usage examples:
#
#   # Dry-run (validates payload, prints JSON, skips Kafka)
#   python scripts/publish_pr_webhook_event.py \
#     --action opened \
#     --repo OmniNode-ai/omnibase_infra \
#     --pr-number 42 \
#     --head-ref feature/my-branch \
#     --head-sha abc123def456 \
#     --dry-run
#
#   # Real publish (requires Kafka env vars)
#   python scripts/publish_pr_webhook_event.py \
#     --action closed \
#     --repo OmniNode-ai/omnibase_infra \
#     --pr-number 99 \
#     --head-ref feature/merged \
#     --head-sha abc123def456 \
#     --merged true \
#     --actor johndoe \
#     --changed-files "src/foo.py,src/bar.py"
#
# Exit codes:
#   0  Event published (or dry-run succeeded)
#   1  Validation error or delivery failure

from __future__ import annotations

import json
import os
import re
import sys

import click

TOPIC = "onex.evt.github.pr-webhook.v1"

# Ticket-ID pattern: e.g. OMN-1234, ABC-999
_TICKET_RE = re.compile(r"\b([A-Z]{2,8}-\d+)\b")


def _parse_bool(value: str) -> bool:
    """Parse a truthy/falsy string to bool.

    Accepts: true/false, True/False, 1/0, yes/no.
    """
    return value.strip().lower() in ("true", "1", "yes")


def _extract_ticket_ids(text: str) -> list[str]:
    """Extract Linear ticket IDs from free text."""
    return _TICKET_RE.findall(text)


def _build_event(
    action: str,
    repo: str,
    pr_number: int,
    head_ref: str,
    head_sha: str,
    changed_files: list[str],
    ticket_ids: list[str],
    actor: str | None,
    merged: bool,
) -> dict[str, object]:
    """Validate and serialise the event via ModelPRWebhookEvent."""
    import pathlib
    import sys
    import types

    src_dir = pathlib.Path(__file__).parent.parent / "src"

    # Register lightweight namespace stubs for each intermediate package so
    # that Python's import system can locate model_pr_webhook_event.py without
    # executing any __init__.py that would pull in omnibase_core (not installed
    # in the GitHub Actions environment).  Each stub must expose __path__ so
    # the import machinery treats it as a package and can descend into it.
    node_pkg = "node_artifact_change_detector_effect"
    pkg_paths: dict[str, pathlib.Path] = {
        "omnibase_infra": src_dir / "omnibase_infra",
        "omnibase_infra.nodes": src_dir / "omnibase_infra" / "nodes",
        f"omnibase_infra.nodes.{node_pkg}": (
            src_dir / "omnibase_infra" / "nodes" / node_pkg
        ),
        f"omnibase_infra.nodes.{node_pkg}.models": (
            src_dir / "omnibase_infra" / "nodes" / node_pkg / "models"
        ),
    }
    for pkg, pkg_path in pkg_paths.items():
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
            stub.__package__ = pkg
            sys.modules[pkg] = stub

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_pr_webhook_event import (
        ModelPRWebhookEvent,
    )

    event = ModelPRWebhookEvent(
        action=action,  # type: ignore[arg-type]
        repo=repo,
        pr_number=pr_number,
        head_ref=head_ref,
        head_sha=head_sha,
        changed_files=changed_files,
        ticket_ids=ticket_ids,
        actor=actor or None,
        merged=merged,
    )
    return dict(json.loads(event.model_dump_json()))


def _publish(
    payload: dict[str, object], bootstrap_servers: str, username: str, password: str
) -> None:
    """Publish the event to Kafka using confluent-kafka with SASL_SSL."""
    from confluent_kafka import Producer  # type: ignore[import-untyped]

    producer_config: dict[str, str | int | float | bool] = {
        "bootstrap.servers": bootstrap_servers,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": username,
        "sasl.password": password,
    }

    producer = Producer(producer_config)

    delivery_error: BaseException | None = None

    def _on_delivery(err: object, _msg: object) -> None:  # type: ignore[misc]
        nonlocal delivery_error
        if err is not None:
            delivery_error = RuntimeError(str(err))

    message = json.dumps(payload, default=str).encode("utf-8")
    key = f"{payload['repo']}/pr/{payload['pr_number']}".encode()

    producer.produce(
        topic=TOPIC,
        key=key,
        value=message,
        on_delivery=_on_delivery,
    )
    producer.flush(timeout=30)

    if delivery_error is not None:
        raise RuntimeError(f"Kafka delivery failed: {delivery_error}") from None


@click.command()
@click.option(
    "--action",
    required=True,
    help="GitHub pull_request action (opened, synchronize, closed, reopened)",
)
@click.option(
    "--repo",
    required=True,
    help="Repository full name, e.g. OmniNode-ai/omnibase_infra",
)
@click.option("--pr-number", required=True, type=int, help="Pull request number")
@click.option("--head-ref", required=True, help="Source branch name")
@click.option("--head-sha", required=True, help="Head commit SHA")
@click.option("--actor", default=None, help="GitHub login of the PR actor")
@click.option(
    "--merged", default="false", help="'true' when PR was merged (closed action only)"
)
@click.option(
    "--changed-files",
    default="",
    help="Comma-separated list of changed file paths",
)
@click.option(
    "--pr-title",
    default="",
    help="PR title (used to extract ticket IDs)",
)
@click.option(
    "--pr-body",
    default="",
    help="PR body text (used to extract ticket IDs)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate and print event JSON without publishing",
)
def main(
    action: str,
    repo: str,
    pr_number: int,
    head_ref: str,
    head_sha: str,
    actor: str | None,
    merged: str,
    changed_files: str,
    pr_title: str,
    pr_body: str,
    dry_run: bool,
) -> None:
    """Publish a GitHub PR webhook event to Kafka.

    Validates the event via ModelPRWebhookEvent (Pydantic), then publishes
    JSON to onex.evt.github.pr-webhook.v1 using SASL_SSL authentication.

    When --action is ``closed`` and --merged is falsy, the script exits 0
    without publishing (a PR closed without merge is not interesting for
    artifact reconciliation).
    """
    is_merged = _parse_bool(merged)

    # Skip closed-without-merge events — not interesting for reconciliation.
    if action == "closed" and not is_merged:
        click.echo(f"Skipping closed (not merged) PR #{pr_number} in {repo}")
        sys.exit(0)

    # Parse changed files (comma-separated, strip whitespace, drop empties).
    files: list[str] = (
        [f.strip() for f in changed_files.split(",") if f.strip()]
        if changed_files
        else []
    )

    # Extract ticket IDs from title + body.
    ticket_ids: list[str] = list(
        {*_extract_ticket_ids(pr_title), *_extract_ticket_ids(pr_body)}
    )

    try:
        payload = _build_event(
            action=action,
            repo=repo,
            pr_number=pr_number,
            head_ref=head_ref,
            head_sha=head_sha,
            changed_files=files,
            ticket_ids=ticket_ids,
            actor=actor,
            merged=is_merged,
        )
    except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
        click.echo(f"Validation error: {exc}", err=True)
        sys.exit(1)

    payload_json = json.dumps(payload, indent=2, default=str)
    click.echo(payload_json)

    if dry_run:
        click.echo("(dry-run: skipping Kafka publish)")
        sys.exit(0)

    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
    username = os.environ.get("KAFKA_SASL_USERNAME", "")
    password = os.environ.get("KAFKA_SASL_PASSWORD", "")

    if not bootstrap_servers:
        click.echo("KAFKA_BOOTSTRAP_SERVERS is not set — skipping publish")
        sys.exit(0)
    if not username or not password:
        click.echo("KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD must be set", err=True)
        sys.exit(1)

    try:
        _publish(
            payload,
            bootstrap_servers=bootstrap_servers,
            username=username,
            password=password,
        )
    except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
        click.echo(f"Delivery error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Published to {TOPIC} (PR #{pr_number}, action={action})")


if __name__ == "__main__":
    main()
