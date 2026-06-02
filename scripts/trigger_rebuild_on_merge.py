#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# trigger_rebuild_on_merge.py
#
# Publishes onex.cmd.omnimarket.redeploy-start.v1 (consumed by node_redeploy)
# when a merged PR contains runtime changes. Called from the
# runtime-rebuild-trigger GHA workflow on PR merge to dev or main.
#
# node_redeploy owns the deployment lifecycle (lane policy, digest pinning,
# readiness, rollback) and is the SOLE emitter of
# onex.cmd.deploy.rebuild-requested.v1 to the deploy agent. CI publishes a typed
# start command only; it never talks to the deploy agent directly.
#
# Triggers when:
#   - PR had the "runtime_change" label, OR
#   - Any changed file matches src/omnimarket/** or src/omnibase_infra/nodes/**
#
# Lane policy (the triggering ref decides the lane — no hardcoded origin/main):
#   - merge to dev  -> runtime_lane=dev,            source_branch=dev
#   - merge to main -> runtime_lane=stability-test, source_branch=main
#     (dev->main promotion proves the stability lane; prod deploys the
#      stability-proven digest later via node_redeploy, not from CI)
#
# Tickets: OMN-8917 (original auto-trigger), OMN-12573 (re-point to node_redeploy)
#
# Required environment variables (when not --dry-run):
#   KAFKA_BOOTSTRAP_SERVERS   -- broker address(es), e.g. host:9092
#   KAFKA_SASL_USERNAME       -- SASL username / API key
#   KAFKA_SASL_PASSWORD       -- SASL password / API secret
#   DEPLOY_AGENT_HMAC_SECRET  -- HMAC secret for payload signing
#
# Usage:
#   python scripts/trigger_rebuild_on_merge.py \
#     --changed-files "src/omnimarket/nodes/foo/handler.py,README.md" \
#     --labels "runtime_change,bug" \
#     --base-branch "dev" \
#     --source-sha "<merge_commit_sha>" \
#     [--dry-run]

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import os
import sys
import uuid
from datetime import UTC, datetime

import click

# CI publishes the node_redeploy start command; node_redeploy is the sole
# emitter of the deploy-agent rebuild command downstream.
TOPIC = "onex.cmd.omnimarket.redeploy-start.v1"

_RUNTIME_PATH_PATTERNS = [
    "src/omnimarket/*",
    "src/omnibase_infra/nodes/*",
]

_RUNTIME_LABEL = "runtime_change"

# Maps the merged PR's base branch to a node_redeploy runtime lane. Values match
# deploy_agent.events.EnumRuntimeLane (dev | stability-test | prod). prod is not
# triggerable from CI: production deploys the stability-proven digest through
# node_redeploy's promotion gate, never from a merge event.
_BASE_BRANCH_LANES: dict[str, str] = {
    "dev": "dev",
    "main": "stability-test",
}


def should_trigger(changed_files: list[str], labels: list[str]) -> bool:
    """Return True if a rebuild should be triggered."""
    if _RUNTIME_LABEL in labels:
        return True
    for f in changed_files:
        for pattern in _RUNTIME_PATH_PATTERNS:
            if fnmatch.fnmatch(f, pattern) or f.startswith(pattern.rstrip("*")):
                return True
    return False


def lane_for_base_branch(base_branch: str) -> str:
    """Map a merged PR's base branch to a node_redeploy runtime lane.

    Fails closed on unmapped branches: a misconfigured trigger must not silently
    pick a default lane and rebuild the wrong runtime.
    """
    lane = _BASE_BRANCH_LANES.get(base_branch)
    if lane is None:
        allowed = ", ".join(sorted(_BASE_BRANCH_LANES))
        msg = (
            f"No runtime lane mapping for base branch {base_branch!r}; "
            f"allowed base branches: {allowed}"
        )
        raise ValueError(msg)
    return lane


def _sign_envelope(envelope: dict[str, object], secret: str) -> dict[str, object]:
    body_dict = {k: v for k, v in envelope.items() if k != "_signature"}
    body = json.dumps(body_dict, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {**envelope, "_signature": signature}


def publish_redeploy_start_event(
    bootstrap_servers: str,
    username: str,
    password: str,
    hmac_secret: str,
    runtime_lane: str,
    source_branch: str,
    source_sha: str,
    correlation_id: str,
    requested_by: str,
) -> None:
    """Publish a signed redeploy-start command to node_redeploy via SASL_SSL."""
    from confluent_kafka import Producer

    envelope = {
        "correlation_id": correlation_id,
        "requested_by": requested_by,
        "runtime_lane": runtime_lane,
        "source_branch": source_branch,
        "source_sha": source_sha,
        # dev dogfoods OCC drafting; stability gates on readiness before prod.
        "requires_occ": True,
        "requires_readiness_gate": runtime_lane != "dev",
        "requested_at": datetime.now(UTC).isoformat(),
    }
    signed = _sign_envelope(envelope, hmac_secret)

    producer_config: dict[str, str | int | float | bool] = {
        "bootstrap.servers": bootstrap_servers,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": username,
        "sasl.password": password,
    }
    producer = Producer(producer_config)

    delivery_error: BaseException | None = None

    def _on_delivery(err: object, _msg: object) -> None:
        nonlocal delivery_error
        if err is not None:
            delivery_error = RuntimeError(str(err))

    message = json.dumps(signed, default=str).encode("utf-8")
    key = f"gha-redeploy/{correlation_id}".encode()

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
    "--changed-files",
    default="",
    help="Comma-separated list of changed file paths",
)
@click.option(
    "--labels",
    default="",
    help="Comma-separated list of PR label names",
)
@click.option(
    "--base-branch",
    required=True,
    help="Merged PR base branch (dev | main) — decides the runtime lane",
)
@click.option(
    "--source-sha",
    required=True,
    help="Merge commit SHA of the triggering PR (the ref node_redeploy rebuilds)",
)
@click.option(
    "--requested-by",
    default="gha-runtime-rebuild-trigger",
    help="Identifier for who is requesting the redeploy",
)
@click.option(
    "--correlation-id",
    default="",
    help="Correlation ID (auto-generated if not provided)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Check trigger conditions and print decision without publishing",
)
def main(
    changed_files: str,
    labels: str,
    base_branch: str,
    source_sha: str,
    requested_by: str,
    correlation_id: str,
    dry_run: bool,
) -> None:
    """Publish a node_redeploy start command if a PR contains runtime changes.

    Triggers when PR had the runtime_change label OR changed files match
    src/omnimarket/** or src/omnibase_infra/nodes/**. The triggering base branch
    decides the runtime lane; the merge SHA is the ref node_redeploy rebuilds.
    """
    files: list[str] = (
        [f.strip() for f in changed_files.split(",") if f.strip()]
        if changed_files
        else []
    )
    label_list: list[str] = (
        [lb.strip() for lb in labels.split(",") if lb.strip()] if labels else []
    )

    corr_id = correlation_id or str(uuid.uuid4())

    runtime_lane = lane_for_base_branch(base_branch)

    if not should_trigger(files, label_list):
        click.echo(
            "No rebuild trigger: no runtime_change label or runtime path changes detected."
        )
        sys.exit(0)

    click.echo(
        f"Redeploy triggered: runtime_lane={runtime_lane} source_branch={base_branch} "
        f"source_sha={source_sha} correlation_id={corr_id} labels={label_list} "
        f"files_matched={[f for f in files if any(f.startswith(p.rstrip('*')) for p in _RUNTIME_PATH_PATTERNS)]}"
    )

    if dry_run:
        click.echo("(dry-run: skipping Kafka publish)")
        sys.exit(0)

    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
    username = os.environ.get("KAFKA_SASL_USERNAME", "")
    password = os.environ.get("KAFKA_SASL_PASSWORD", "")
    hmac_secret = os.environ.get("DEPLOY_AGENT_HMAC_SECRET", "")

    if not bootstrap_servers:
        click.echo("KAFKA_BOOTSTRAP_SERVERS is not set -- skipping publish")
        sys.exit(0)
    if not username or not password:
        click.echo("KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD must be set", err=True)
        sys.exit(1)
    if not hmac_secret:
        click.echo("DEPLOY_AGENT_HMAC_SECRET must be set", err=True)
        sys.exit(1)

    try:
        publish_redeploy_start_event(
            bootstrap_servers=bootstrap_servers,
            username=username,
            password=password,
            hmac_secret=hmac_secret,
            runtime_lane=runtime_lane,
            source_branch=base_branch,
            source_sha=source_sha,
            correlation_id=corr_id,
            requested_by=requested_by,
        )
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Delivery error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Published redeploy-start to {TOPIC} (correlation_id={corr_id})")


if __name__ == "__main__":
    main()
