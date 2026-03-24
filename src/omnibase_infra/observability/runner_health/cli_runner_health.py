# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: T201, BLE001
"""CLI for runner health collection and alerting.

Usage:
    uv run python -m omnibase_infra.observability.runner_health.cli_runner_health [FLAGS]

Flags:
    --json    Print snapshot as JSON to stdout
    --emit    Emit snapshot to Kafka topic (best-effort)
    --alert   Send Slack alert if any runners are degraded (best-effort)
    --host    Override RUNNER_HEALTH_HOST env var

Environment:
    RUNNER_HEALTH_HOST            CI host address (required if --host not given)
    RUNNER_HEALTH_GITHUB_ORG      GitHub org (default: OmniNode-ai)
    RUNNER_HEALTH_EXPECTED_COUNT  Expected runner count (default: 10)
    KAFKA_BOOTSTRAP_SERVERS       Kafka brokers for --emit
    SLACK_BOT_TOKEN               Slack bot token for --alert
    SLACK_CHANNEL_ID              Slack channel for --alert
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from uuid import uuid4

from omnibase_infra.observability.runner_health.collector_runner_health import (
    CollectorRunnerHealth,
)
from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.model_runner_health_alert import (
    ModelRunnerHealthAlert,
)
from omnibase_infra.observability.runner_health.model_runner_health_snapshot import (
    ModelRunnerHealthSnapshot,
)

# Config from environment -- no hardcoded lab values
GITHUB_ORG = os.environ.get("RUNNER_HEALTH_GITHUB_ORG", "OmniNode-ai")
RUNNER_HOST = os.environ.get("RUNNER_HEALTH_HOST", "")
RUNNER_COUNT = int(os.environ.get("RUNNER_HEALTH_EXPECTED_COUNT", "10"))


async def main(args: list[str]) -> int:
    """Run runner health collection with optional Kafka emit and Slack alert."""
    # CLI flag overrides for env vars
    host = RUNNER_HOST
    for i, arg in enumerate(args):
        if arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
    if not host:
        print(
            "[runner-health] ERROR: RUNNER_HEALTH_HOST not set and --host not provided."
        )
        return 1

    collector = CollectorRunnerHealth(
        github_org=GITHUB_ORG,
        runner_host=host,
        runner_count=RUNNER_COUNT,
    )

    correlation_id = uuid4()
    snapshot = await collector.collect(correlation_id=correlation_id)

    if "--json" in args:
        print(snapshot.model_dump_json(indent=2))

    if "--emit" in args:
        await _emit_to_kafka(snapshot)

    if "--alert" in args and snapshot.degraded_count > 0:
        degraded = tuple(
            r for r in snapshot.runners if r.state != EnumRunnerHealthState.HEALTHY
        )
        alert = ModelRunnerHealthAlert(
            correlation_id=correlation_id,
            degraded_runners=degraded,
            total_runners=snapshot.expected_runners,
            healthy_count=snapshot.healthy_count,
            host=snapshot.host,
        )
        await _send_slack_alert(alert)
    elif "--alert" in args:
        print(
            f"[runner-health] All {snapshot.expected_runners} runners healthy. No alert."
        )

    if not any(f in args for f in ("--json", "--emit", "--alert")):
        # Default: print summary
        print(
            f"Runner Health: {snapshot.healthy_count}/{snapshot.expected_runners} healthy"
        )
        for r in snapshot.runners:
            marker = (
                "[ok]" if r.state == EnumRunnerHealthState.HEALTHY else "[DEGRADED]"
            )
            print(
                f"  {marker} {r.name}: {r.state.value} "
                f"(GH:{r.github_status} Docker:{r.docker_status})"
            )
        if snapshot.host_disk_percent >= 70:
            print(f"  [WARN] Host disk: {snapshot.host_disk_percent:.0f}%")

    return 0


async def _emit_to_kafka(snapshot: ModelRunnerHealthSnapshot) -> None:
    """Emit snapshot to Kafka. Best-effort -- does not fail the CLI."""
    try:
        from aiokafka import AIOKafkaProducer

        from omnibase_infra.topics.platform_topic_suffixes import (
            SUFFIX_RUNNER_HEALTH_SNAPSHOT,
        )

        bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if not bootstrap:
            print("[runner-health] KAFKA_BOOTSTRAP_SERVERS not set. Skipping emit.")
            return

        producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await producer.start()
        try:
            await producer.send_and_wait(
                SUFFIX_RUNNER_HEALTH_SNAPSHOT,
                value=snapshot.model_dump(mode="json"),
                key=snapshot.host.encode(),
            )
            print("[runner-health] Snapshot emitted to Kafka.")
        finally:
            await producer.stop()
    except Exception as e:
        print(f"[runner-health] Kafka emit failed (non-fatal): {e}")


async def _send_slack_alert(alert: ModelRunnerHealthAlert) -> None:
    """Send alert to Slack via existing Slack webhook. Best-effort."""
    try:
        import aiohttp

        from omnibase_infra.handlers.handler_slack_webhook import (
            HandlerSlackWebhook,
        )
        from omnibase_infra.handlers.models.enum_alert_severity import (
            EnumAlertSeverity,
        )
        from omnibase_infra.handlers.models.model_slack_alert_payload import (
            ModelSlackAlert,
        )

        bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
        channel_id = os.environ.get("SLACK_CHANNEL_ID", "")
        if not bot_token or not channel_id:
            print(
                "[runner-health] SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set. "
                "Skipping alert."
            )
            return

        async with aiohttp.ClientSession() as session:
            slack = HandlerSlackWebhook(
                http_session=session,
                bot_token=bot_token,
                default_channel=channel_id,
            )
            slack_alert = ModelSlackAlert(
                severity=EnumAlertSeverity.WARNING,
                message=alert.to_slack_message(),
                title="Runner Health Alert",
                correlation_id=alert.correlation_id,
            )
            result = await slack.handle(slack_alert)
            if result.success:
                print("[runner-health] Slack alert sent.")
            else:
                print(f"[runner-health] Slack alert failed: {result.error}")
    except Exception as e:
        print(f"[runner-health] Slack alert failed (non-fatal): {e}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
