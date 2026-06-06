# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: T201, BLE001
"""CLI for runner health collection and alerting.

Usage:
    uv run python -m omnibase_infra.observability.runner_health.cli_runner_health [FLAGS]

Flags:
    --json      Print snapshot as JSON to stdout
    --emit      Emit snapshot to Kafka topic (best-effort)
    --alert     Send Slack alert if any runners are degraded (best-effort)
    --network   Run the bounded Docker network janitor + subnet-pool collection.
                Honours --emit (network-pool-status topic) and --alert (Slack
                pool-pressure alert). Dry-run unless --reclaim is also passed.
    --reclaim   With --network, actually remove reclaim-eligible networks
                (declared-ownership, idle, aged-out only). Without it the
                janitor runs read-only.
    --host      Override RUNNER_HEALTH_HOST env var

Environment:
    RUNNER_FLEET_CONFIG_PATH      Path to config/runner_fleet.yaml
    RUNNER_HEALTH_HOST            Optional CI host override
    RUNNER_HEALTH_GITHUB_ORG      Optional GitHub org override
    RUNNER_HEALTH_RUNNER_PREFIX   Optional runner name prefix override
    KAFKA_BOOTSTRAP_SERVERS       Kafka brokers for --emit
    SLACK_BOT_TOKEN               Slack bot token for --alert
    SLACK_CHANNEL_ID              Slack channel for --alert
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from uuid import UUID, uuid4

from omnibase_infra.observability.runner_health.collector_network_pool import (
    CollectorNetworkPool,
)
from omnibase_infra.observability.runner_health.collector_runner_health import (
    CollectorRunnerHealth,
)
from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.janitor_docker_network import (
    JanitorDockerNetwork,
)
from omnibase_infra.observability.runner_health.model_network_janitor_result import (
    ModelNetworkJanitorResult,
)
from omnibase_infra.observability.runner_health.model_network_pool_alert import (
    ModelNetworkPoolAlert,
    build_pool_alert_if_pressured,
)
from omnibase_infra.observability.runner_health.model_network_pool_status import (
    ModelNetworkPoolStatus,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)
from omnibase_infra.observability.runner_health.model_runner_health_alert import (
    ModelRunnerHealthAlert,
)
from omnibase_infra.observability.runner_health.model_runner_health_snapshot import (
    ModelRunnerHealthSnapshot,
)


async def main(args: list[str]) -> int:
    """Run runner health collection with optional Kafka emit and Slack alert."""
    try:
        fleet_config = load_runner_fleet_config()
    except Exception as exc:
        print(f"[runner-health] ERROR: {exc}")
        return 1

    # CLI flag overrides for env vars
    host = os.environ.get("RUNNER_HEALTH_HOST", fleet_config.runner_host)
    for i, arg in enumerate(args):
        if arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
    if not host:
        print(
            "[runner-health] ERROR: RUNNER_HEALTH_HOST not set and --host not provided."
        )
        return 1
    github_org = os.environ.get("RUNNER_HEALTH_GITHUB_ORG", fleet_config.github_org)
    runner_prefix = os.environ.get(
        "RUNNER_HEALTH_RUNNER_PREFIX", fleet_config.runner_name_prefix
    )

    collector = CollectorRunnerHealth(
        github_org=github_org,
        runner_host=host,
        runner_count=fleet_config.expected_count,
        runner_prefix=runner_prefix,
    )

    correlation_id = uuid4()
    snapshot = await collector.collect(correlation_id=correlation_id)

    # Run the bounded network janitor + subnet-pool collection first so its
    # evidence can ride the SAME Kafka producer and Slack session as the
    # runner-health snapshot/alert (single credential read per transport).
    network_events: tuple[tuple[str, dict[str, object], str], ...] = ()
    pool_pressure_message = ""
    if "--network" in args:
        network_events, pool_pressure_message = await _run_network_pass(
            host=host,
            correlation_id=correlation_id,
            pool_capacity=fleet_config.network_pool_capacity,
            warn_ratio=fleet_config.network_pool_warn_ratio,
            reclaim="--reclaim" in args,
        )

    if "--json" in args:
        print(snapshot.model_dump_json(indent=2))

    if "--emit" in args:
        await _emit_to_kafka(snapshot, extra_events=network_events)

    if "--alert" in args and (snapshot.degraded_count > 0 or pool_pressure_message):
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
        await _send_slack_alert(
            alert,
            extra_message=pool_pressure_message,
            extra_correlation_id=correlation_id,
        )
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


async def _run_network_pass(
    host: str,
    correlation_id: UUID,
    pool_capacity: int,
    warn_ratio: float,
    reclaim: bool,
) -> tuple[tuple[tuple[str, dict[str, object], str], ...], str]:
    """Run the bounded network janitor + subnet-pool collection on ``host``.

    The janitor is read-only unless ``reclaim`` is True. Returns:
      * the Kafka events (topic, value, key) to publish via the shared producer
        so subnet-pool + janitor evidence is durable on the runner surface, and
      * a Slack pool-pressure message (empty unless the pool crosses the
        pre-exhaustion threshold).

    Pool capacity and warn ratio come from the typed runner-fleet config
    (contract-config, not env vars).
    """
    from omnibase_infra.topics.platform_topic_suffixes import (
        SUFFIX_NETWORK_POOL_STATUS,
    )

    janitor = JanitorDockerNetwork(runner_host=host)
    janitor_result = await janitor.run(
        correlation_id=correlation_id, dry_run=not reclaim
    )
    mode = "RECLAIM" if reclaim else "DRY-RUN"
    print(
        f"[network-janitor] {mode}: "
        f"{janitor_result.reclaim_candidate_count} reclaimable, "
        f"{janitor_result.preserved_count} preserved, "
        f"{len(janitor_result.reclaimed)} removed."
    )
    for err in janitor_result.reclaim_errors:
        print(f"[network-janitor] removal error: {err}")

    pool_collector = CollectorNetworkPool(
        runner_host=host,
        pool_capacity=pool_capacity,
        warn_threshold_ratio=warn_ratio,
    )
    pool_status = await pool_collector.collect()
    print(
        f"[network-pool] {pool_status.network_count}/{pool_status.pool_capacity} "
        f"networks (remaining {pool_status.remaining_capacity}, "
        f"over_threshold={pool_status.is_over_threshold})"
    )

    events: tuple[tuple[str, dict[str, object], str], ...] = (
        (
            SUFFIX_NETWORK_POOL_STATUS,
            pool_status.model_dump(mode="json"),
            pool_status.host,
        ),
        (
            SUFFIX_NETWORK_POOL_STATUS,
            janitor_result.model_dump(mode="json"),
            janitor_result.host,
        ),
    )

    pool_alert = build_pool_alert_if_pressured(
        pool_status,
        correlation_id=correlation_id,
        reclaim_candidate_count=janitor_result.reclaim_candidate_count,
    )
    pressure_message = ""
    if pool_alert is not None:
        pressure_message = pool_alert.to_slack_message()
    else:
        print("[network-pool] Below threshold. No alert.")

    return events, pressure_message


async def _emit_to_kafka(
    snapshot: ModelRunnerHealthSnapshot,
    extra_events: tuple[tuple[str, dict[str, object], str], ...] = (),
) -> None:
    """Emit snapshot to Kafka. Best-effort -- does not fail the CLI.

    ``extra_events`` (topic, value, key) ride the SAME producer + bootstrap
    read so the network-pool/janitor evidence (OMN-12566) is published without
    a second KAFKA_BOOTSTRAP_SERVERS read or a second connection.
    """
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
            for topic, value, key in extra_events:
                await producer.send_and_wait(topic, value=value, key=key.encode())
                print(f"[network-pool] Event emitted to Kafka ({topic}).")
        finally:
            await producer.stop()
    except Exception as e:
        print(f"[runner-health] Kafka emit failed (non-fatal): {e}")


async def _send_slack_alert(
    alert: ModelRunnerHealthAlert,
    extra_message: str = "",
    extra_title: str = "",
    extra_correlation_id: UUID | None = None,
) -> None:
    """Send alert to Slack via existing Slack webhook. Best-effort.

    ``extra_message`` (with ``extra_title``/``extra_correlation_id``) sends an
    additional alert through the SAME session + credential read so the
    subnet-pool pressure alert (OMN-12566) needs no second SLACK token read.
    """
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
            if alert.degraded_runners:
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
            if extra_message:
                pool_slack = ModelSlackAlert(
                    severity=EnumAlertSeverity.WARNING,
                    message=extra_message,
                    title=extra_title or "Docker Subnet Pool Pressure",
                    correlation_id=extra_correlation_id or alert.correlation_id,
                )
                pool_result = await slack.handle(pool_slack)
                if pool_result.success:
                    print("[network-pool] Slack alert sent.")
                else:
                    print(f"[network-pool] Slack alert failed: {pool_result.error}")
    except Exception as e:
        print(f"[runner-health] Slack alert failed (non-fatal): {e}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
