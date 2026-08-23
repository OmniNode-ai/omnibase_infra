# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Path-verifying canary probe for the gateway container healthcheck (OMN-15741).

Replaces ``test -f /tmp/gateway-forwarder-ready`` (a sentinel written once at
startup and never re-verified) with a real produce+readback round trip against
each broker leg, using the exact transport and credentials real gateway
traffic uses (``KafkaTransport`` over the resolved ``local_bus``/``cloud_bus``
config). This is the failure mode of the 2026-08-04 outage: the process was up
and the ready-file was present for four days while the cloud leg forwarded 0%
-- a liveness signal that cannot see past "the process started once."

The probe checks the local leg and the cloud leg independently and reports
each leg's outcome on its own line, so ``docker inspect ...State.Health.Log``
distinguishes "local leg healthy, cloud leg dead" from "both legs healthy" --
the acceptance criterion this ticket names explicitly. Overall exit is
non-zero if either leg fails.

To avoid spamming either broker, a real check only runs once every
contract-declared ``canary.cadence_seconds``; between real checks this process
reports the last real result from ``--state-file`` (default
``/tmp/gateway-canary-probe-state.json``), refreshed on every real run. This
process never depends on the long-running forwarder process being alive or
reachable -- it dials the brokers directly, so a healthy forwarder container
whose cloud leg is actually dead is caught even though ``docker ps`` and the
old ready-file check would both call it fine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
)
from omnibase_infra.runtime.gateway_forwarder import (
    load_gateway_forwarder_runtime_config,
)
from omnibase_infra.topics.model_topic_provisioning_policy import (
    ModelTopicProvisioningPolicy,
)

logger = logging.getLogger(__name__)

_CANARY_HEADER_NAME = "canary-correlation-id"
_POLL_SLICE_MS = 2000


class ModelCanaryLegResult(NamedTuple):
    """Outcome of one leg's real produce+readback attempt."""

    leg: str
    passed: bool
    detail: str


async def _ensure_topic_exists(
    bus_config: ModelKafkaEventBusConfig,
    topic: str,
    *,
    admin_factory: Callable[..., AIOKafkaAdminClient] = AIOKafkaAdminClient,
) -> None:
    """Best-effort idempotent creation of the dedicated canary topic.

    OMN-15810 / OMN-16420: on a lane where the canary topic was never
    provisioned, every real check failed with "not found in cluster
    metadata" -- a permanent false negative on an otherwise-healthy leg. The
    canary topic never carries real gateway traffic (it is scratch space
    dedicated to this probe alone), so provisioning it here -- rather than
    routing through the runtime's full contract-driven ``TopicProvisioner``
    -- keeps this probe a standalone process that dials brokers directly,
    matching the rest of this module's design.

    Provisioning failure is never fatal to the probe: if the topic genuinely
    cannot be created (e.g. insufficient broker permissions, or a cloud
    cluster's replication-factor policy rejects ``replication_factor=1``),
    the caller's own produce/readback attempt fails right after this with a
    clear connect/produce error -- that is the real, honest signal.
    """
    admin = admin_factory(
        bootstrap_servers=bus_config.bootstrap_servers,
        **build_aiokafka_auth_kwargs(bus_config),
    )
    try:
        await admin.start()
        # OMN-15395: replication factor must resolve through the policy seam,
        # never a hardcoded literal (the canary topic has no owning contract,
        # so `declared=None` -- the policy still resolves it: RF2 on a managed
        # MSK cluster, RF1 on an unmeasured self-hosted broker).
        policy = ModelTopicProvisioningPolicy.from_kafka_config(bus_config)
        replication_factor = policy.resolve_replication_factor(
            topic=topic, declared=None
        )
        await admin.create_topics(
            [
                NewTopic(
                    name=topic,
                    num_partitions=1,
                    replication_factor=replication_factor,
                )
            ],
        )
    except TopicAlreadyExistsError:
        pass
    except Exception:  # noqa: BLE001 -- best-effort; the leg check below is the real signal
        logger.warning("canary probe: failed to ensure topic %s exists", topic)
    finally:
        try:
            await admin.close()
        except Exception:  # noqa: BLE001 -- cleanup must not mask the real result
            logger.warning("canary probe: admin client close failed for %s", topic)


async def check_canary_leg(
    *,
    leg: str,
    bus_config: ModelKafkaEventBusConfig,
    topic: str,
    canary: ModelGatewayCanaryConfig,
    transport_factory: type[KafkaTransport] = KafkaTransport,
    admin_factory: Callable[..., AIOKafkaAdminClient] = AIOKafkaAdminClient,
) -> ModelCanaryLegResult:
    """Produce one tiny canary record to ``topic`` and confirm readback.

    Uses a fresh, unique consumer group so every invocation joins at
    ``auto_offset_reset="latest"`` -- the probe only ever waits for the record
    it just produced, never replays history, so repeated runs cannot build up
    unbounded lag on the dedicated canary topic.
    """
    await _ensure_topic_exists(bus_config, topic, admin_factory=admin_factory)
    correlation_id = uuid4().hex.encode("ascii")
    group = f"gateway-canary-probe-{leg}-{uuid4().hex}"
    transport = transport_factory(
        config=bus_config,
        group=group,
        topics=(topic,),
        auto_offset_reset="latest",
    )
    started = False
    try:
        try:
            await asyncio.wait_for(
                transport.start(), timeout=canary.produce_deadline_seconds
            )
        except Exception as exc:  # noqa: BLE001 -- any failure here means the leg is dead
            return ModelCanaryLegResult(
                leg=leg, passed=False, detail=f"{leg} leg connect failed: {exc}"
            )
        started = True

        try:
            await asyncio.wait_for(
                transport.send(
                    topic,
                    None,
                    correlation_id,
                    {_CANARY_HEADER_NAME: correlation_id},
                ),
                timeout=canary.produce_deadline_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- any failure here means the leg is dead
            return ModelCanaryLegResult(
                leg=leg, passed=False, detail=f"{leg} leg produce failed: {exc}"
            )

        deadline_at = time.monotonic() + canary.readback_deadline_seconds
        while time.monotonic() < deadline_at:
            remaining_ms = max(int((deadline_at - time.monotonic()) * 1000), 1)
            try:
                messages = await transport.poll(
                    max_messages=32,
                    timeout_ms=min(remaining_ms, _POLL_SLICE_MS),
                )
            except Exception as exc:  # noqa: BLE001 -- any failure here means the leg is dead
                return ModelCanaryLegResult(
                    leg=leg, passed=False, detail=f"{leg} leg readback failed: {exc}"
                )
            for message in messages:
                if message.value == correlation_id:
                    return ModelCanaryLegResult(
                        leg=leg,
                        passed=True,
                        detail=f"{leg} leg produce+readback confirmed",
                    )
        return ModelCanaryLegResult(
            leg=leg,
            passed=False,
            detail=(
                f"{leg} leg readback timed out after "
                f"{canary.readback_deadline_seconds}s -- canary record was "
                "produced but never read back"
            ),
        )
    finally:
        if started:
            try:
                await transport.close()
            except Exception:  # noqa: BLE001 -- cleanup must not mask the real result
                logger.warning("canary probe: %s leg transport close failed", leg)


async def run_canary_check(
    config: ModelGatewayForwarderRuntimeConfig,
) -> tuple[ModelCanaryLegResult, ModelCanaryLegResult]:
    """Check the local leg and cloud leg concurrently; return both outcomes.

    The cloud-leg wire topic carries the tenant prefix, matching exactly what
    real outbound traffic does (``prefix_topic``) -- the probe is not a
    parallel code path, it drives the same transform real traffic drives.
    """
    canary = config.forwarder.canary
    tenant_slug = config.forwarder.tenant_identity.tenant_slug
    local_result, cloud_result = await asyncio.gather(
        check_canary_leg(
            leg="local",
            bus_config=config.local_bus,
            topic=canary.topic,
            canary=canary,
        ),
        check_canary_leg(
            leg="cloud",
            bus_config=config.cloud_bus,
            topic=prefix_topic(tenant_slug, canary.topic),
            canary=canary,
        ),
    )
    return local_result, cloud_result


def _load_cached_passing_state(state_path: Path, cadence_seconds: int) -> str | None:
    """Return the cached report only if it was a PASS still within cadence.

    OMN-16420: the prior version of this helper also served cached FAILURES,
    prefixed with a ``CANARY_STALE_FAILURE_CACHED`` marker, for the entire
    cadence window. Docker calls the healthcheck far more often than
    ``cadence_seconds``, so once one real check failed, every subsequent tick
    replayed that same stale failure verbatim -- a live recovery (e.g. a
    transient broker blip clearing) stayed invisible in the health log until
    the cache happened to expire. A cached PASS is still safe to replay (it
    only ever avoids redundant spam on the already-healthy path); a cached
    FAIL is never replayed -- ``probe()`` always runs a fresh real check
    instead, so the reported verdict is never staler than "right now."
    """
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    checked_at = raw.get("checked_at")
    passed = raw.get("passed")
    report = raw.get("report")
    if not isinstance(checked_at, (int, float)) or passed is not True:
        return None
    if not isinstance(report, str):
        return None
    if time.time() - checked_at >= cadence_seconds:
        return None
    return report


def _write_state(
    state_path: Path, *, passed: bool, report: str, checked_at: float
) -> None:
    payload = {"checked_at": checked_at, "passed": passed, "report": report}
    try:
        state_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        # State-file persistence is a spam-avoidance optimization, not a
        # correctness requirement -- a write failure must not turn a real
        # PASS into a probe crash.
        logger.warning("canary probe: failed to persist state to %s", state_path)


async def probe(
    config: ModelGatewayForwarderRuntimeConfig,
    *,
    state_path: Path,
    force: bool = False,
) -> tuple[bool, str]:
    """Return ``(passed, report)``, consulting/refreshing the cadence cache.

    Only a cached PASS may be served early; a cached FAIL always triggers an
    immediate fresh real check (OMN-16420 -- see ``_load_cached_passing_state``).
    """
    canary = config.forwarder.canary
    if not force:
        cached = _load_cached_passing_state(state_path, canary.cadence_seconds)
        if cached is not None:
            return True, cached

    local_result, cloud_result = await run_canary_check(config)
    passed = local_result.passed and cloud_result.passed
    report = "\n".join(
        f"{'PASS' if result.passed else 'FAIL'}: {result.detail}"
        for result in (local_result, cloud_result)
    )
    _write_state(state_path, passed=passed, report=report, checked_at=time.time())
    return passed, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Path-verifying canary healthcheck: produce+readback a canary "
            "record across the local and cloud gateway legs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the resolved, typed gateway forwarder YAML",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("/tmp/gateway-canary-probe-state.json"),  # noqa: S108 -- container-local scratch state
        help="Cadence cache so repeated healthcheck ticks do not spam the brokers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the cadence cache and always run a real produce+readback check",
    )
    parser.add_argument(
        "--broker-ref-map",
        type=Path,
        required=True,
        help=(
            "Path to the operator-supplied broker-ref resolution map (OMN-15743). "
            "Same file the gateway-forwarder process itself is given -- required, "
            "no default, since load_gateway_forwarder_runtime_config fails closed "
            "without it"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint. Exits non-zero when either leg's path is dead."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_parser().parse_args(argv)
    config = load_gateway_forwarder_runtime_config(
        args.config,
        broker_ref_map_path=args.broker_ref_map,
    )
    passed, report = asyncio.run(
        probe(config, state_path=args.state_file, force=args.force)
    )
    print(report)  # noqa: T201 -- captured by `docker inspect ...State.Health.Log`
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
