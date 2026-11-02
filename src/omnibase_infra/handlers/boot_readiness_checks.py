# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot roster, boot bound and join priority checks (OMN-18087).

The runtime policy contract declares main's boot-to-ready bound, join limit and
join priority, and the compose files render them into the process environment.
These functions read those values and check the boot against them. They observe
only: they never relax the readiness gate, drop a required topic, or stop the
process.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.models.model_boot_roster_partition import ModelBootRosterPartition

if TYPE_CHECKING:
    from omnibase_infra.runtime.auto_wiring.report import ModelContractWiringResult

logger = logging.getLogger(__name__)

READY_BOUND_POLL_SECONDS: float = 30.0
# The moment this module was imported: the process start the boot bound counts from.
PROCESS_START_MONOTONIC: float = time.monotonic()

ReadinessCheck = Callable[[], Awaitable[dict[str, object]]]


def parse_priority_contracts(env: Mapping[str, str]) -> tuple[str, ...]:
    """Read the rendered priority list, dropping blanks and repeats."""
    raw = env.get("ONEX_BOOT_PRIORITY_CONTRACTS", "")
    return tuple(dict.fromkeys(n.strip() for n in raw.split(",") if n.strip()))


def parse_ready_bound_seconds(env: Mapping[str, str]) -> float | None:
    """Read the rendered boot bound; unset or invalid declares no bound."""
    try:
        value = float(env.get("ONEX_BOOT_READY_BOUND_SECONDS") or math.nan)
    except ValueError:
        return None
    return value if value > 0.0 and math.isfinite(value) else None


def required_topics_without_partitions(
    event_bus_readiness: Mapping[str, object],
) -> tuple[str, ...]:
    """Name the readiness-required topics that hold no partition assignment.

    Reads the dumped ``ModelEventBusReadiness``. A required topic absent from
    ``assignments`` or mapped to an empty list is named; no topic is dropped
    from the required set.
    """
    required = event_bus_readiness.get("required_topics")
    assigned = event_bus_readiness.get("assignments")
    if not isinstance(required, list | tuple) or not isinstance(assigned, Mapping):
        return ()
    return tuple(sorted(str(t) for t in required if not assigned.get(t)))


def verify_roster_partition(
    roster: Collection[str],
    attach_results: Sequence[ModelContractAttachResult],
    exclusions: Sequence[ModelContractAttachExclusion],
) -> ModelBootRosterPartition:
    """Check that every roster contract is attempted or excluded exactly once."""
    names = set(roster)
    attempted = {r.contract_name for r in attach_results}
    excluded = {e.contract_name for e in exclusions}
    return ModelBootRosterPartition(
        roster=tuple(sorted(names)),
        attempted=tuple(sorted(attempted)),
        excluded=tuple(sorted(excluded)),
        unaccounted=tuple(sorted(names - attempted - excluded)),
        in_both=tuple(sorted(attempted & excluded)),
        foreign=tuple(sorted(attempted.union(excluded) - names)),
    )


def report_roster_partition(
    report_results: Sequence[ModelContractWiringResult],
    attach_results: Sequence[ModelContractAttachResult],
    exclusions: Sequence[ModelContractAttachExclusion],
) -> ModelBootRosterPartition:
    """Log the roster partition, at ERROR naming each violating contract."""
    partition = verify_roster_partition(
        [r.contract_name for r in report_results], attach_results, exclusions
    )
    if partition.is_partition:
        logger.info(
            "Boot roster partition: roster=%d attempted=%d excluded=%d (OMN-18087)",
            len(partition.roster),
            len(partition.attempted),
            len(partition.excluded),
        )
    else:
        logger.error(
            "Boot roster partition VIOLATED: %s (OMN-18087)",
            partition.describe_violations(),
        )
    return partition


async def watch_ready_bound(
    readiness_check: ReadinessCheck,
    *,
    bound_seconds: float,
    started_monotonic: float,
    poll_seconds: float = READY_BOUND_POLL_SECONDS,
) -> bool:
    """Poll ``/ready`` until it answers 200 and report whether that was in bound.

    The bound counts from *started_monotonic*. Past it the watch logs once, at
    ERROR, what still blocks readiness, naming the required topics that hold no
    partition, and keeps polling, so a late ready is logged with its real time.
    A poll that raises is logged with its traceback and polled again; it never ends the
    watch.

    Returns:
        True when readiness arrived inside the bound.
    """
    deadline = started_monotonic + bound_seconds
    exceeded_logged = False
    while True:
        try:
            result = await readiness_check()
        except Exception:  # boundary: a failed poll is retried
            logger.exception(
                "Boot bound watch: readiness check raised; polling again (OMN-18087)"
            )
            result = {}
        elapsed = time.monotonic() - started_monotonic
        if result.get("ready") is True:
            logger.info(
                "Runtime ready %.1f s after process start, bound=%.0f s "
                "within_bound=%s (OMN-18087)",
                elapsed,
                bound_seconds,
                elapsed <= bound_seconds,
            )
            return elapsed <= bound_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0 and not exceeded_logged:
            exceeded_logged = True
            _log_bound_exceeded(result, elapsed, bound_seconds)
        await asyncio.sleep(
            min(poll_seconds, remaining) if remaining > 0 else poll_seconds
        )


def _log_bound_exceeded(
    result: Mapping[str, object], elapsed: float, bound_seconds: float
) -> None:
    bus = result.get("event_bus_readiness")
    supplemental = result.get("supplemental_readiness")
    logger.error(
        "Boot bound exceeded: not ready %.1f s after process start, "
        "bound=%.0f s, is_running=%s, required topics without "
        "partitions=%s, supplemental not ready=%s (OMN-18087)",
        elapsed,
        bound_seconds,
        result.get("is_running"),
        list(
            required_topics_without_partitions(bus) if isinstance(bus, Mapping) else ()
        ),
        sorted(
            name
            for name, detail in (
                supplemental.items() if isinstance(supplemental, Mapping) else ()
            )
            if isinstance(detail, Mapping) and not detail.get("ready")
        ),
    )


def start_ready_bound_watch(
    readiness_check: ReadinessCheck,
    env: Mapping[str, str],
    stops_out: list[Callable[[], Awaitable[None]]],
) -> None:
    """Start the bound watch when the policy declares a bound; else do nothing.

    The watch's stop callable is appended to *stops_out* for shutdown.
    """
    bound = parse_ready_bound_seconds(env)
    if bound is None:
        return
    task = asyncio.create_task(
        watch_ready_bound(
            readiness_check,
            bound_seconds=bound,
            started_monotonic=PROCESS_START_MONOTONIC,
        ),
        name="boot-ready-bound-watch",
    )

    async def stop() -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # boundary: shutdown runs every stop
            logger.exception("Boot bound watch ended with an error (OMN-18087)")

    stops_out.append(stop)
