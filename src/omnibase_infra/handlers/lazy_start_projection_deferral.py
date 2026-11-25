# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A projection the lazy-start lane left dormant is deferred, not unattached (OMN-26316).

The runtime health monitor reads the manifest of every contract the profile owns
and judges each declared projection against the live bus registry. On a lazy-start
lane (OMN-21592) most contracts are dormant: their consumers are absent from the
registry by design until a first message or request wakes them. Measured on the
dev runtime on 2026-11-24, 37 of the 38 projections without a consumer group were
dormant ones, which held ``projection_attachment`` DEGRADED and the container
unhealthy.

This module is the one place that answers which unattached projections are only
waiting, and renders them. The same file holds the two other readings the monitor
needs for that dimension: the deferral the boot subscription demand split already
grants (OMN-21629) and the complete name lists the capped detail cannot carry.

A dormant projection is deferred only while the front door has neither woken it
nor recorded a failed wake (``LazyStartFrontDoor.awaiting_first_request``): a
woken or failed-to-wake projection is judged by the registry again, like any other.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable

from omnibase_infra.models.health.model_projection_contract_ref import (
    ModelProjectionContractRef,
)
from omnibase_infra.models.health.model_projection_liveness_verdict import (
    ModelProjectionLivenessVerdict,
)

__all__: list[str] = [
    "awaiting_first_request",
    "describe_attachment",
    "split_unattached",
    "without_idle",
]

logger = logging.getLogger(__name__)


def awaiting_first_request() -> frozenset[str]:
    """Dormant contracts no message or request has reached yet."""
    # Imported here: lazy_start_boot_split reaches the wiring seam, which imports
    # the liveness module that imports this one.
    from omnibase_infra.handlers.lazy_start_boot_split import StoreLazyStartDormant

    front_door = StoreLazyStartDormant.front_door()
    # Nothing until the front door exists: a dormant contract nothing can wake yet
    # is not one that waits by design, so it is never excused from the check.
    return frozenset() if front_door is None else front_door.awaiting_first_request


def without_idle(
    projections: Iterable[ModelProjectionContractRef], idle: Collection[str]
) -> tuple[ModelProjectionContractRef, ...]:
    """Drop projections whose every topic a boot demand watch still awaits (OMN-21629)."""
    return tuple(
        ref
        for ref in projections
        if not ref.subscribe_topics or not set(idle).issuperset(ref.subscribe_topics)
    )


def split_unattached(
    projections: Iterable[ModelProjectionContractRef],
    attached_topics: Collection[str],
    awaiting: Collection[str],
) -> tuple[list[str], list[str]]:
    """Names of the projections with an unattached topic, as (unattached, dormant).

    A projection whose every topic has a consumer is in neither. One with a topic
    that has none is dormant when it is among *awaiting*, else unattached. Both
    lists are logged in full.
    """
    unattached: list[str] = []
    dormant: list[str] = []
    for ref in projections:
        if all(topic in attached_topics for topic in ref.subscribe_topics):
            continue
        (dormant if ref.name in awaiting else unattached).append(ref.name)
    # The dimension detail names MAX_NAMED_PROJECTIONS and counts the rest
    # ("+31 more"): 30 of the 38 projections the dev runtime read as unattached
    # were named nowhere. The detail is capped because it ships on every /health
    # response; a log line is not.
    if unattached:
        logger.warning(
            "Runtime health projection_attachment: %d unattached projection(s): %s",
            len(unattached),
            ", ".join(unattached),
        )
    if dormant:
        logger.info(
            "Runtime health projection_attachment: %d dormant projection(s) "
            "awaiting their first request: %s",
            len(dormant),
            ", ".join(dormant),
        )
    return unattached, dormant


def describe_attachment(
    verdict: ModelProjectionLivenessVerdict,
    name_list: Callable[[tuple[str, ...]], str],
) -> str:
    """The ``projection_attachment`` detail of an evaluated verdict."""
    dormant = (
        f"; {len(verdict.dormant_projections)} dormant by lazy start, attached on "
        f"their first request (OMN-26316): {name_list(verdict.dormant_projections)}"
        if verdict.dormant_projections
        else ""
    )
    if verdict.unattached_projections:
        return (
            f"{len(verdict.unattached_projections)}/{verdict.projection_count} "
            "declared projection(s) have no attached consumer and persist nothing: "
            f"{name_list(verdict.unattached_projections)}{dormant}"
        )
    attached = (
        f"{verdict.projection_count - len(verdict.dormant_projections)} of "
        f"{verdict.projection_count}"
        if verdict.dormant_projections
        else f"All {verdict.projection_count}"
    )
    return f"{attached} declared projection(s) attached{dormant}"
