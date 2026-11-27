# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reading of a boot demand measure (OMN-18087, OMN-24223, OMN-25856).

Moved out of ``handler_kafka_topic_demand_source``, whose baseline only shrinks.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping

from aiokafka.structs import TopicPartition

__all__: list[str] = ["log_demand_measure", "split_in_use"]


def log_demand_measure(
    logger: logging.Logger,
    topics: Collection[str],
    retained: Collection[str],
    readable: Collection[str],
) -> None:
    """Log the split's counts in one line (OMN-24223).

    A topic is retained (holds a record), unreadable (missing or not visible) or
    idle and watched; ``retained + unreadable`` is the number in use. The line
    starts with the words the boot log is searched for.
    """
    logger.info(
        "Boot subscription demand measure: topics=%d retained=%d unreadable=%d "
        "idle_watched=%d (OMN-24223)",
        len(topics),
        len(retained),
        len(set(topics) - set(readable)),
        len(set(readable) - set(retained)),
    )


def split_in_use(
    logical_of: Mapping[str, str],
    with_records: Collection[str],
    readable: Collection[str],
    watch_retained: Collection[str],
    measured: Collection[TopicPartition],
) -> tuple[frozenset[str], list[TopicPartition]]:
    """Return the in-use contract topics and the partitions to watch.

    A retained topic the caller asked to watch (OMN-25856) is watched from its high
    watermark and is not in use; an unreadable topic is always in use.
    """
    keeps_boot = {p for p in with_records if logical_of[p] not in watch_retained}
    in_use = frozenset(
        logical
        for physical, logical in logical_of.items()
        if physical in keeps_boot or physical not in readable
    )
    return in_use, [tp for tp in measured if tp.topic not in keeps_boot]
