# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Low-watermark commit positions for a concurrently dispatched subscription.

OMN-20117. A subscription that declares ``consume_concurrency`` (OMN-18852)
keeps several records in flight across polls. Under ``enable_auto_commit`` the
client commits the FETCH position on its own cadence, and after a poll that
position is past every record still running. A redeploy that stops the process
while a delegation is running therefore loses the command: the restarted
consumer resumes past it, and the caller waits out its window for a reply that
nobody is producing. Measured on the ``.201`` dev lane on 2026-09-30: the
delegate-skill group's committed offset moved from 4847 to 4853 across a warm
restart while six commands were still in flight, and none of them was
redelivered.

This ledger is the position the concurrent driver commits instead. For each
partition it is the LOWEST of:

* the lowest offset still in flight (a record that has not finished is never
  committed past, so a stop or a crash redelivers it);
* the rewind point a failed record asked for (OMN-15232/OMN-18852: a record
  whose DLQ copy is unconfirmed must be refetched);
* one past the highest offset dispatched (everything below it has finished).

It is pure bookkeeping with no I/O, so the commit decision can be tested
without a broker.
"""

from __future__ import annotations

from collections import Counter

__all__ = ["ConcurrentCommitLedger"]


class ConcurrentCommitLedger:
    """Tracks dispatched and finished offsets per partition.

    In-flight offsets are counted, not merely recorded, because a rejoined
    replacement consumer resumes from the committed offset and can deliver a
    record again while the first delivery of it is still running. Each
    delivery finishes on its own, and the position must stay pinned until the
    last of them has.
    """

    __slots__ = ("_committed", "_in_flight", "_next")

    def __init__(self) -> None:
        self._in_flight: dict[int, Counter[int]] = {}
        self._next: dict[int, int] = {}
        self._committed: dict[int, int] = {}

    def dispatched(self, partition: int, offset: int) -> None:
        """Record that ``offset`` has been handed to a handler."""
        self._in_flight.setdefault(partition, Counter())[offset] += 1
        self._next[partition] = max(self._next.get(partition, offset + 1), offset + 1)

    def finished(self, partition: int, offset: int) -> None:
        """Record that one delivery of ``offset`` has settled."""
        running = self._in_flight.get(partition)
        if running is None or running[offset] <= 0:
            return
        running[offset] -= 1
        if running[offset] <= 0:
            del running[offset]

    def in_flight(self, partition: int) -> int:
        """Deliveries of ``partition`` that have not settled."""
        running = self._in_flight.get(partition)
        return 0 if running is None else sum(running.values())

    def rebase(self, partition: int, offset: int) -> None:
        """Move the dispatch frontier back to ``offset`` after a rewind seek.

        Called with nothing in flight: the driver drains before it seeks. The
        records from ``offset`` on are fetched again, so the frontier restarts
        there rather than at the pre-rewind high-water mark, which would let
        the next commit skip them.
        """
        self._next[partition] = offset

    def position(
        self, partition: int, *, rewind_floor: int | None = None
    ) -> int | None:
        """The offset that may be committed for ``partition``, or None.

        None means nothing has been dispatched on the partition yet, so there
        is nothing this ledger can vouch for.
        """
        frontier = self._next.get(partition)
        if frontier is None:
            return None
        candidates = [frontier]
        running = self._in_flight.get(partition)
        if running:
            candidates.append(min(running))
        if rewind_floor is not None:
            candidates.append(rewind_floor)
        return min(candidates)

    def advanced(
        self, *, rewind_floors: dict[int, int] | None = None
    ) -> dict[int, int]:
        """Partitions whose committable position moved past the last commit."""
        floors = rewind_floors or {}
        moved: dict[int, int] = {}
        for partition in self._next:
            position = self.position(partition, rewind_floor=floors.get(partition))
            if position is None:
                continue
            if position > self._committed.get(partition, -1):
                moved[partition] = position
        return moved

    def committed(self, partition: int, offset: int) -> None:
        """Record a commit the broker accepted."""
        self._committed[partition] = max(self._committed.get(partition, -1), offset)
