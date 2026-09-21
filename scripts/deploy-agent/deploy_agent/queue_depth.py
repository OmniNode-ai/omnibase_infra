# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What is ahead of a command in this agent's queue, and how fast it drains (OMN-18144).

WHY THE VERIFY JOB CANNOT ANSWER THIS FOR ITSELF
------------------------------------------------
The post-merge lab-pass guard has to decide how long to wait for a lane. Until
now it decided with a clock: a fixed window, later anchored to the agent's
ACCEPTANCE of the command (OMN-18573). Anchoring fixed the half after
acceptance. It left the half before it — the wait for the agent to reach this
command at all — bounded by nothing but the CI job's own ceiling.

Measured 2026-09-18 (report ``dev-lane-agent-dispatch-diag-1700``): four
runtime merges landed inside 33 minutes against an agent servicing roughly 32
minutes per command. The fourth merge's command sat at control-topic offset 293
with a consumer lag of 2, its ``/job/<correlation_id>`` answered 404 for the
whole verify window, and the compose-dev receipt for ``11e8951f`` was minted
FAIL with ``deployed_revision`` INDETERMINATE — against a lane that was healthy
and strictly monotone the entire time. A manual re-run after the agent caught
up passed on the first attempt.

Nothing reachable from the CI runner could have told that job it was third in
line. The agent's job records are on this host's disk and its consumer position
is inside this process, so the agent is the only party that can say it, which
is why this endpoint exists rather than a cleverer CI-side inference.

THE DEPTH IS TWO NUMBERS, AND EITHER ALONE UNDERCOUNTS
-------------------------------------------------------
* ``store_depth`` — job records this agent has ACCEPTED and not finished. A
  command the agent has not consumed yet has no record at all, so this misses
  every queued command.
* ``control_topic_lag`` — records on the control topic past the agent's
  committed offset. A job already in flight has been committed past, so this
  misses the one command that is actually running.

``commands_ahead`` is their sum. The 2026-09-18 stall needed both halves: one
in flight (store) and two unconsumed (lag).

WHY LAG IS MEASURED FROM THE COMMITTED OFFSET, NOT THE FETCH POSITION
----------------------------------------------------------------------
``poll_and_accept`` processes the FIRST record of a batch and returns, leaving
the rest buffered, so the consumer's fetch ``position`` is already past records
the agent has not looked at. Measuring against it would report a lag of zero
while two commands waited — the same off-by-a-batch that OMN-18613 found in
``_commit_through``, seen from the other side. The committed offset advances
only past a record this agent has processed or deliberately refused, so it is
the one that answers "what has this agent not dealt with yet".

WHAT AN UNREADABLE HALF DOES
-----------------------------
It is reported as unreadable, never as zero. An unread queue and an empty queue
must not be the same value: the whole defect being repaired is a guard treating
an absence of evidence as evidence of absence. ``ModelControlTopicLag`` carries
a ``value`` OR a ``reason`` and refuses to carry both or neither, and the CI
reader falls back to its previous behaviour by name when it sees one.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
It observes. Nothing here cancels, reorders, coalesces or times out a command:
a reader that could curtail a rebuild to make its own window fit would
manufacture exactly the false FAIL this ticket removes. Coalescing duplicate
rebuilds of an identical ref is OMN-18143 and is owned there.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, Field, model_validator

from deploy_agent.job_state import JobState, JobStore

#: Job statuses that mean the agent still owes this command work. Mirrors the
#: ``Literal`` on :class:`~deploy_agent.job_state.JobState.status`; pinned by
#: ``test_queue_depth_omn18144.py`` so a new status cannot be added there
#: without deciding which side of this line it falls on.
NON_TERMINAL_STATUSES: Final = frozenset({"accepted", "in_progress"})

#: Job statuses that are terminal but did NOT occupy the agent (OMN-18143).
#:
#: A ``superseded`` record is born terminal in the same millisecond it is
#: written: the command was folded into a newer one and never ran. Counting it
#: in the rolling mean below would divide real service time by a number of
#: jobs that includes refusals, and the bound OMN-18144 derives from that mean
#: -- commands ahead multiplied by mean service time -- would collapse toward
#: zero exactly when the queue is deepest, which is the one moment it matters.
#: Measured shape of the hazard: a night with forty merges and one running job
#: produces thirty-nine sub-second records against ten real ones.
#:
#: They stay OUT of ``NON_TERMINAL_STATUSES`` as well, because the agent owes
#: them no further work; they are terminal, just not service.
UNSERVICED_TERMINAL_STATUSES: Final = frozenset({"superseded"})

#: How many recent completed jobs the mean service time is taken over. Not a
#: tuning knob standing in for a measurement: it is the window over which "how
#: long does this agent take per command" is a question with a current answer.
#: Too long and a host that has since got faster still reads slow; too short
#: and one outlier is the whole estimate. Ten is roughly a working day of
#: rebuilds on this lane.
DEFAULT_SERVICE_SAMPLE_SIZE: Final = 10

#: How old a control-topic lag sample may be and still be served as a number
#: (OMN-18990).
#:
#: The lag was sampled ONLY inside ``poll_and_accept`` until this bound
#: existed, and this agent's run loop does not poll while it executes a
#: command. So for the whole 20-40 minutes of a rebuild the endpoint served
#: the sample taken BEFORE that rebuild started, as a confident integer, with
#: nothing on the wire saying how old it was. That window is exactly the window
#: in which later merges publish commands and queue up: the measurement was
#: blind during the only period it exists to measure.
#:
#: Measured 2026-09-21. Receipt ``lab-pass-receipt-compose-dev-430ff3434cc3...``
#: (artifact 10632895303) reported ``commands_ahead=0`` at 09:18:50Z while two
#: commands sat unconsumed, derived a 1560s wait bound from that zero, and gave
#: up at 0h26m. The agent accepted the receipt's own command at 10:01:09Z,
#: 42m21s after the probe began and 5m30s after the receipt was written.
#:
#: 120s is two of the refresher's own intervals, so a single missed refresh
#: does not flip a healthy agent to unreadable, and a refresher that has died
#: is caught within two minutes rather than being believed for half an hour.
MAX_LAG_SAMPLE_AGE_SECONDS: Final = 120.0


class ModelControlTopicLag(BaseModel):
    """Records on the control topic this agent has not dealt with, or why unknown.

    Exactly one of ``value`` and ``reason``. Both, or neither, would make "the
    queue is empty" and "nobody could look" the same answer, which is the
    collapse this module exists to undo.
    """

    model_config = {"frozen": True}

    value: int | None = None
    reason: str = ""
    #: Which offset the lag was measured against: ``committed`` is the real
    #: answer, ``position`` is the fetch-position fallback used before this
    #: consumer has committed anything, and it is named rather than hidden
    #: because it can under-report by a batch.
    basis: str = ""
    observed_at: datetime | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ModelControlTopicLag:
        if (self.value is None) == (not self.reason):
            msg = (
                "a control-topic lag carries EXACTLY one of a value and a "
                f"reason; got value={self.value!r}, reason={self.reason!r}"
            )
            raise ValueError(msg)
        if self.value is not None and self.value < 0:
            msg = f"lag cannot be negative, got {self.value}"
            raise ValueError(msg)
        if self.value is not None and not self.basis:
            msg = (
                "a lag with a value must name the offset basis it was measured against"
            )
            raise ValueError(msg)
        # OMN-18990. A count whose age cannot be established is not a count.
        # The staleness bound one class down is only enforceable if every
        # valued sample says when it was taken, and the failure this repairs
        # was a half-hour-old zero that looked exactly like a fresh one.
        if self.value is not None and self.observed_at is None:
            msg = (
                "a lag with a value must carry the time it was observed; "
                "an undateable count cannot be shown to be current"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def unknown(cls, reason: str) -> ModelControlTopicLag:
        return cls(value=None, reason=reason)


class ModelQueueSnapshot(BaseModel):
    """One look at what this agent owes, and how fast it has been paying it off."""

    model_config = {"frozen": True}

    observed_at: datetime
    #: The correlation id the agent is executing right now, if any.
    in_flight_correlation_id: str | None = None
    #: Non-terminal job records. Includes the in-flight one.
    store_depth: int = Field(ge=0)
    control_topic_lag: ModelControlTopicLag
    #: Mean wall-clock seconds from acceptance to completion over the sample.
    #: ``None`` when this agent has completed nothing it can measure — never a
    #: zero, which would derive a bound of no time at all.
    mean_service_time_seconds: float | None = None
    service_sample_size: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _service_time_is_evidenced(self) -> ModelQueueSnapshot:
        if (self.mean_service_time_seconds is None) != (self.service_sample_size == 0):
            msg = (
                "a mean service time and its sample size stand or fall "
                f"together; got mean={self.mean_service_time_seconds!r}, "
                f"sample_size={self.service_sample_size}"
            )
            raise ValueError(msg)
        if (
            self.mean_service_time_seconds is not None
            and self.mean_service_time_seconds <= 0
        ):
            msg = (
                "a mean service time must be positive; a zero would derive a "
                "queue bound of no time at all"
            )
            raise ValueError(msg)
        return self

    @property
    def lag_age_seconds(self) -> float | None:
        """Seconds between the lag sample and this snapshot, or ``None``.

        ``None`` only where the lag carries no value at all; a valued lag is
        required by its own model to carry an observation time.
        """
        observed = self.control_topic_lag.observed_at
        if observed is None:
            return None
        return (self.observed_at - observed).total_seconds()

    @property
    def lag_staleness_reason(self) -> str:
        """Why this lag sample is too old to serve as a count, or ``""``.

        OMN-18990. The sampler is a cache written by the consumer's poll, and
        the run loop does not poll while a rebuild executes. An old sample is
        an UNREAD queue, not an empty one -- the same collapse this module's
        docstring refuses for an unreadable half, seen a second time from the
        time axis rather than the readability one.
        """
        if self.control_topic_lag.value is None:
            # Already unreadable, and it carries its own reason. Two reasons
            # for one absence would make the payload ambiguous about which
            # applied.
            return ""
        age = self.lag_age_seconds
        if age is None:  # pragma: no cover - the model forbids this pairing
            return (
                "this lag sample carries no observation time, so its age "
                "cannot be established and it is not treated as current"
            )
        if age > MAX_LAG_SAMPLE_AGE_SECONDS:
            return (
                f"this lag sample was observed {age:.0f}s ago, beyond the "
                f"{MAX_LAG_SAMPLE_AGE_SECONDS:.0f}s bound, so it describes the "
                "queue as it was and not as it is -- an unread queue, which is "
                "not an empty one"
            )
        return ""

    @property
    def commands_ahead(self) -> int | None:
        """Commands a newly published one must wait behind, or ``None`` if unknown.

        Unknown when the lag half could not be read: the store half alone
        cannot see an unconsumed command, and reporting it as the total would
        under-count exactly the case this endpoint exists for.

        Unknown ALSO when the lag half is stale (OMN-18990). A sample taken
        before the running rebuild started answers a question about a different
        moment, and serving it as this moment's count is how a zero survived
        two queued commands.
        """
        if self.control_topic_lag.value is None or self.lag_staleness_reason:
            return None
        return self.store_depth + self.control_topic_lag.value

    def to_payload(self) -> dict[str, object]:
        """The JSON the agent serves. Flat and explicit; no client re-derivation."""
        return {
            "observed_at": self.observed_at.isoformat(),
            "in_flight_correlation_id": self.in_flight_correlation_id,
            "store_depth": self.store_depth,
            "control_topic_lag": self.control_topic_lag.value,
            # OMN-18990. Exactly one of the two reasons is ever non-empty, so
            # a reader never has to decide which absence it is looking at.
            "control_topic_lag_reason": (
                self.control_topic_lag.reason or self.lag_staleness_reason
            ),
            "control_topic_lag_basis": self.control_topic_lag.basis,
            # The sample's OWN time, not the snapshot's. The snapshot is always
            # fresh -- it is built per request -- so a reader that had only
            # `observed_at` above saw a current timestamp beside a half-hour-old
            # count and had no way to tell.
            "control_topic_lag_observed_at": (
                self.control_topic_lag.observed_at.isoformat()
                if self.control_topic_lag.observed_at is not None
                else None
            ),
            "control_topic_lag_age_seconds": self.lag_age_seconds,
            "commands_ahead": self.commands_ahead,
            "mean_service_time_seconds": self.mean_service_time_seconds,
            "service_sample_size": self.service_sample_size,
        }


class LagSampler:
    """The consumer's latest lag observation, handed across a thread boundary.

    The consumer runs on the agent's worker pool and the HTTP handler on the
    event loop, so this is the seam between them. It holds a value under a lock
    and nothing else: no kafka client reference, because a client touched from
    the loop thread while the worker is polling it is a data race, and no
    fallback, because an unsampled lag is unknown rather than zero.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = ModelControlTopicLag.unknown(
            "the consumer has not completed a poll in this process yet, so its "
            "control-topic lag has never been sampled"
        )
        self._committed: dict[object, int] = {}

    def record(self, lag: ModelControlTopicLag) -> None:
        with self._lock:
            self._latest = lag

    def latest(self) -> ModelControlTopicLag:
        with self._lock:
            return self._latest

    def note_commit(self, partition: object, offset: int) -> None:
        """Remember the offset this agent has committed through on a partition.

        The sampler owns this rather than the consumer because it is the only
        thing that reads it: a lag measured against anything else is the
        off-by-a-batch this module's docstring describes, and keeping the two
        together means a commit site cannot advance the offset without the
        measurement following it.

        A self-update rewind moves the value BACKWARDS, deliberately: the
        rewound record is queued again and the lag must say so.
        """
        with self._lock:
            self._committed[partition] = offset

    def committed(self, partition: object) -> int | None:
        with self._lock:
            return self._committed.get(partition)


def _service_seconds(job: JobState) -> float | None:
    """Seconds this job took, or ``None`` when its own record cannot say.

    A completed job whose ``completed_at`` precedes its ``accepted_at`` is a
    clock artefact, not a fast rebuild, and contributing a negative to a mean
    would make the queue look faster than it is.
    """
    if job.completed_at is None:
        return None
    seconds = (job.completed_at - job.accepted_at).total_seconds()
    return seconds if seconds > 0 else None


def mean_service_time(
    jobs: list[JobState], sample_size: int = DEFAULT_SERVICE_SAMPLE_SIZE
) -> tuple[float | None, int]:
    """Rolling mean over the most recently SERVICED jobs.

    Failed jobs count. The question the bound asks is "how long until the agent
    reaches my command", and a command that failed after 30 minutes occupied
    the agent for 30 minutes exactly as a successful one did.

    Superseded jobs do NOT count, for the same reason and in the other
    direction: they occupied the agent for no time at all. See
    ``UNSERVICED_TERMINAL_STATUSES``.
    """
    completed = sorted(
        (
            job
            for job in jobs
            if job.completed_at is not None
            and job.status not in UNSERVICED_TERMINAL_STATUSES
        ),
        key=lambda job: job.completed_at,  # type: ignore[arg-type,return-value]
        reverse=True,
    )[:sample_size]
    durations = [
        seconds for job in completed if (seconds := _service_seconds(job)) is not None
    ]
    if not durations:
        return None, 0
    return sum(durations) / len(durations), len(durations)


def load_jobs(store: JobStore) -> list[JobState]:
    """Every job record this store can currently parse.

    An unparseable record is skipped rather than fatal, matching what the
    health handler already does with the same glob: one corrupt file must not
    take out the surface that reports the queue.
    """
    jobs: list[JobState] = []
    for path in sorted(store.state_dir.glob("*.json")):
        try:
            jobs.append(JobState.model_validate_json(path.read_text()))
        except Exception:  # noqa: BLE001 - a corrupt record is not this surface's business
            continue
    return jobs


def compute_queue_snapshot(
    store: JobStore,
    lag: ModelControlTopicLag,
    *,
    now: datetime | None = None,
    sample_size: int = DEFAULT_SERVICE_SAMPLE_SIZE,
) -> ModelQueueSnapshot:
    """Read the queue off this agent's own durable state plus the sampled lag."""
    jobs = load_jobs(store)
    non_terminal = [job for job in jobs if job.status in NON_TERMINAL_STATUSES]
    active = store.load_active()
    mean, sampled = mean_service_time(jobs, sample_size=sample_size)
    return ModelQueueSnapshot(
        observed_at=now or datetime.now(UTC),
        in_flight_correlation_id=str(active.correlation_id) if active else None,
        store_depth=len(non_terminal),
        control_topic_lag=lag,
        mean_service_time_seconds=mean,
        service_sample_size=sampled,
    )
