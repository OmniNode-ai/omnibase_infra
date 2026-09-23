# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka consumer with acceptance protocol.

POISON PILLS ARE QUARANTINED, NEVER RE-READ FOREVER (OMN-16442)
---------------------------------------------------------------

``onex.cmd.deploy.rebuild-requested.v1`` is a control topic with one consumer:
this agent. A record it cannot decode used to take the whole control plane
down, because the failure happened *before* the offset commit and redelivery
reproduced it exactly. Measured on the .201 dev lane on 2026-09-08: one record
published with snappy compression by an ``rpk topic produce`` default, against
a ``kafka-python`` client with no snappy codec installed, raised
``UnsupportedCodecError`` on every poll -- twelve crashes in sixty seconds, six
systemd restarts, then the start limit, ``ActiveState=failed``. Nothing on the
host could recover it: every restart re-read the same offset.

Two distinct classes reach the agent, and they surface in different places:

* **Fetch-level** -- the batch cannot be decompressed or is corrupt on the
  wire. ``kafka-python`` raises inside ``poll()``; no record object exists, so
  the offset must be recovered from the consumer's own position.
* **Record-level** -- the bytes decompress but are not a JSON object, or are a
  JSON object that ``ModelRebuildRequested`` refuses. A record object exists
  and names its own partition and offset.

Both are quarantined the same way: a durable local record, a best-effort
dead-letter publish, a CRITICAL log, and **the offset committed past it**.

WHY COMMITTING IS RIGHT HERE, WHERE THE BOUNDARY DLQ WITHHOLDS
--------------------------------------------------------------

``omnibase_infra.runtime.auto_wiring`` deliberately withholds the offset when a
dead-letter write is not durable (``BoundaryDlqNotPersistedError``): there, the
record still exists on the source topic, so withholding preserves it for the
rewind path. That argument does not transfer. A command this agent cannot
decode can never be decoded by redelivery -- retrying is not recovery, it is
the outage -- and the cost of withholding is not one lost record but every
subsequent deploy command on the partition, which is what actually happened.
So the commit is unconditional and the durability is best-effort *and*
recorded, in that order. The gateway-delivery path made the same call for the
same reason (OMN-15748).

WHAT IS DELIBERATELY *NOT* QUARANTINED
--------------------------------------

Only the two decode-failure error classes are caught at fetch level. A broker
that is down, a SASL handshake that fails, a rebalance -- all of those are
transient and must NOT advance an offset, so they propagate. Skipping records
because the network hiccuped would be a worse failure than the one this fixes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from kafka import KafkaConsumer, TopicPartition
from kafka.errors import CorruptRecordError, UnsupportedCodecError
from kafka.structs import OffsetAndMetadata

from deploy_agent.auth import verify_command
from deploy_agent.coalesce import (
    AncestryResolver,
    ModelCoalescePlan,
    ModelQueuedCommand,
    ModelSupersession,
    plan_coalesce,
)
from deploy_agent.events import (
    TOPIC_DEPLOY_COMMAND_DLQ,
    TOPIC_REBUILD_REQUESTED,
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRejectionNotice,
    Scope,
)
from deploy_agent.job_state import JobStore
from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from deploy_agent.lane_policy import (
    LaneNotAllowedError,
    assert_lane_allowed,
)
from deploy_agent.lineage_fence import (
    EnumLineageVerdict,
    ModelLineageDecision,
    RunningBuildRefReader,
    decide_lineage,
)
from deploy_agent.queue_depth import LagSampler, ModelControlTopicLag

logger = logging.getLogger(__name__)

# OMN-16442. Invoked at the PRE_ACCEPT job boundary with a callback that rewinds
# this consumer's committed offset to the command being examined. The hook is
# expected not to return when it decides to update: it replaces the process
# image, and the rewound offset is what makes the replacement process re-read
# the command instead of skipping it.
SelfUpdateHook = Callable[[Callable[[], None]], None]

# OMN-18143. Invoked once per command the coalescing scan folds into a newer
# one, AFTER its durable ``superseded`` job record has been written and BEFORE
# the runner is accepted. The agent supplies the publisher; this consumer holds
# no producer of its own, and giving it one so it could emit its own terminal
# events would put a second bus writer in the process for no gain.
#
# A hook that raises is logged and the scan continues: the durable record is
# already written and carries ``result_publish_pending``, so the agent's own
# retry loop owes the event either way. Losing the coalescing decision because
# a broker was briefly away would be the worse trade.
SupersededHook = Callable[[ModelSupersession], None]

# OMN-17079. The consumer decides six of the eight rejection reasons and, before this
# hook, expressed each as a bare string returned to a caller that only logged it -- so
# six of eight refusals never reached the rejection topic at all. This is their route to
# the agent's single publish helper, and it deliberately mirrors ``SupersededHook``
# rather than inventing a second mechanism.
RejectedHook = Callable[[ModelRejectionNotice], None]

# Stamped on the rewound offset so `rpk group describe` shows WHY the group's
# committed offset moved BACKWARDS onto a record it had already fetched.
SELF_UPDATE_REWIND_METADATA = "onex-deploy-agent:rewound-for-self-update"

ENV_QUARANTINE_DIR = "DEPLOY_AGENT_QUARANTINE_DIR"

# Fetch-level failures that redelivery can never fix. Kept deliberately narrow:
# anything transient must propagate rather than advance an offset.
UNDECODABLE_FETCH_ERRORS: tuple[type[BaseException], ...] = (
    UnsupportedCodecError,
    CorruptRecordError,
)

# Bytes of a rejected payload kept in the quarantine record, so an operator can
# recognise the message without the record becoming a copy of an arbitrary blob.
RAW_PREVIEW_BYTES = 512

# Stamped on the committed offset so `rpk group describe` shows WHY the group
# is past a record it never processed, without anyone reading the journal.
QUARANTINE_COMMIT_METADATA = "onex-deploy-agent:quarantined-undecodable"

#: Stamped on every offset `_process_message` commits, so `rpk group
#: describe` shows that the commit was bounded to ONE record rather than to
#: the consumer's fetch position (OMN-18613).
PROCESSED_COMMIT_METADATA = "onex-deploy-agent:processed-one-record"

# kafka-python requires an explicit leader epoch; -1 is its "unknown" sentinel.
UNKNOWN_LEADER_EPOCH = -1


@dataclass(frozen=True)
class UndecodableValue:
    """Marker returned by the deserializer instead of raising inside ``poll()``.

    Raising in a ``value_deserializer`` is the record-level shape of the same
    stall: the exception escapes ``poll()``, so the record object that knows its
    own offset is never handed back and the agent cannot commit past it.
    """

    error: str
    raw_preview: str
    raw_bytes: int


def deserialize_command_value(raw: bytes) -> Any:
    """Decode one command record, returning a marker rather than raising."""
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — any decode failure is one class here
        return UndecodableValue(
            error=f"{type(exc).__name__}: {exc}",
            raw_preview=repr(raw[:RAW_PREVIEW_BYTES]),
            raw_bytes=len(raw),
        )
    if not isinstance(decoded, dict):
        return UndecodableValue(
            error=(
                "command payload decoded to "
                f"{type(decoded).__name__}, expected a JSON object"
            ),
            raw_preview=repr(raw[:RAW_PREVIEW_BYTES]),
            raw_bytes=len(raw),
        )
    return decoded


class DeployConsumer:
    #: OMN-18144. The queue observer, optional and absent by default because
    #: this consumer is also built by tests and by paths that are not serving
    #: the queue endpoint. Declared on the CLASS so a partially constructed
    #: consumer has the attribute: an observer nobody injected is None, which
    #: means nothing is sampled, which means the endpoint reports the lag
    #: unknown -- never zero.
    lag_sampler: LagSampler | None = None
    #: OMN-18143. Declared on the CLASS for the same reason ``lag_sampler`` is:
    #: this consumer is also built by tests and by paths that construct it
    #: without going through ``__init__``, and a partially constructed
    #: consumer must read as "coalesces nothing" rather than raising
    #: ``AttributeError`` from the middle of the accept protocol.
    ancestry_resolver: AncestryResolver | None = None
    on_superseded: SupersededHook | None = None
    #: OMN-17079, on the same terms as the two above: a consumer built without
    #: ``__init__`` must read as "notifies nobody" rather than raising
    #: AttributeError from the middle of a refusal, which would turn a handled
    #: rejection into an unhandled exception on the poll loop.
    on_rejected: RejectedHook | None = None
    #: OMN-19270, on the same terms again: a consumer built without a reader
    #: runs no lineage fence, which is the pre-change behaviour.
    running_build_ref: RunningBuildRefReader | None = None
    tracking_ref: str | None = None

    def __init__(
        self,
        kafka_config: ModelDeployAgentKafkaConfig,
        job_store: JobStore,
        allowed_lanes: frozenset[EnumRuntimeLane],
        self_update_hook: SelfUpdateHook,
        quarantine_dir: Path | None = None,
        lag_sampler: LagSampler | None = None,
        ancestry_resolver: AncestryResolver | None = None,
        on_superseded: SupersededHook | None = None,
        on_rejected: RejectedHook | None = None,
        running_build_ref: RunningBuildRefReader | None = None,
        tracking_ref: str | None = None,
    ) -> None:
        self.consumer = KafkaConsumer(
            TOPIC_REBUILD_REQUESTED,
            **kafka_config.consumer_kwargs(),
            group_id="onex-deploy-agent",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=deserialize_command_value,
        )
        self.kafka_config = kafka_config
        self.job_store = job_store
        self.allowed_lanes = allowed_lanes
        self.self_update_hook = self_update_hook
        self.quarantine_dir = Path(
            quarantine_dir
            or os.environ.get(ENV_QUARANTINE_DIR)
            or job_store.state_dir.parent / "quarantine"
        )
        # OMN-18144. The observer also holds the committed offsets the lag is
        # measured against: `consumer.committed()` is a coordinator round trip
        # and this is sampled once per poll, so the values this process itself
        # committed are both cheaper and exact.
        self.lag_sampler = lag_sampler
        # OMN-18143. Absent by default: a consumer built without a resolver
        # coalesces nothing and behaves exactly as it did before this change.
        # That is the same fail-closed direction every refusal inside the scan
        # takes -- run every command, in order.
        self.ancestry_resolver = ancestry_resolver
        self.on_superseded = on_superseded
        self.on_rejected = on_rejected
        # OMN-19270. The lineage fence reads the running build through this
        # reader and compares refs through ``ancestry_resolver``; it runs only
        # when both are present.
        self.running_build_ref = running_build_ref
        self.tracking_ref = tracking_ref
        logger.info(
            "Deploy agent lane fence: %s",
            ",".join(sorted(lane.value for lane in self.allowed_lanes)),
        )

    def poll_and_accept(self) -> tuple[ModelRebuildRequested | None, str | None]:
        """Poll for one command.

        Returns (command, None) on accept.
        Returns (None, reason) on reject.
        Returns (None, None) on no message.

        Protocol:
        1. Poll message — an undecodable batch is quarantined, not re-read
        2. Verify HMAC signature
        3. Validate payload (schema, scope, services legality)
        4. Check the lane fence -> reject "lane_not_allowed"
        5. Check busy (has_active_job) -> reject "busy"
        6. Check dedup (is_duplicate) -> reject "duplicate"
        6a. Lineage fence (OMN-19270) -> reject "superseded_by_running_build"
            or "divergent_ref"
        7. Self-update boundary (OMN-16442) -- may not return
        8. Persist job state (accepted)
        9. Commit Kafka offset
        10. Return (command, None)
        """
        try:
            records = self.consumer.poll(timeout_ms=1000)
        except UNDECODABLE_FETCH_ERRORS as exc:
            return None, self._quarantine_undecodable_fetch(exc)

        # OMN-18144. Sampled after the poll, on EVERY path including the empty
        # one, because "nothing arrived" is exactly when a reader most needs to
        # know whether the queue behind it is empty or three deep. Sampling is
        # read-only and never fails the poll: a lag that cannot be read is
        # recorded as unknown.
        self._sample_lag()

        if not records:
            return None, None

        # OMN-18143. The batch this poll ALREADY fetched is the look-ahead, and
        # nothing beyond it is polled for.
        #
        # That is a deliberate bound, not a shortcut. The fetch position is
        # already past every record in this batch -- ``_commit_through``'s own
        # docstring records what that costs -- so scanning records the client
        # has in hand adds no exposure at all, while polling AGAIN to see
        # further would advance the position over records this scan then
        # declines to fold, widening exactly the window OMN-18613 measured a
        # lost command in. A queue deeper than one fetch simply coalesces
        # across successive polls instead of in one, which is slower and
        # equally correct.
        ordered, lookahead_usable = self._order_batch(records)
        if not ordered:
            return None, None
        return self._process_message(
            ordered[0], lookahead=ordered[1:] if lookahead_usable else []
        )

    def _order_batch(self, records: dict[Any, list[Any]]) -> tuple[list[Any], bool]:
        """The fetched batch in control-topic order, and whether it may be scanned.

        Coalescing reorders nothing, so it may only reason about records whose
        relative order Kafka actually guarantees -- which is per partition and
        nowhere else. A poll that returned records on more than one partition
        therefore yields ONE message and no look-ahead: the head is processed
        exactly as it was before this change, and the rest are read on later
        polls. The control topic has a single partition today; this refuses to
        depend on that staying true.
        """
        populated = [(tp, msgs) for tp, msgs in records.items() if msgs]
        if not populated:
            return [], False
        topic_partition, messages = populated[0]
        ordered = sorted(messages, key=lambda m: m.offset)
        if len(populated) > 1:
            logger.info(
                "coalesce: this poll returned records on %d partitions, so no "
                "look-ahead is used and %s is processed alone",
                len(populated),
                topic_partition,
            )
            return ordered[:1], False
        return ordered, True

    def _reject(
        self,
        reason: EnumRejectionReason,
        *,
        raw: object = None,
        cmd: ModelRebuildRequested | None = None,
    ) -> str:
        """Fire the rejection hook and return the reason token the caller returns.

        OMN-17079. Returning the token keeps every call site a one-liner
        (``return None, self._reject(...)``), so a future refusal cannot be added as a
        bare ``return None, "reason"`` without visibly departing from the shape its six
        neighbours use.

        THE IDENTIFIERS ARE RESOLVED, NEVER INVENTED. With a validated ``cmd`` both come
        straight off it. Without one -- an undecodable record, a bad signature, a payload
        the contract refuses -- this parses what the raw record still offers and records
        ``None`` for whatever will not parse. A notice carrying ``None`` cannot become an
        event, which is the intended outcome: a rejection published under a fabricated
        correlation id is a durable record pointing at a command that never existed.

        Never raises. A refusal whose notification fails is still a refusal, and the
        offset is already committed by the caller; losing the hook must not convert that
        into an unhandled exception on the poll loop.
        """
        correlation_id = cmd.correlation_id if cmd is not None else None
        scope: Scope | None = cmd.scope if cmd is not None else None

        if cmd is None and isinstance(raw, dict):
            try:
                correlation_id = UUID(str(raw["correlation_id"]))
            except (KeyError, ValueError, TypeError):
                correlation_id = None
            try:
                scope = Scope(str(raw["scope"]))
            except (KeyError, ValueError, TypeError):
                scope = None

        if self.on_rejected is not None:
            try:
                self.on_rejected(
                    ModelRejectionNotice(
                        reason=reason,
                        correlation_id=correlation_id,
                        scope=scope,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — boundary: refusal still stands
                logger.warning(
                    "Publishing the %s rejection failed (%s: %s); the refusal itself "
                    "stands and its offset is committed",
                    reason.value,
                    type(exc).__name__,
                    exc,
                )
        return reason.value

    def _process_message(
        self, msg: Any, lookahead: list[Any] | None = None
    ) -> tuple[ModelRebuildRequested | None, str | None]:
        payload = msg.value

        # Step 1b: the deserializer hands back a marker rather than raising, so
        # this record's own offset is available to commit past.
        if isinstance(payload, UndecodableValue):
            self._quarantine_record(
                msg,
                reason="undecodable_payload",
                detail=payload.error,
                raw_preview=payload.raw_preview,
            )
            self._commit_through(msg)
            return None, self._reject(
                EnumRejectionReason.UNDECODABLE_PAYLOAD, raw=payload.raw_preview
            )

        correlation_id_str = payload.get("correlation_id", "unknown")

        # Step 2: Verify HMAC signature
        if not verify_command(payload):
            logger.warning(
                "Rejecting command (correlation_id=%s): invalid_signature",
                correlation_id_str,
            )
            self._commit_through(msg)
            return None, self._reject(
                EnumRejectionReason.INVALID_SIGNATURE, raw=payload
            )

        # Step 3: Validate payload. The signature is transport metadata, not
        # part of the command contract itself.
        try:
            command_payload = {k: v for k, v in payload.items() if k != "_signature"}
            cmd = ModelRebuildRequested.model_validate(command_payload)
        except Exception as e:  # noqa: BLE001
            # A signed command the contract refuses is as permanent as an
            # undecodable one -- the bytes will not change on redelivery -- so
            # it is quarantined with the reason rather than merely logged. This
            # is the class that took the dev lane's operator entry point out:
            # the trigger script published `reason` and omitted `runtime_lane`
            # for weeks and the only trace was a warning line.
            self._quarantine_record(
                msg,
                reason="invalid_payload",
                detail=str(e),
                raw_preview=None,
            )
            self._commit_through(msg)
            return None, self._reject(EnumRejectionReason.INVALID_PAYLOAD, raw=payload)

        # Step 4: Lane fence (OMN-16939). The dev control bus carries both dev
        # and stability-test rebuild commands, so "which bus am I on" does not
        # bound which lane this process may mutate. The offset is committed:
        # the command is not for this agent and re-reading it forever would
        # stall every command behind it.
        try:
            assert_lane_allowed(cmd.runtime_lane, self.allowed_lanes)
        except LaneNotAllowedError as e:
            logger.warning(
                "Rejecting command %s: lane_not_allowed (%s)", cmd.correlation_id, e
            )
            self._commit_through(msg)
            return None, self._reject(EnumRejectionReason.LANE_NOT_ALLOWED, cmd=cmd)

        # Step 5: Check busy
        if self.job_store.has_active_job():
            logger.info("Rejecting command %s: agent busy", cmd.correlation_id)
            self._commit_through(msg)
            return None, self._reject(EnumRejectionReason.BUSY, cmd=cmd)

        # Step 6: Check dedup
        if self.job_store.is_duplicate(cmd.correlation_id):
            logger.info("Rejecting command %s: duplicate", cmd.correlation_id)
            self._commit_through(msg)
            return None, self._reject(EnumRejectionReason.DUPLICATE, cmd=cmd)

        # Step 6a: Lineage fence (OMN-19270). A command whose ref is a strict
        # ancestor of the build the lane already runs would roll the lane
        # back. That is what a stale, replayed redeploy did on 2026-09-23, when
        # a command stranded upstream since 12:09Z arrived at 15:50Z. Refused
        # with a typed reason and committed past, so it is acknowledged rather
        # than left to block the commands behind it. A signed rollback
        # declaration is exempt. Every comparison the host cannot make lets
        # the command run as before.
        lineage = self._lineage_decision(cmd)
        if lineage is not None and lineage.verdict.refuses:
            logger.warning(
                "Rejecting command %s: %s", cmd.correlation_id, lineage.journal_line()
            )
            self._commit_through(msg)
            if lineage.verdict is EnumLineageVerdict.STALE_ANCESTOR:
                return None, self._reject(
                    EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD, cmd=cmd
                )
            return None, self._reject(EnumRejectionReason.DIVERGENT_REF, cmd=cmd)

        # Step 6b: Coalesce (OMN-18143). The newest foldable command in the
        # batch runs; every one it replaces gets a durable terminal record and
        # a terminal event naming it. Everything below this point -- the
        # self-update boundary, the accept, the commit and the returned
        # command -- is about the RUNNER, which is this message only when
        # nothing was folded.
        runner_cmd, runner_msg, superseded_ids = self._coalesce(cmd, msg, lookahead)

        # Step 7: Self-update boundary (OMN-16442). This is the last point at
        # which nothing is in flight: the command has passed every acceptance
        # check but is not yet marked started and its offset is not yet
        # committed. If the agent is behind its tracking ref, the hook replaces
        # the process image here and does not return -- having first rewound
        # the committed offset to this message, so the replacement process
        # re-reads it. Update-then-process, rather than the process-then-die
        # that killed command 8d0c861a-f91e-4ca2-954e-a073759dd39d when the
        # same check ran mid-deploy.
        #
        # A self-update failure must not cost a valid command: it is logged and
        # the command is processed on the current image, exactly as the
        # method's own dirty-tree and fetch-failure rails already do.
        try:
            self.self_update_hook(lambda: self._rewind_committed_offset_to(runner_msg))
        except Exception as e:  # noqa: BLE001
            logger.error(  # noqa: TRY400
                "Self-update at the pre-accept boundary failed for %s, "
                "proceeding on the current image: %s "
                "friction_type=self_update_boundary_failed",
                runner_cmd.correlation_id,
                e,
            )

        # Step 8: Persist job state
        self.job_store.accept(
            correlation_id=runner_cmd.correlation_id,
            command=self._command_payload(runner_msg),
            superseded_correlation_ids=superseded_ids,
        )

        # Step 9: Commit offset. The runner's offset is at or past every
        # superseded record's, so one commit covers the whole group.
        self._commit_through(runner_msg)

        # Step 10: Return accepted command
        logger.info(
            "Accepted command %s (scope=%s, superseded=%d)",
            runner_cmd.correlation_id,
            runner_cmd.scope,
            len(superseded_ids),
        )
        return runner_cmd, None

    def _lineage_decision(
        self, cmd: ModelRebuildRequested
    ) -> ModelLineageDecision | None:
        """The lineage fence's verdict for ``cmd``, or ``None`` when it is off.

        Off means this consumer was built without a running-build reader or
        without an ancestry resolver, which is every test that does not ask
        for the fence. The verdict is journalled whether it refuses or not, so
        an accepted command still records which comparison let it through.
        """
        if self.running_build_ref is None or self.ancestry_resolver is None:
            return None
        reader = self.running_build_ref
        decision = decide_lineage(
            cmd,
            read_running_ref=lambda: reader(cmd.runtime_lane),
            contains=self.ancestry_resolver,
            tracking_ref=self.tracking_ref,
        )
        if not decision.verdict.refuses:
            logger.info("%s %s", cmd.correlation_id, decision.journal_line())
        return decision

    @staticmethod
    def _command_payload(msg: Any) -> dict[str, Any]:
        """The command as the contract sees it: the record minus its signature.

        The signature is transport metadata, not part of the command, and the
        job record has never carried it. Factored out because the coalescing
        scan needs the same projection for a message the head path never
        decoded.
        """
        return {k: v for k, v in msg.value.items() if k != "_signature"}

    # ── coalescing (OMN-18143) ───────────────────────────────────────────────
    def _coalesce(
        self,
        cmd: ModelRebuildRequested,
        msg: Any,
        lookahead: list[Any] | None,
    ) -> tuple[ModelRebuildRequested, Any, list[UUID]]:
        """Decide which of the fetched batch runs; record the ones it replaces.

        Returns the command to run, its Kafka record, and the correlation ids
        it superseded. With no resolver, no look-ahead, or nothing foldable,
        it returns the head unchanged and an empty list -- the pre-change
        behaviour, reached by the same code path rather than by a branch
        around it.
        """
        if self.ancestry_resolver is None or not lookahead:
            return cmd, msg, []

        queued = [
            ModelQueuedCommand(command=cmd, partition=msg.partition, offset=msg.offset)
        ]
        messages_by_offset: dict[int, Any] = {msg.offset: msg}
        for candidate in lookahead:
            decoded = self._decode_for_lookahead(candidate)
            if decoded is None:
                # A record the scan cannot cleanly read is left completely
                # alone -- not quarantined, not committed past, not counted.
                # It will be polled again and handled by the head path, which
                # is the one place that owns refusing a command. The group
                # ends here because the scan may not reorder around it.
                break
            queued.append(
                ModelQueuedCommand(
                    command=decoded,
                    partition=candidate.partition,
                    offset=candidate.offset,
                )
            )
            messages_by_offset[candidate.offset] = candidate

        plan = plan_coalesce(queued, contains=self.ancestry_resolver)
        logger.info("%s", plan.journal_line())
        if not plan.superseded:
            return cmd, msg, []

        return (
            plan.runner.command,
            messages_by_offset[plan.runner.offset],
            self._record_supersessions(plan, messages_by_offset),
        )

    def _record_supersessions(
        self, plan: ModelCoalescePlan, messages_by_offset: dict[int, Any]
    ) -> list[UUID]:
        """Write each superseded command's terminal record, then announce it.

        Record first, announce second, and never the reverse: the durable
        record is what makes the supersession survive a broker outage or a
        crash, and it carries ``result_publish_pending`` so the agent's own
        retry loop owes the event even if the hook below never fires.
        """
        ids: list[UUID] = []
        for supersession in plan.superseded:
            superseded_cmd = supersession.superseded.command
            record_msg = messages_by_offset[supersession.superseded.offset]
            self.job_store.record_superseded(
                correlation_id=superseded_cmd.correlation_id,
                command=self._command_payload(record_msg),
                superseded_by_sha=supersession.superseded_by_sha,
                superseded_by_correlation_id=supersession.superseded_by_correlation_id,
            )
            logger.info(
                "coalesce: %s (ref=%s) superseded by %s (ref=%s)",
                superseded_cmd.correlation_id,
                superseded_cmd.git_ref,
                supersession.superseded_by_correlation_id,
                supersession.superseded_by_sha,
            )
            if self.on_superseded is not None:
                try:
                    self.on_superseded(supersession)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "coalesce: publishing the superseded event for %s "
                        "failed (%s: %s); the durable record is written and "
                        "the publish is still owed",
                        superseded_cmd.correlation_id,
                        type(exc).__name__,
                        exc,
                    )
            ids.append(superseded_cmd.correlation_id)
        return ids

    def _decode_for_lookahead(self, msg: Any) -> ModelRebuildRequested | None:
        """Read a queued record without consuming it, or return ``None``.

        Deliberately SILENT about every refusal and deliberately without side
        effects. The head path owns quarantining, dead-lettering and
        committing past a bad record, and doing any of that from here would
        mean a record is refused by a scan that was only supposed to look at
        it -- with its rejection event published before the agent had even
        reached it in the queue.
        """
        payload = msg.value
        if isinstance(payload, UndecodableValue) or not isinstance(payload, dict):
            return None
        if not verify_command(payload):
            return None
        try:
            cmd = ModelRebuildRequested.model_validate(
                {k: v for k, v in payload.items() if k != "_signature"}
            )
        except Exception:  # noqa: BLE001 - an unreadable candidate is not foldable
            return None
        try:
            assert_lane_allowed(cmd.runtime_lane, self.allowed_lanes)
        except LaneNotAllowedError:
            return None
        return cmd

    def _sample_lag(self) -> None:
        """Record how many control-topic records this agent has not dealt with.

        Measured against the COMMITTED offset, not the fetch position. After a
        poll that returned a batch the position is already past records
        ``poll_and_accept`` left buffered and has never looked at, so a
        position-based lag reports zero while commands wait -- the same
        off-by-a-batch OMN-18613 found in ``_commit_through``, from the other
        side. Where this process has not committed anything yet the position is
        used and the basis says so, because a first-poll under-report by one
        batch is better than no number at all and the reader can see which it got.

        Never raises. A sampler that could fail a poll would trade the agent's
        actual job for an observation of it.
        """
        if self.lag_sampler is None:
            return
        try:
            assignment = self.consumer.assignment()
            if not assignment:
                self.lag_sampler.record(
                    ModelControlTopicLag.unknown(
                        "this consumer holds no partition assignment yet, so "
                        "there is no offset to measure a lag against"
                    )
                )
                return
            total = 0
            bases: set[str] = set()
            for topic_partition in assignment:
                highwater = self.consumer.highwater(topic_partition)
                if highwater is None:
                    self.lag_sampler.record(
                        ModelControlTopicLag.unknown(
                            f"no highwater is known for {topic_partition} yet, "
                            "so the records beyond this agent cannot be counted"
                        )
                    )
                    return
                committed = self.lag_sampler.committed(topic_partition)
                if committed is None:
                    committed = self.consumer.position(topic_partition)
                    basis = "position"
                else:
                    basis = "committed"
                if committed is None:
                    self.lag_sampler.record(
                        ModelControlTopicLag.unknown(
                            f"no offset is known for {topic_partition} yet, so "
                            "this agent's place in the topic cannot be read"
                        )
                    )
                    return
                bases.add(basis)
                total += max(0, int(highwater) - int(committed))
            self.lag_sampler.record(
                ModelControlTopicLag(
                    value=total,
                    basis="+".join(sorted(bases)),
                    observed_at=datetime.now(UTC),
                )
            )
        except Exception as exc:  # noqa: BLE001 - an unreadable lag is unknown, never fatal
            logger.debug("control-topic lag unreadable: %s", exc)
            self.lag_sampler.record(
                ModelControlTopicLag.unknown(
                    f"the control-topic lag could not be read: "
                    f"{type(exc).__name__}: {exc}"
                )
            )

    def _commit_through(self, msg: Any) -> None:
        """Commit past THIS record and no further.

        A bare ``self.consumer.commit()`` commits the consumer's POSITION for
        every assigned partition. After a ``poll()`` that returned a batch the
        position is past every record FETCHED, not past the one record
        ``_process_message`` was handed -- ``poll_and_accept`` deliberately
        processes the first record and returns, leaving the rest buffered. So a
        bare commit silently marks records the agent has never looked at as
        done, and they are gone the moment the client buffer is discarded, which
        the ``post_terminal`` self-update re-exec does routinely.

        Measured 2026-09-17 (OMN-18613): accepting offset 262 committed 264,
        past a buffered 263 carrying the rebuild command for omnimarket#2622.
        That command was never delivered again -- no job record, no acceptance
        line, no rejection line and no quarantine record.

        Advancing past a record the agent REFUSES stays deliberate: step 4's own
        comment notes that re-reading a refused command forever "would stall
        every command behind it". What is bounded here is the reach of that
        advance, not its direction.
        """
        topic_partition = TopicPartition(msg.topic, msg.partition)
        self.consumer.commit(
            {
                topic_partition: OffsetAndMetadata(
                    msg.offset + 1, PROCESSED_COMMIT_METADATA, UNKNOWN_LEADER_EPOCH
                )
            }
        )
        # OMN-18144: what the next lag sample measures against.
        if self.lag_sampler is not None:
            self.lag_sampler.note_commit(topic_partition, msg.offset + 1)

    def _rewind_committed_offset_to(self, msg: Any) -> None:
        """Commit this message's own offset so it is re-read, not skipped.

        Called immediately before the process image is replaced. Seeking to
        ``msg.offset`` and committing the resulting position makes the
        committed offset point AT this command rather than past it, so the
        replacement process fetches the same command again.

        Relying on the message simply being uncommitted is not enough: this
        consumer is configured ``auto_offset_reset="latest"``, so a group with
        no committed offset yet -- the first command a freshly created group
        ever sees -- would resume past the message and lose it.
        """
        topic_partition = TopicPartition(msg.topic, msg.partition)
        self.consumer.seek(topic_partition, msg.offset)
        self.consumer.commit(
            {
                topic_partition: OffsetAndMetadata(
                    msg.offset, SELF_UPDATE_REWIND_METADATA, UNKNOWN_LEADER_EPOCH
                )
            }
        )
        # OMN-18144: a rewind moves the committed offset BACK, and a lag
        # measured against the old value would under-report the record this
        # rewind exists to have re-read.
        if self.lag_sampler is not None:
            self.lag_sampler.note_commit(topic_partition, msg.offset)
        logger.info(
            "Rewound committed offset to %s@%d:%d before self-update re-exec",
            msg.topic,
            msg.partition,
            msg.offset,
        )

    # ── quarantine ───────────────────────────────────────────────────────────

    def _quarantine_undecodable_fetch(self, exc: BaseException) -> str:
        """Advance past a batch the client cannot decode, one offset at a time.

        No record object exists, so the poison offset is read from the
        consumer's own position. Only partitions with an unconsumed record
        (``position < highwater``) are candidates: a caught-up partition has
        nothing pending and advancing it would skip a future command.

        Exactly ONE offset is skipped per candidate partition per event, never
        a bulk seek to the end. On the single-partition control topic that is
        precisely the poison record; were the topic ever repartitioned, the
        worst case is one good command per extra candidate partition, and every
        skipped coordinate is written to the quarantine record rather than
        being lost silently.
        """
        advanced: list[dict[str, Any]] = []
        for tp in sorted(
            self.consumer.assignment(), key=lambda p: (p.topic, p.partition)
        ):
            position = self.consumer.position(tp)
            highwater = self.consumer.highwater(tp)
            if position is None or highwater is None or position >= highwater:
                continue
            self.consumer.seek(tp, position + 1)
            self.consumer.commit(
                {
                    tp: OffsetAndMetadata(
                        position + 1, QUARANTINE_COMMIT_METADATA, UNKNOWN_LEADER_EPOCH
                    )
                }
            )
            # OMN-18144: a quarantined record has been dealt with, so the lag
            # must not keep counting it.
            if self.lag_sampler is not None:
                self.lag_sampler.note_commit(tp, position + 1)
            advanced.append(
                {
                    "topic": tp.topic,
                    "partition": tp.partition,
                    "skipped_offset": position,
                    "committed_offset": position + 1,
                }
            )

        record = {
            "quarantined_at": datetime.now(UTC).isoformat(),
            "class": "undecodable_fetch",
            "reason": f"{type(exc).__name__}: {exc}",
            "advanced": advanced,
        }

        if not advanced:
            # Honest rather than silent: the stall was NOT cleared, and saying
            # so is the only way the next restart is diagnosable.
            logger.critical(
                "metric_name=deploy_command_quarantine_failed class=undecodable_fetch "
                "error=%s: no assigned partition had a pending record to skip, so "
                "the offset could not be advanced and this fetch will fail again. "
                "friction_type=deploy_command_poison_pill_unrecoverable",
                record["reason"],
            )
            self._write_quarantine_file(record)
            return "undecodable_fetch_unrecovered"

        logger.critical(
            "metric_name=deploy_command_quarantined class=undecodable_fetch "
            "error=%s advanced=%s — offset committed past the undecodable record. "
            "friction_type=deploy_command_poison_pill",
            record["reason"],
            json.dumps(advanced),
        )
        self._write_quarantine_file(record)
        self._publish_dlq(record)
        return "undecodable_fetch"

    def _quarantine_record(
        self,
        msg: Any,
        *,
        reason: str,
        detail: str,
        raw_preview: str | None,
    ) -> None:
        """Record one permanently-unprocessable message before committing past it."""
        record = {
            "quarantined_at": datetime.now(UTC).isoformat(),
            "class": reason,
            "reason": detail,
            "topic": msg.topic,
            "partition": msg.partition,
            "skipped_offset": msg.offset,
            "committed_offset": msg.offset + 1,
            "key": None if msg.key is None else str(msg.key),
            "raw_preview": raw_preview,
        }
        logger.critical(
            "metric_name=deploy_command_quarantined class=%s topic=%s partition=%s "
            "offset=%s error=%s — offset committed past it. "
            "friction_type=deploy_command_poison_pill",
            reason,
            msg.topic,
            msg.partition,
            msg.offset,
            detail,
        )
        self._write_quarantine_file(record)
        self._publish_dlq(record)

    def _write_quarantine_file(self, record: dict[str, Any]) -> None:
        """Write the durable local copy. Best-effort, and loud when it fails."""
        try:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            stamp = record["quarantined_at"].replace(":", "").replace("-", "")
            path = self.quarantine_dir / f"{stamp}-{record['class']}.json"
            fd, tmp = tempfile.mkstemp(dir=self.quarantine_dir, suffix=".tmp")
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
            Path(tmp).replace(path)
            logger.info("Quarantine record written: %s", path)
        except Exception as exc:
            logger.exception(
                "metric_name=deploy_command_quarantine_write_failed error=%s record=%s",
                exc,
                json.dumps(record, default=str),
            )

    def _publish_dlq(self, record: dict[str, Any]) -> None:
        """Dead-letter the quarantine record. Best-effort: the commit already stands."""
        from kafka import KafkaProducer

        try:
            producer = KafkaProducer(
                **self.kafka_config.producer_kwargs(),
                compression_type=None,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            )
            try:
                producer.send(TOPIC_DEPLOY_COMMAND_DLQ, value=record)
                producer.flush(timeout=10)
            finally:
                producer.close()
            logger.info(
                "Dead-lettered quarantine record to %s", TOPIC_DEPLOY_COMMAND_DLQ
            )
        except Exception as exc:
            logger.exception(
                "metric_name=deploy_command_dlq_publish_failed topic=%s error=%s — "
                "the durable local quarantine record is the surviving evidence",
                TOPIC_DEPLOY_COMMAND_DLQ,
                exc,
            )

    def close(self) -> None:
        self.consumer.close()
