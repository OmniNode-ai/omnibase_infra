# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler for the live event-chain canary (OMN-16773).

What this proves that nothing else did
--------------------------------------
``delegation-seam-gate.yml`` already drives the delegation seam at PR time
over ``InMemoryTransport``, and it was green the entire time the live chain
was dead. It has to be: an in-memory seam test constructs its own wiring, so
it cannot see a *deployed* wiring arm choosing the wrong dispatch path.

OMN-16767 is exactly that failure. An omnimarket contract change on
2026-08-23 gave ``node_delegation_routing_reducer`` a ``db_io`` block; the
runtime's ``_prepare_handler_wiring`` selects the projection dispatch arm on
``db_io.db_tables`` alone, so a typed def-B handler started receiving a raw
projection dict, raised ``AttributeError``, and every delegation fell into
the platform quarantine sink. Zero terminals for four days. Nobody noticed,
because the only thing that exercised the live chain was a human running a
recorded recipe by hand.

So this node runs that recipe on a schedule. It fires ONE delegation class
through the real deployed ingress and asserts two things a green unit suite
cannot: a terminal came back inside a budget, and the run's own correlation
id did not land in quarantine.

Design constraints worth stating
--------------------------------
* **Fresh correlation id, minted here, not settable by the caller.** A
  canary that can be handed a fixed id is a canary whose results can be
  confused across runs. ``uuid4()`` per ``handle()`` call, and that same id
  is what goes on the wire and what the quarantine scan asks about.
* **The quarantine leg is a tail scan.** The sink held ~8,878,924 records
  when OMN-16767 was diagnosed. Reading it whole is not an option, and is
  not the question anyway — the canary only needs the window its own
  request just landed in. Aggregate sink depth is OMN-16769's job.
* **An unconfigured check reports itself.** ``SKIPPED_NOT_CONFIGURED`` is
  not ``CLEAN``. The whole reason this ticket exists is that a check nobody
  ran looked exactly like a check that passed.
* **No node ever asserts a lane it did not probe.** ``probe_url`` is
  required with no default (Rule 8). The dev lane is the pre-authorized
  mutable lane; stability-test, judge, and prod are out of scope here and
  this node makes no claim about them.

What OMN-16931 changed, and why
-------------------------------
The first cut of this handler derived ``terminal_landed`` from the
synchronous ``/skill`` HTTP response. That is a CLAIM made by the request
path about the chain, not evidence from the chain, and it failed in both
directions on the live lane:

* **False RED.** Run 33251822642 (2026-08-29T12:10:26Z) reported
  ``terminal_missing`` at 4,369 ms of a 120,000 ms budget because the
  ingress carried ``ok=false`` — a provider 429 on an escalation rung the
  local model had already answered three times (OMN-16932). The runtime log
  for that run's own correlation id shows the terminal published to
  ``delegate-skill-completed.v1`` at 12:10:23. The chain was alive. The
  canary sent whoever read it hunting a dead chain.
* **False GREEN.** OMN-15468 is the standing live proof that this lane can
  answer ``ok=true`` with a terminal name while the FAILED run's SUCCESS
  terminal is what actually got republished. An ingress-derived verdict
  cannot see that at all.

So the terminal now comes off the broker: a correlation-scoped readback of
the contract-declared terminal topics, running for the remainder of the
budget. ``TERMINAL_MISSING`` is reported only when that readback finds
nothing. An unconfigured readback is RED (``TERMINAL_READBACK_NOT_CONFIGURED``)
rather than a fallback to the ingress — the fallback is the defect.

The second half of OMN-16931 is honesty about scope. OMN-16025 is a
FIVE-link gate; this probe has legs for three of those links and none for
link 2 (projection readback, owed by OMN-16963) or link 5 (ledger chain +
replay, owed by OMN-16964). A single scalar verdict let run 33215999994's
GREEN read as a five-link proof. The receipt now carries a status per link,
``links_proven``/``links_total``, and ``chain_proof_complete`` — which is
False on the best run this probe can currently produce.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_projection_readback_status import (
    EnumProjectionReadbackStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_quarantine_check_status import (
    EnumQuarantineCheckStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_terminal_readback_status import (
    EnumTerminalReadbackStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_link_verdict import (
    ModelChainLinkVerdict,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

__all__ = ["HandlerChainCanary"]

# Kill switch, checked first and unconditionally. A SEPARATE variable from
# every other sweep's switch on purpose: silencing one canary must never
# silently silence another.
_KILL_SWITCH_ENV_VAR = "ONEX_CHAIN_CANARY_DISABLED"
_KILL_SWITCH_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# Extra client-side slack over the runtime's own budget. Without it the HTTP
# client can abort first and every slow-but-working chain reads as
# INGRESS_UNREACHABLE — a false RED that would teach people to ignore this
# canary, which is the one outcome worse than not having it.
_CLIENT_SLACK_SECONDS = 15.0

# Links with no leg in this probe carry the ticket that owes them, so the
# receipt routes a reader to the work instead of leaving a silent gap.
_LINK5_OWNING_TICKET = "OMN-16964"

# (response_json, transport_error, elapsed_ms)
TypeIngressPost = Callable[
    [str, dict[str, object], float],
    Awaitable[tuple[dict[str, object] | None, str, int]],
]
# (found, records_scanned, error) — found is None when undeterminable.
TypeQuarantineScan = Callable[
    [str, str, str, int, float],
    Awaitable[tuple[bool | None, int, str]],
]
# (bootstrap, topics, correlation_id, max_records, window_s)
#   -> (topic_found, records_scanned, error)
# topic_found is "" for "read and not there" and None for "could not read" —
# the same three-state convention as the quarantine scan, for the same
# reason: a check that could not run is not a check that passed.
TypeTerminalReadback = Callable[
    [str, tuple[str, ...], str, int, float],
    Awaitable[tuple[str | None, int, str]],
]
# (dsn, correlation_id, timeout_s) -> (fsm_state, error)
# fsm_state is "" for "read and no row" and None for "could not read" — the
# same three-state convention as the two broker legs, for the same reason.
TypeProjectionReadback = Callable[
    [str, str, float],
    Awaitable[tuple[str | None, str]],
]

# The FSM states that count as terminal in delegation_workflow_state. Anything
# else that exists as a row is stranded mid-flight — OMN-14843 measured
# INFERENCE_COMPLETED, RECEIVED and ROUTED, but the set is defined by what IS
# terminal rather than by enumerating what is not, so a new intermediate state
# is stranded by default instead of silently passing.
_TERMINAL_FSM_STATES: frozenset[str] = frozenset({"COMPLETED", "FAILED"})


def _terminal_topics(request: ModelChainCanaryRequest) -> tuple[str, ...]:
    """Success and failure terminals, read on one pass.

    Both are needed: a failure terminal still discharges link 4 (the
    emission was outbox-confirmed on the bus) while failing link 3
    (execution did not complete). Scanning only the success topic would
    report a cleanly-failed delegation as a missing terminal — the same
    misdiagnosis class OMN-16931 exists to remove.
    """
    return tuple(request.terminal_success_topics) + tuple(
        request.terminal_failure_topics
    )


def _readback_window_seconds(
    request: ModelChainCanaryRequest, elapsed_ms: int
) -> float:
    """How long link 4 is entitled to wait for the terminal.

    OMN-16025 link 4 is scoped "inside the budget", so the window is
    whatever is LEFT of the budget once the ingress has answered — not
    the ingress's own elapsed time. Run 33251822642 gave up at 4,369 ms
    of 120,000 ms; that is the whole bug in one number. The configured
    timeout is a floor, for the case where the ingress itself consumed
    the entire budget and there would otherwise be no window at all.
    """
    remaining_s = max(0.0, (request.budget_ms - elapsed_ms) / 1000.0)
    return max(float(request.terminal_readback_timeout_seconds), remaining_s)


async def _post_skill_via_httpx(
    url: str, body: dict[str, object], timeout_s: float
) -> tuple[dict[str, object] | None, str, int]:
    """POST the delegation command to the runtime's generic /skill edge."""
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(url, json=body)
    except Exception as exc:  # noqa: BLE001 - transport failures are a verdict
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return None, sanitize_error_message(exc), elapsed_ms
    elapsed_ms = int((time.monotonic() - started) * 1000)

    # A non-2xx is still an answer from the ingress, and its body usually
    # carries the typed error. Decode first, classify on content.
    try:
        decoded = response.json()
    except ValueError as exc:
        return (
            None,
            f"ingress returned HTTP {response.status_code} with a non-JSON "
            f"body: {sanitize_error_message(exc)}",
            elapsed_ms,
        )
    if not isinstance(decoded, dict):
        return (
            None,
            f"ingress returned HTTP {response.status_code} with a non-object body",
            elapsed_ms,
        )
    return decoded, "", elapsed_ms


async def _scan_topics_for_correlation(
    bootstrap: str,
    topics: tuple[str, ...],
    correlation_id: str,
    max_records: int,
    timeout_s: float,
    *,
    wait_for_arrival: bool,
) -> tuple[str | None, int, str]:
    """Read ``topics`` for ``correlation_id`` and say which topic carried it.

    Returns ``(topic, scanned, "")`` on a hit, ``("", scanned, "")`` when the
    topics were read and the id was not there, and ``(None, scanned, error)``
    when the scan could not be completed — the caller turns that last one
    into a failing verdict rather than a pass. A check that could not run is
    not a check that passed.

    ``wait_for_arrival`` is the difference between the two legs that use
    this. The quarantine leg asks a question about the PAST ("did my request
    already land in the sink"), so it stops as soon as every partition is
    read up to its high-water mark. The terminal readback (OMN-16931) asks a
    question about a window that is still open ("does the terminal arrive
    inside the budget"), so it keeps polling until the deadline. Both seek
    backwards first: run 33251822642's terminal was published THREE SECONDS
    BEFORE the ingress answered, so a forward-only consumer would have
    missed the very record it exists to find.
    """
    from aiokafka import AIOKafkaConsumer

    from omnibase_infra.event_bus.kafka_auth import (
        build_aiokafka_auth_kwargs_from_env,
    )

    if not topics:
        return None, 0, "no topics were declared for this scan"

    needle = correlation_id.encode()
    # The topic is passed to the CONSTRUCTOR, not assigned afterwards. A
    # bare consumer + assign() looked like the tidier shape and does not
    # work: partitions_for_topic() reads CACHED cluster metadata, and a
    # topic this consumer never subscribed to is absent from that cache. On
    # the first live run (2026-08-27, .201 dev lane) topics() listed all
    # 1626 topics INCLUDING this one, while partitions_for_topic() returned
    # None for it — so the scan fail-closed on a topic that plainly existed
    # and was sitting at 8,878,933 records. Subscribing at construction
    # makes aiokafka fetch that topic's metadata and auto-assign its
    # partitions (no group_id, so no coordinator and no group churn).
    consumer = AIOKafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        **build_aiokafka_auth_kwargs_from_env(),
    )
    scanned = 0
    try:
        await asyncio.wait_for(consumer.start(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        return None, 0, f"consumer start failed: {sanitize_error_message(exc)}"

    try:
        partitions = sorted(
            consumer.assignment(), key=lambda tp: (tp.topic, tp.partition)
        )
        if not partitions:
            return (
                None,
                0,
                f"topics {list(topics)!r} resolved no partitions "
                "(absent from the broker, or not readable by this client)",
            )
        end_offsets = await consumer.end_offsets(partitions)

        # Split the record budget across partitions so one hot partition
        # cannot consume the whole window and hide the record we want.
        per_partition = max(1, max_records // len(partitions))
        empty = True
        for partition in partitions:
            end = end_offsets[partition]
            start = max(0, end - per_partition)
            if start < end:
                empty = False
            consumer.seek(partition, start)
        if empty and not wait_for_arrival:
            # Every partition is empty: nothing there, definitively.
            return "", 0, ""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not wait_for_arrival and scanned >= max_records:
                break
            batches = await consumer.getmany(timeout_ms=1_000, max_records=500)
            if not batches:
                # In arrival-waiting mode an empty poll means "not yet", not
                # "not coming" — the whole point of link 4 is that the
                # terminal may still be in flight when the ingress answers.
                if wait_for_arrival:
                    continue
                break
            for topic_partition, records in batches.items():
                for record in records:
                    scanned += 1
                    if record.value and needle in record.value:
                        return topic_partition.topic, scanned, ""
            # Stop once every partition has been read up to its high-water
            # mark; otherwise a quiet topic would burn the whole timeout.
            # Written as an explicit loop on purpose: `position()` is a
            # coroutine, and folding it into an all(...) generator builds an
            # async generator that all() cannot iterate ("TypeError:
            # 'async_generator' object is not iterable" — hit live on
            # 2026-08-27, after the scan had already read its 300 records).
            if wait_for_arrival:
                continue
            caught_up = True
            for partition in partitions:
                if await consumer.position(partition) < end_offsets[partition]:
                    caught_up = False
                    break
            if caught_up:
                break
        return "", scanned, ""
    except Exception as exc:  # noqa: BLE001
        return None, scanned, f"topic scan failed: {sanitize_error_message(exc)}"
    finally:
        # aiokafka's coordinator shutdown cancels its own background tasks
        # and surfaces the CancelledError out of stop(). That is teardown
        # noise, not a verdict — catching only Exception here lets it
        # escape (CancelledError is a BaseException since 3.8) and destroys
        # an otherwise complete scan result. Observed live 2026-08-27
        # against the .201 dev-lane broker.
        try:
            await consumer.stop()
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            logger.warning("canary consumer stop failed: %s", exc)


async def _scan_quarantine_tail_via_aiokafka(
    bootstrap: str,
    topic: str,
    correlation_id: str,
    max_records: int,
    timeout_s: float,
) -> tuple[bool | None, int, str]:
    """Quarantine leg: did this correlation id already land in the sink?

    A question about the past, so no arrival wait. Boolean-shaped because
    the quarantine sink is one topic and the caller only needs the fact.
    """
    found, scanned, error = await _scan_topics_for_correlation(
        bootstrap,
        (topic,),
        correlation_id,
        max_records,
        timeout_s,
        wait_for_arrival=False,
    )
    if found is None:
        return None, scanned, error
    return bool(found), scanned, error


async def _readback_terminal_via_aiokafka(
    bootstrap: str,
    topics: tuple[str, ...],
    correlation_id: str,
    max_records: int,
    timeout_s: float,
) -> tuple[str | None, int, str]:
    """Link-4 leg (OMN-16931): is the terminal on the bus for this run?

    Waits for arrival, because the question is about a window that is still
    open when the ingress answers.
    """
    return await _scan_topics_for_correlation(
        bootstrap,
        topics,
        correlation_id,
        max_records,
        timeout_s,
        wait_for_arrival=True,
    )


async def _readback_projection_via_asyncpg(
    dsn: str,
    correlation_id: str,
    timeout_s: float,
) -> tuple[str | None, str]:
    """Link-2 leg (OMN-16963): what state does the PROJECTION hold for this run?

    Reads ``delegation_workflow_state`` scoped to the probe's own correlation
    id. This is the readback OMN-16025 link 2 asks for — from the projection,
    not from logs and not from the publish return.

    Returns the FSM state, ``""`` when the projection carries no row for this
    correlation id, and ``None`` when the read could not be completed at all.
    The last case is deliberately distinct: a read that failed is not a read
    that found nothing.
    """
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - asyncpg is a hard dep
        return None, f"asyncpg unavailable: {exc}"

    connection = None
    try:
        connection = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout_s)
        row = await asyncio.wait_for(
            connection.fetchrow(
                "SELECT state FROM delegation_workflow_state WHERE correlation_id = $1",
                correlation_id,
            ),
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - fails closed, never to a verdict
        return None, sanitize_error_message(exc)
    finally:
        if connection is not None:
            await connection.close()

    if row is None:
        return "", ""
    return str(row["state"] or ""), ""


def _extract_error(response: dict[str, object]) -> tuple[str, str]:
    """Pull (code, message) out of the ingress's typed error block."""
    error = response.get("error")
    if isinstance(error, dict):
        code = str(error.get("code", "") or "")
        message = str(error.get("message", "") or "")
        return code, message
    if isinstance(error, str) and error:
        return "", error
    return "", ""


def _extract_terminal(response: dict[str, object]) -> str:
    """Return the terminal event type, or '' when there is none.

    ``terminal_event`` is sometimes a bare string and sometimes an object
    carrying the type — accept both rather than reporting a real terminal as
    missing on a shape difference.
    """
    terminal = response.get("terminal_event")
    if isinstance(terminal, str):
        return terminal
    if isinstance(terminal, dict):
        for key in ("event_type", "type", "name"):
            value = terminal.get(key)
            if isinstance(value, str) and value:
                return value
        return "terminal_event"  # present but unnamed — still a terminal
    return ""


def _kill_switch_engaged(raw: str) -> bool:
    return raw.strip().lower() in _KILL_SWITCH_TRUTHY_VALUES


class HandlerChainCanary:
    """Fire one live delegation and report whether the chain carried it."""

    def __init__(
        self,
        ingress: TypeIngressPost | None = None,
        quarantine_scan: TypeQuarantineScan | None = None,
        terminal_readback: TypeTerminalReadback | None = None,
        projection_readback: TypeProjectionReadback | None = None,
        kill_switch_disabled: bool | None = None,
    ) -> None:
        self._ingress: TypeIngressPost = ingress or _post_skill_via_httpx
        self._quarantine_scan: TypeQuarantineScan = (
            quarantine_scan or _scan_quarantine_tail_via_aiokafka
        )
        self._terminal_readback: TypeTerminalReadback = (
            terminal_readback or _readback_terminal_via_aiokafka
        )
        self._projection_readback: TypeProjectionReadback = (
            projection_readback or _readback_projection_via_asyncpg
        )
        # Read at construction, overridable for tests, re-read in handle()
        # so a zero-arg contract-driven construction cannot miss it.
        # ONEX_EXCLUDE below: a scheduled canary's own *_DISABLED kill switch,
        # the same shape and rationale already accepted for
        # node_dlq_depth_monitor_effect, node_sync_revert_watchdog_effect and
        # node_evidence_autoclose_sweep_effect. The switch must be able to stop a
        # runtime that is ALREADY wired, so it cannot arrive through container
        # config resolved once at startup.
        env_disabled = _kill_switch_engaged(
            os.environ.get(_KILL_SWITCH_ENV_VAR, "")  # ONEX_EXCLUDE
        )
        self._kill_switch_ctor = (
            kill_switch_disabled if kill_switch_disabled is not None else env_disabled
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelChainCanaryRequest) -> ModelChainCanaryResult:
        # Re-read per run (ONEX_EXCLUDE, same rationale as the constructor): the
        # runtime builds handlers once and the canary then fires on a schedule for
        # the life of that process, so a value frozen at construction would make
        # the switch inert until a redeploy.
        env_disabled = _kill_switch_engaged(
            os.environ.get(_KILL_SWITCH_ENV_VAR, "")  # ONEX_EXCLUDE
        )
        if self._kill_switch_ctor or env_disabled:
            logger.warning(
                "%s is set — chain canary disabled, zero I/O performed.",
                _KILL_SWITCH_ENV_VAR,
            )
            return ModelChainCanaryResult(
                correlation_id=request.correlation_id,
                probe_correlation_id=uuid4(),
                verdict=EnumChainCanaryVerdict.SKIPPED_DISABLED,
                success=True,
                kill_switch_engaged=True,
                detail=(
                    f"{_KILL_SWITCH_ENV_VAR} set; no probe fired and no claim "
                    "is made about the chain."
                ),
                probe_url=request.probe_url,
                runtime_command=request.runtime_command,
                task_type=request.task_type,
                budget_ms=request.budget_ms,
                link_verdicts=_all_links_unevaluated(
                    f"{_KILL_SWITCH_ENV_VAR} set — no probe fired"
                ),
                links_proven=0,
                links_total=len(EnumChainLink),
                chain_proof_complete=False,
            )

        # AC1: minted here, per run, never caller-supplied.
        probe_correlation_id = uuid4()

        response, transport_error, elapsed_ms = await self._ingress(
            f"{request.probe_url}/skill",
            self._build_body(request, str(probe_correlation_id)),
            (request.budget_ms / 1000.0) + _CLIENT_SLACK_SECONDS,
        )

        base = {
            "correlation_id": request.correlation_id,
            "probe_correlation_id": probe_correlation_id,
            "probe_url": request.probe_url,
            "runtime_command": request.runtime_command,
            "task_type": request.task_type,
            "budget_ms": request.budget_ms,
            "elapsed_ms": elapsed_ms,
            "quarantine_topic": request.quarantine_topic,
        }

        if response is None:
            return ModelChainCanaryResult(
                **base,  # type: ignore[arg-type]
                verdict=EnumChainCanaryVerdict.INGRESS_UNREACHABLE,
                success=False,
                ingress_error_message=transport_error,
                detail=(
                    f"could not reach the runtime ingress at "
                    f"{request.probe_url}/skill: {transport_error}"
                ),
                link_verdicts=_build_link_verdicts(
                    request=request,
                    ingress_reachable=False,
                    terminal_readback_status=EnumTerminalReadbackStatus.SKIPPED_NOT_CONFIGURED,
                    terminal_topic="",
                    quarantine_status=EnumQuarantineCheckStatus.SKIPPED_NOT_CONFIGURED,
                    projection_readback_status=EnumProjectionReadbackStatus.SKIPPED_NOT_CONFIGURED,
                    projection_state="",
                    projection_error="",
                ),
                links_proven=0,
                links_total=len(EnumChainLink),
                chain_proof_complete=False,
            )

        ingress_ok = bool(response.get("ok", False))
        error_code, error_message = _extract_error(response)
        ingress_terminal_event = _extract_terminal(response)

        # Both broker legs run CONCURRENTLY. The terminal readback holds its
        # consumer open for the remainder of the budget, and running the
        # quarantine tail scan behind it would push that scan up to two
        # minutes past the event it is sampling for — on a sink that held
        # ~8.9M records when OMN-16767 was diagnosed, a 500-record tail can
        # roll well past the window in that time.
        readback_window_s = _readback_window_seconds(request, elapsed_ms)
        (
            (
                terminal_readback_status,
                terminal_topic,
                terminal_scanned,
                terminal_error,
            ),
            (
                quarantine_status,
                scanned,
                quarantine_error,
            ),
            (
                projection_readback_status,
                projection_state,
                projection_error,
            ),
        ) = await asyncio.gather(
            self._readback_terminal(
                request, str(probe_correlation_id), readback_window_s
            ),
            self._check_quarantine(request, str(probe_correlation_id)),
            self._readback_projection(
                request, str(probe_correlation_id), readback_window_s
            ),
        )

        verdict, detail = self._decide(
            request=request,
            terminal_readback_status=terminal_readback_status,
            terminal_topic=terminal_topic,
            terminal_readback_error=terminal_error,
            readback_window_s=readback_window_s,
            ingress_ok=ingress_ok,
            ingress_terminal_event=ingress_terminal_event,
            error_code=error_code,
            error_message=error_message,
            elapsed_ms=elapsed_ms,
            quarantine_status=quarantine_status,
            quarantine_error=quarantine_error,
        )

        link_verdicts = _build_link_verdicts(
            request=request,
            ingress_reachable=True,
            terminal_readback_status=terminal_readback_status,
            terminal_topic=terminal_topic,
            quarantine_status=quarantine_status,
            projection_readback_status=projection_readback_status,
            projection_state=projection_state,
            projection_error=projection_error,
        )
        links_proven = sum(
            1 for link in link_verdicts if link.status is EnumChainLinkStatus.PASS
        )

        return ModelChainCanaryResult(
            **base,  # type: ignore[arg-type]
            verdict=verdict,
            success=verdict
            in (
                EnumChainCanaryVerdict.GREEN,
                EnumChainCanaryVerdict.SKIPPED_DISABLED,
            ),
            detail=detail,
            ingress_ok=ingress_ok,
            ingress_error_code=error_code,
            ingress_error_message=error_message,
            ingress_terminal_event=ingress_terminal_event,
            terminal_event=terminal_topic,
            terminal_readback_status=terminal_readback_status,
            terminal_topic=terminal_topic,
            terminal_topics_scanned=_terminal_topics(request),
            terminal_readback_records_scanned=terminal_scanned,
            terminal_readback_window_seconds=readback_window_s,
            terminal_readback_error=terminal_error,
            quarantine_status=quarantine_status,
            quarantine_records_scanned=scanned,
            quarantine_error=quarantine_error,
            link_verdicts=link_verdicts,
            links_proven=links_proven,
            links_total=len(EnumChainLink),
            chain_proof_complete=links_proven == len(EnumChainLink),
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _build_body(
        request: ModelChainCanaryRequest, probe_correlation_id: str
    ) -> dict[str, object]:
        """The recorded dispatch shape (omnidash server/routes.ts:216-234).

        Kept deliberately identical to what the dashboard sends, so a green
        canary is evidence about the path real callers take rather than
        about a bespoke probe-only path that could drift away from it.
        """
        return {
            "command_name": request.runtime_command,
            "correlation_id": probe_correlation_id,
            "timeout_ms": request.budget_ms,
            "payload": {
                "prompt": request.prompt,
                "task_type": request.task_type,
                "source": "external-client",
                "wait": True,
                "correlation_id": probe_correlation_id,
                "max_tokens": request.max_tokens,
                "metadata": {
                    "requested_by": "chain-canary",
                    "source_surface": "scheduled-chain-canary",
                },
            },
        }

    async def _readback_projection(
        self,
        request: ModelChainCanaryRequest,
        probe_correlation_id: str,
        window_s: float,
    ) -> tuple[EnumProjectionReadbackStatus, str, str]:
        """Read delegation_workflow_state for this run's correlation id.

        Never consults the bus terminal. That separation is the point of the
        ticket: the broker says what landed on the topic, the projection says
        what the FSM did with it, and OMN-14843 is the proof those can
        disagree — 26 of 38 correlations stranded mid-FSM while the topic
        layer was healthy at the same moment.

        Scoped to the probe's own correlation id, never table-wide: a
        table-wide check would go green on somebody else's terminal row.
        """
        if not request.projection_dsn.strip():
            return EnumProjectionReadbackStatus.SKIPPED_NOT_CONFIGURED, "", ""

        state, error = await self._projection_readback(
            request.projection_dsn,
            probe_correlation_id,
            window_s,
        )
        if state is None:
            return (
                EnumProjectionReadbackStatus.ERROR,
                "",
                error or "projection readback failed",
            )
        if not state:
            return EnumProjectionReadbackStatus.ROW_ABSENT, "", ""
        if state.strip().upper() in _TERMINAL_FSM_STATES:
            return EnumProjectionReadbackStatus.TERMINAL, state, ""
        return EnumProjectionReadbackStatus.STRANDED, state, ""

    async def _readback_terminal(
        self,
        request: ModelChainCanaryRequest,
        probe_correlation_id: str,
        window_s: float,
    ) -> tuple[EnumTerminalReadbackStatus, str, int, str]:
        """Read the declared terminal topics for this run's correlation id.

        Never consults the ingress response. That separation is the point of
        the ticket: the ingress says what the request path believes, the
        broker says what actually landed, and only the second one is
        evidence.
        """
        if not request.terminal_bootstrap_servers.strip():
            return EnumTerminalReadbackStatus.SKIPPED_NOT_CONFIGURED, "", 0, ""

        found, scanned, error = await self._terminal_readback(
            request.terminal_bootstrap_servers,
            _terminal_topics(request),
            probe_correlation_id,
            request.terminal_scan_records,
            window_s,
        )
        if found is None:
            return (
                EnumTerminalReadbackStatus.ERROR,
                "",
                scanned,
                error or "terminal readback failed",
            )
        if found:
            return EnumTerminalReadbackStatus.FOUND, found, scanned, ""
        return EnumTerminalReadbackStatus.NOT_FOUND, "", scanned, ""

    async def _check_quarantine(
        self, request: ModelChainCanaryRequest, probe_correlation_id: str
    ) -> tuple[EnumQuarantineCheckStatus, int, str]:
        if not request.quarantine_bootstrap_servers.strip():
            return EnumQuarantineCheckStatus.SKIPPED_NOT_CONFIGURED, 0, ""

        # The DLQ write happens after the handler raises, so scanning the
        # instant the ingress answers can miss a record that is about to
        # appear — and a missed record downgrades a precise QUARANTINED
        # verdict into a vague TERMINAL_MISSING one.
        if request.settle_seconds > 0:
            await asyncio.sleep(request.settle_seconds)

        found, scanned, error = await self._quarantine_scan(
            request.quarantine_bootstrap_servers,
            request.quarantine_topic,
            probe_correlation_id,
            request.quarantine_scan_records,
            request.quarantine_timeout_seconds,
        )
        if found is None:
            return EnumQuarantineCheckStatus.ERROR, scanned, error or "scan failed"
        if found:
            return EnumQuarantineCheckStatus.FOUND, scanned, ""
        return EnumQuarantineCheckStatus.CLEAN, scanned, ""

    @staticmethod
    def _decide(
        *,
        request: ModelChainCanaryRequest,
        terminal_readback_status: EnumTerminalReadbackStatus,
        terminal_topic: str,
        terminal_readback_error: str,
        readback_window_s: float,
        ingress_ok: bool,
        ingress_terminal_event: str,
        error_code: str,
        error_message: str,
        elapsed_ms: int,
        quarantine_status: EnumQuarantineCheckStatus,
        quarantine_error: str,
    ) -> tuple[EnumChainCanaryVerdict, str]:
        """Rank the verdicts. Most specific diagnosis wins.

        QUARANTINED outranks everything because in the OMN-16767 incident
        several symptoms are true at once and only that one names the
        defect. A canary that reported "timed out" there would have sent
        someone looking at latency instead of at the dispatch seam.

        Below that, the ranking is: can we read the bus at all (fail closed
        if not) → is the terminal on the bus → and only then does the
        ingress response matter, and only to distinguish "the chain worked
        and the request path reported an error" from "the chain worked
        cleanly". The ingress response is never allowed to decide whether a
        terminal exists (OMN-16931).
        """
        if quarantine_status is EnumQuarantineCheckStatus.FOUND:
            return (
                EnumChainCanaryVerdict.QUARANTINED,
                (
                    "the probe's own correlation id was found in the "
                    f"quarantine sink {request.quarantine_topic} — a handler "
                    "received this event and refused or errored on it"
                    + (f" (ingress reported {error_code})" if error_code else "")
                ),
            )
        if quarantine_status is EnumQuarantineCheckStatus.ERROR:
            return (
                EnumChainCanaryVerdict.QUARANTINE_PROBE_FAILED,
                (
                    "the quarantine check was configured but could not run: "
                    f"{quarantine_error}. Failing closed — an unrunnable "
                    "check is not a passing one."
                ),
            )
        if (
            terminal_readback_status
            is EnumTerminalReadbackStatus.SKIPPED_NOT_CONFIGURED
        ):
            return (
                EnumChainCanaryVerdict.TERMINAL_READBACK_NOT_CONFIGURED,
                (
                    "no broker was configured for the terminal readback, so "
                    "this run has NO evidence about the terminal. Reporting "
                    "red rather than falling back to the ingress response: "
                    "that fallback is the OMN-16931 defect."
                ),
            )
        if terminal_readback_status is EnumTerminalReadbackStatus.ERROR:
            return (
                EnumChainCanaryVerdict.TERMINAL_READBACK_FAILED,
                (
                    "the terminal readback was configured but could not run: "
                    f"{terminal_readback_error}. Failing closed — an "
                    "unrunnable check is not a passing one."
                ),
            )
        if terminal_readback_status is EnumTerminalReadbackStatus.NOT_FOUND:
            reason = (
                "the ingress claimed terminal "
                f"{ingress_terminal_event!r} that the bus never carried — "
                "the ok=true-without-a-durable-terminal shape (OMN-15468)"
                if ingress_ok and ingress_terminal_event
                else (
                    f"ingress also returned ok=false ({error_code or 'no code'}: "
                    f"{error_message or 'no message'})"
                    if not ingress_ok
                    else "ingress returned ok=true and named no terminal"
                )
            )
            return (
                EnumChainCanaryVerdict.TERMINAL_MISSING,
                (
                    "the probe's correlation id was not on any declared "
                    f"terminal topic after reading the bus for "
                    f"{readback_window_s:.0f}s (budget {request.budget_ms} ms, "
                    f"ingress answered at {elapsed_ms} ms): {reason}"
                ),
            )

        # Terminal IS on the bus. The only remaining question is whether the
        # request path also reported success.
        quarantine_suffix = (
            " (quarantine check not configured — no claim made about the "
            "quarantine sink)"
            if quarantine_status is EnumQuarantineCheckStatus.SKIPPED_NOT_CONFIGURED
            else " and the quarantine sink is clean for this correlation id"
        )
        if not ingress_ok:
            return (
                EnumChainCanaryVerdict.INGRESS_ERROR_TERMINAL_PRESENT,
                (
                    f"the terminal IS on the bus ({terminal_topic}) for this "
                    "correlation id, but the ingress reported an error "
                    f"({error_code or 'no code'}: "
                    f"{error_message or 'no message'}). The chain carried the "
                    "request; the failure is in the request path, not a dead "
                    f"chain{quarantine_suffix}"
                ),
            )
        return (
            EnumChainCanaryVerdict.GREEN,
            (
                f"terminal read back off {terminal_topic} for this run's "
                f"correlation id; ingress answered ok in {elapsed_ms} ms "
                f"(budget {request.budget_ms} ms){quarantine_suffix}"
            ),
        )


# -- per-link verdicts -------------------------------------------------


def _all_links_unevaluated(reason: str) -> tuple[ModelChainLinkVerdict, ...]:
    return tuple(
        ModelChainLinkVerdict(
            link=link,
            status=EnumChainLinkStatus.NOT_EVALUATED,
            detail=reason,
            owning_ticket=(
                _LINK5_OWNING_TICKET if link is EnumChainLink.LEDGER_REPLAY else ""
            ),
        )
        for link in EnumChainLink
    )


def _build_link_verdicts(
    *,
    request: ModelChainCanaryRequest,
    ingress_reachable: bool,
    terminal_readback_status: EnumTerminalReadbackStatus,
    terminal_topic: str,
    quarantine_status: EnumQuarantineCheckStatus,
    projection_readback_status: EnumProjectionReadbackStatus,
    projection_state: str,
    projection_error: str,
) -> tuple[ModelChainLinkVerdict, ...]:
    """One status per OMN-16025 link, so a 4/5 probe cannot read as 5/5.

    The remaining NO_LEG row is the honest part. It is not a failure and
    not a pass — it is a link this probe has no instrument for, and it
    names the ticket that owes the instrument.
    """
    link2, link2_detail = _link_two(
        projection_readback_status, projection_state, projection_error
    )
    link4, link4_detail = _link_four(terminal_readback_status, terminal_topic)
    link3, link3_detail = _link_three(
        request=request,
        terminal_readback_status=terminal_readback_status,
        terminal_topic=terminal_topic,
        quarantine_status=quarantine_status,
    )
    if not ingress_reachable:
        link2 = link3 = link4 = EnumChainLinkStatus.NOT_EVALUATED
        link2_detail = link3_detail = link4_detail = (
            "not evaluated — the ingress was unreachable, so nothing "
            "downstream of it was observed"
        )

    return (
        ModelChainLinkVerdict(
            link=EnumChainLink.INGRESS_ACCEPTED,
            status=(
                EnumChainLinkStatus.PASS
                if ingress_reachable
                else EnumChainLinkStatus.FAIL
            ),
            detail=(
                f"the live {request.probe_url}/skill ingress answered and "
                "the probe's correlation id went on the wire"
                if ingress_reachable
                else "the live ingress could not be reached at all"
            ),
        ),
        ModelChainLinkVerdict(
            link=EnumChainLink.ROUTING_PROJECTED,
            status=link2,
            detail=link2_detail,
        ),
        ModelChainLinkVerdict(
            link=EnumChainLink.DELEGATED_EXECUTION,
            status=link3,
            detail=link3_detail,
        ),
        ModelChainLinkVerdict(
            link=EnumChainLink.TERMINAL_ON_BUS,
            status=link4,
            detail=link4_detail,
        ),
        ModelChainLinkVerdict(
            link=EnumChainLink.LEDGER_REPLAY,
            status=EnumChainLinkStatus.NO_LEG,
            detail=(
                "this canary assembles no ledger chain, runs no replay, "
                "and invokes no tier-2 verifier"
            ),
            owning_ticket=_LINK5_OWNING_TICKET,
        ),
    )


def _link_two(
    projection_readback_status: EnumProjectionReadbackStatus,
    projection_state: str,
    projection_error: str,
) -> tuple[EnumChainLinkStatus, str]:
    """Link 2: routing decision PUBLISHED and PROJECTED.

    Read from ``delegation_workflow_state`` for this run's own correlation id.
    STRANDED is the OMN-14843 signature and is the reason this leg exists: a
    lane can terminalize on the bus while the projection leaves the row
    mid-FSM, and every other leg of this canary watches the layer that stays
    healthy in that condition.
    """
    if projection_readback_status is EnumProjectionReadbackStatus.TERMINAL:
        return (
            EnumChainLinkStatus.PASS,
            f"delegation_workflow_state reached {projection_state} for this "
            "run's own correlation id — projection evidence, not logs",
        )
    if projection_readback_status is EnumProjectionReadbackStatus.STRANDED:
        return (
            EnumChainLinkStatus.FAIL,
            f"the projection row for this correlation id stopped at "
            f"{projection_state} — the OMN-14843 signature, and invisible to "
            "every other leg of this probe",
        )
    if projection_readback_status is EnumProjectionReadbackStatus.ROW_ABSENT:
        return (
            EnumChainLinkStatus.FAIL,
            "delegation_workflow_state carried no row at all for this "
            "correlation id, so no routing decision was projected",
        )
    if projection_readback_status is EnumProjectionReadbackStatus.ERROR:
        return (
            EnumChainLinkStatus.ERROR,
            "the projection readback could not be completed, so no claim is "
            f"made about the routing decision: {projection_error}",
        )
    return (
        EnumChainLinkStatus.NOT_CONFIGURED,
        "no projection store configured for the readback — SKIP is not PASS",
    )


def _link_four(
    terminal_readback_status: EnumTerminalReadbackStatus, terminal_topic: str
) -> tuple[EnumChainLinkStatus, str]:
    """Link 4: emission OUTBOX-CONFIRMED via broker readback."""
    if terminal_readback_status is EnumTerminalReadbackStatus.FOUND:
        return (
            EnumChainLinkStatus.PASS,
            f"read back off {terminal_topic} for this run's own "
            "correlation id — broker evidence, not publish-return",
        )
    if terminal_readback_status is EnumTerminalReadbackStatus.NOT_FOUND:
        return (
            EnumChainLinkStatus.FAIL,
            "the declared terminal topics were read for the budget "
            "window and carried nothing for this correlation id",
        )
    if terminal_readback_status is EnumTerminalReadbackStatus.ERROR:
        return (
            EnumChainLinkStatus.ERROR,
            "the broker readback could not be completed, so no claim is "
            "made about the terminal",
        )
    return (
        EnumChainLinkStatus.NOT_CONFIGURED,
        "no broker configured for the terminal readback — SKIP is not PASS",
    )


def _link_three(
    *,
    request: ModelChainCanaryRequest,
    terminal_readback_status: EnumTerminalReadbackStatus,
    terminal_topic: str,
    quarantine_status: EnumQuarantineCheckStatus,
) -> tuple[EnumChainLinkStatus, str]:
    """Link 3: delegated execution completes.

    Derived from WHICH terminal topic carried the correlation id, plus
    the quarantine leg. Stated with its own caveat rather than sold as
    more than it is: OMN-15468 is live proof that on this lane a FAILED
    run can be republished onto the SUCCESS terminal, so arrival topic
    is strong evidence, not proof. Closing OMN-15468 is what upgrades
    this row.
    """
    if quarantine_status is EnumQuarantineCheckStatus.FOUND:
        return (
            EnumChainLinkStatus.FAIL,
            "a handler received this event and refused or errored on it "
            f"({request.quarantine_topic})",
        )
    if terminal_readback_status is EnumTerminalReadbackStatus.ERROR:
        return (
            EnumChainLinkStatus.ERROR,
            "the terminal readback could not run, so execution status is unknown",
        )
    if terminal_readback_status is EnumTerminalReadbackStatus.SKIPPED_NOT_CONFIGURED:
        return (
            EnumChainLinkStatus.NOT_CONFIGURED,
            "no broker configured, so execution status is unknown",
        )
    if terminal_topic in tuple(request.terminal_failure_topics):
        return (
            EnumChainLinkStatus.FAIL,
            f"the terminal landed on the failure topic {terminal_topic}",
        )
    if terminal_topic in tuple(request.terminal_success_topics):
        return (
            EnumChainLinkStatus.PASS,
            f"the terminal landed on the success topic {terminal_topic} "
            "(arrival topic is strong evidence, not proof, while "
            "OMN-15468 is open)",
        )
    return (
        EnumChainLinkStatus.FAIL,
        "no terminal was read back for this correlation id",
    )


def render_receipt(result: ModelChainCanaryResult) -> str:
    """Render the run receipt as pretty JSON for the workflow job summary."""
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
