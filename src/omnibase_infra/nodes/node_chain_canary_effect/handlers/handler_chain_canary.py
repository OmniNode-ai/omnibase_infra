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
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_quarantine_check_status import (
    EnumQuarantineCheckStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

__all__ = ["HandlerChainCanary"]

# Kill switch, checked first and unconditionally. A SEPARATE variable from
# every other sweep's switch on purpose: silencing one canary must never
# silently silence another.
_KILL_SWITCH_ENV_VAR = "ONEX_CHAIN_CANARY_DISABLED"

# Extra client-side slack over the runtime's own budget. Without it the HTTP
# client can abort first and every slow-but-working chain reads as
# INGRESS_UNREACHABLE — a false RED that would teach people to ignore this
# canary, which is the one outcome worse than not having it.
_CLIENT_SLACK_SECONDS = 15.0

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


async def _scan_quarantine_tail_via_aiokafka(
    bootstrap: str,
    topic: str,
    correlation_id: str,
    max_records: int,
    timeout_s: float,
) -> tuple[bool | None, int, str]:
    """Scan the last ``max_records`` of ``topic`` for ``correlation_id``.

    Returns ``(None, scanned, error)`` when the scan could not be completed —
    the caller turns that into a failing verdict rather than a pass. A check
    that could not run is not a check that passed.
    """
    from aiokafka import AIOKafkaConsumer

    from omnibase_infra.event_bus.kafka_auth import (
        build_aiokafka_auth_kwargs_from_env,
    )

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
        topic,
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
        partitions = sorted(consumer.assignment(), key=lambda tp: tp.partition)
        if not partitions:
            return (
                None,
                0,
                f"quarantine topic {topic!r} resolved no partitions "
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
        if empty:
            # Every partition is empty: nothing quarantined, definitively.
            return False, 0, ""

        deadline = time.monotonic() + timeout_s
        while scanned < max_records and time.monotonic() < deadline:
            batches = await consumer.getmany(timeout_ms=1_000, max_records=500)
            if not batches:
                break
            for records in batches.values():
                for record in records:
                    scanned += 1
                    if record.value and needle in record.value:
                        return True, scanned, ""
            # Stop once every partition has been read up to its high-water
            # mark; otherwise a quiet topic would burn the whole timeout.
            # Written as an explicit loop on purpose: `position()` is a
            # coroutine, and folding it into an all(...) generator builds an
            # async generator that all() cannot iterate ("TypeError:
            # 'async_generator' object is not iterable" — hit live on
            # 2026-08-27, after the scan had already read its 300 records).
            caught_up = True
            for partition in partitions:
                if await consumer.position(partition) < end_offsets[partition]:
                    caught_up = False
                    break
            if caught_up:
                break
        return False, scanned, ""
    except Exception as exc:  # noqa: BLE001
        return None, scanned, f"quarantine scan failed: {sanitize_error_message(exc)}"
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
            logger.warning("quarantine consumer stop failed: %s", exc)


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


class HandlerChainCanary:
    """Fire one live delegation and report whether the chain carried it."""

    def __init__(
        self,
        ingress: TypeIngressPost | None = None,
        quarantine_scan: TypeQuarantineScan | None = None,
        kill_switch_disabled: bool | None = None,
    ) -> None:
        self._ingress: TypeIngressPost = ingress or _post_skill_via_httpx
        self._quarantine_scan: TypeQuarantineScan = (
            quarantine_scan or _scan_quarantine_tail_via_aiokafka
        )
        # Read at construction, overridable for tests, re-read in handle()
        # so a zero-arg contract-driven construction cannot miss it.
        self._kill_switch_ctor = (
            kill_switch_disabled
            if kill_switch_disabled is not None
            else bool(os.environ.get(_KILL_SWITCH_ENV_VAR, ""))
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelChainCanaryRequest) -> ModelChainCanaryResult:
        if self._kill_switch_ctor or os.environ.get(_KILL_SWITCH_ENV_VAR, ""):
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
            )

        ingress_ok = bool(response.get("ok", False))
        error_code, error_message = _extract_error(response)
        terminal_event = _extract_terminal(response)
        terminal_landed = ingress_ok and bool(terminal_event)

        quarantine_status, scanned, quarantine_error = await self._check_quarantine(
            request, str(probe_correlation_id)
        )

        verdict, detail = self._decide(
            request=request,
            terminal_landed=terminal_landed,
            terminal_event=terminal_event,
            ingress_ok=ingress_ok,
            error_code=error_code,
            error_message=error_message,
            elapsed_ms=elapsed_ms,
            quarantine_status=quarantine_status,
            quarantine_error=quarantine_error,
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
            terminal_event=terminal_event,
            quarantine_status=quarantine_status,
            quarantine_records_scanned=scanned,
            quarantine_error=quarantine_error,
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
        terminal_landed: bool,
        terminal_event: str,
        ingress_ok: bool,
        error_code: str,
        error_message: str,
        elapsed_ms: int,
        quarantine_status: EnumQuarantineCheckStatus,
        quarantine_error: str,
    ) -> tuple[EnumChainCanaryVerdict, str]:
        """Rank the verdicts. Most specific diagnosis wins.

        QUARANTINED outranks TERMINAL_MISSING because in the OMN-16767
        incident BOTH are true and only the first one names the defect. A
        canary that reported "timed out" there would have sent someone
        looking at latency instead of at the dispatch seam.
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
        if not terminal_landed:
            reason = (
                f"ingress returned ok=false ({error_code or 'no code'}: "
                f"{error_message or 'no message'})"
                if not ingress_ok
                else "ingress returned ok=true but carried no terminal event "
                "(a cheerful accept is not proof the chain ran — OMN-16027)"
            )
            return (
                EnumChainCanaryVerdict.TERMINAL_MISSING,
                (
                    f"no terminal event inside the {request.budget_ms} ms "
                    f"budget after {elapsed_ms} ms: {reason}"
                ),
            )

        suffix = (
            " (quarantine check not configured — no claim made about the "
            "quarantine sink)"
            if quarantine_status is EnumQuarantineCheckStatus.SKIPPED_NOT_CONFIGURED
            else " and the quarantine sink is clean for this correlation id"
        )
        return (
            EnumChainCanaryVerdict.GREEN,
            (
                f"terminal {terminal_event!r} landed in {elapsed_ms} ms "
                f"(budget {request.budget_ms} ms){suffix}"
            ),
        )


def render_receipt(result: ModelChainCanaryResult) -> str:
    """Render the run receipt as pretty JSON for the workflow job summary."""
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
