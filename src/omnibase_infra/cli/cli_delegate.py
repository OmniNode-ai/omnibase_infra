# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` — single-command local-LLM delegation (OMN-13096).

Phase 2b of the skill-output-suppression slice
(``docs/plans/2026-06-12-skill-output-suppression-plan.md``): the
governing simplification is that **a dispatch skill IS one CLI call**.

``onex delegate "<prompt>" [--task-type X] [--max-tokens N]`` wraps, inside
the CLI entrypoint:

1. task-type classification (when ``--task-type`` is omitted),
2. typed payload construction written to ``<state-root>/tmp/<run_id>.json``
   (run_id-suffixed scratch — never ``/tmp``; ``feedback_no_tmp_use_workspace``),
3. resolution of the packaged ``node_delegate_skill_orchestrator`` contract,
4. dispatch through the OMN-13094 receipt-mode path
   (:func:`omnibase_infra.cli.receipt_mode.run_receipt_mode`).

Bus target (OMN-13532, default flipped under OMN-14376). ``--bus`` is
OPTIONAL, never required for a delegation to reach the shared platform
substrate. When omitted, :func:`resolve_default_bus` mirrors the SAME
probe-then-select precedence the runtime kernel's ``select_event_bus``
already applies (``service_kernel.py`` / ``backends/auto_configure.py``):
``KAFKA_BOOTSTRAP_SERVERS`` unset -> ``inmemory``; set and the broker probes
HEALTHY/AUTHORITATIVE (``backends/backend_probe.py::probe_kafka``, a bounded
TCP + topic-list check) -> ``kafka``; set but unreachable/unhealthy (e.g. the
OMN-14380 advertised-listener gap for an off-box caller) -> graceful fallback
to ``inmemory`` with a WARNING logged (stderr / capture, never stdout) so the
silent-local-SQLite failure mode is never repeated. This makes delegation use
"the same event bus the rest of the system is configured with" BY DEFAULT — no
``--bus kafka`` flag required. When resolved (or explicitly passed) to
``kafka``, the typed ``ModelDelegateSkillRequest`` command is published to the
delegate-skill command topic declared in the contract's
``event_bus.publish_topics``, where the deployed
``node_delegate_skill_orchestrator`` consumer picks it up
(``feedback_bus_is_the_transport`` — the bus is THE transport) and its
projection lands in the shared ``delegation_events`` table, not per-machine
SQLite. ``--bus``/``--kafka-bootstrap`` remain explicit OVERRIDES for forcing a
specific bus; both flow straight through ``backend_overrides`` to
``RuntimeLocal``, which never hardcodes the broker.

stdout receives exactly ONE
:class:`~omnibase_core.models.dispatch.model_skill_result.ModelSkillResult`
JSON whose ``result`` is the FULL
``ModelDelegateSkillResponse`` (status, response, model_name, provider,
task_type, quality_gate_passed, metrics). RuntimeLocal logs, envelope dumps,
and progress go to the capture file + artifact store and never reach the
caller. Non-zero exit on failure.

This replaces the multi-step ``omniclaude`` delegate shim
(payload temp file + ``cd omnimarket`` + ``onex node`` log flood +
``cat workflow_result.json``) with one command, one typed result.

Correlation identity and hard timeout (OMN-14397). Two invocations issued back
to back from the same working directory/state-root were observed sharing a
``correlation_id`` and a hung Kafka-bus call outliving its own ``--timeout``,
requiring a manual ``kill`` on ``.201``. This CLI now mints ``correlation_id``
fresh per invocation and writes it explicitly into the payload rather than
leaving it to an implicit downstream default, and wraps the receipt-mode
dispatch in a ``SIGALRM``-based hard backstop: ``RuntimeLocal``'s own
``asyncio.wait_for`` timeout only preempts at an ``await`` point, so a
response-listener stuck in a synchronous, non-cooperative blocking call never
yields control back and that timeout silently never fires. ``SIGALRM``
interrupts blocking syscalls too, so it aborts the call (and reports a clear
error) even when the hang is not asyncio-cooperative.

.. versionadded:: OMN-13096
.. versionchanged:: OMN-14397
   Fresh ``correlation_id`` per invocation; hard ``SIGALRM`` timeout backstop.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click

from omnibase_infra.backends.backend_probe import probe_kafka
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.cli.cli_node import _resolve_packaged_contract
from omnibase_infra.cli.receipt_mode import (
    default_emit_socket_path,
    run_receipt_mode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DELEGATE_NODE_NAME",
    "DELEGATE_SOURCE",
    "DEFAULT_TASK_TYPE",
    "TASK_TYPE_CHOICES",
    "BUS_CHOICES",
    "DEFAULT_BUS",
    "DelegateTimeoutExceededError",
    "build_backend_overrides",
    "classify_task_type",
    "resolve_default_bus",
    "run_delegate",
]

# The omnimarket node that owns the consumer-facing delegation contract. It is
# registered under the ``onex.nodes`` entry-point group and resolvable from
# this CLI's environment (the delegate node ships its packaged contract.yaml).
DELEGATE_NODE_NAME = "node_delegate_skill_orchestrator"

# Registered adapter source for ``ModelDelegateSkillRequest`` (Literal field).
DELEGATE_SOURCE = "claude-code"

# Fallback classification when no keyword matches (the prompt.md default).
DEFAULT_TASK_TYPE = "research"

# The MVP task taxonomy the delegate node accepts (a subset of the
# ``ModelDelegateSkillRequest`` Literal — the surfaces the shim exposed).
# Source of truth for routing is the node contract's allowed_task_types; this
# CLI exposes the same classification mapping the legacy skill markdown used.
TASK_TYPE_CHOICES = (
    "test",
    "document",
    "research",
    "code_generation",
    "refactor",
    "reasoning",
    "review",
)

# Event-bus targets the CLI can select (OMN-13532). ``inmemory`` runs the
# orchestrator in-process (default, no broker). ``kafka`` publishes the typed
# command to the live broker so a deployed runtime consumer dispatches it.
# These mirror ``RuntimeLocal.SUPPORTED_EVENT_BUS_VALUES`` — the runtime is the
# source of truth and rejects anything outside that set.
BUS_CHOICES = ("inmemory", "kafka")

# Fallback bus when auto-resolution cannot select ``kafka`` (no broker
# configured, or a configured broker that fails the health probe): in-process,
# fully self-contained. This is the SAME value ``resolve_default_bus`` returns
# for both of those cases — it is not merely a historical default anymore, but
# the fail-safe floor auto-resolution always lands on when the shared bus is
# not provably reachable.
DEFAULT_BUS = "inmemory"


def resolve_default_bus(*, kafka_bootstrap: str | None = None) -> tuple[str, str]:
    """Resolve the bus ``--bus`` defaults to when the flag is omitted (OMN-14376).

    Mirrors the SAME probe-then-select precedence the runtime kernel already
    applies (``service_kernel.py`` lines ~1013-1120,
    ``backends/auto_configure.py::select_event_bus``) so delegation defaults to
    "the event bus the rest of the system is configured with" instead of a
    source-hardcoded ``inmemory`` — the OMN-14376 root cause.

    Delegates entirely to :func:`omnibase_infra.backends.backend_probe.probe_kafka`
    for both the ``KAFKA_BOOTSTRAP_SERVERS`` resolution AND the health check —
    this module never reads the environment variable itself (the
    ``check-env-reads`` CI gate blocks raw environment reads outside the
    approved overlay-resolved config surfaces; ``probe_kafka`` is the existing,
    already-approved boundary for this exact lookup).

    Precedence:
      1. No ``kafka_bootstrap`` override and ``KAFKA_BOOTSTRAP_SERVERS`` unset
         -> ``probe_kafka`` short-circuits to ``DISCOVERED`` with no network
         call -> ``("inmemory", <reason>)``. The steady state for a truly
         bus-less local dev box.
      2. A bootstrap IS configured (override or env) and ``probe_kafka``
         reports ``HEALTHY``/``AUTHORITATIVE`` -> ``("kafka", <reason>)``.
      3. A bootstrap IS configured but the probe reports anything weaker
         (``DISCOVERED``/``REACHABLE`` — e.g. TCP connects but the broker's
         advertised listener does not route back to this caller, the
         OMN-14380 off-box gap) -> graceful fallback to
         ``("inmemory", <reason>)``. The caller is expected to log this at
         WARNING so a stale/unhealthy configured broker degrades loudly to
         the safe local default rather than hanging ``onex delegate`` on an
         unreachable broker.

    Only called when ``--bus`` is NOT explicitly supplied — an explicit
    ``--bus`` (kafka or inmemory) is never second-guessed by this probe.
    """
    probe = probe_kafka(bootstrap_servers=kafka_bootstrap)
    if probe.state in (EnumProbeState.HEALTHY, EnumProbeState.AUTHORITATIVE):
        return "kafka", probe.reason
    return DEFAULT_BUS, f"{probe.state.name}: {probe.reason}"


def build_backend_overrides(*, bus: str, kafka_bootstrap: str | None) -> dict[str, str]:
    """Build the ``backend_overrides`` map for ``run_receipt_mode``/``RuntimeLocal``.

    ``bus`` selects the event-bus backend (``inmemory`` or ``kafka``). For
    ``kafka``, an optional ``kafka_bootstrap`` (``host:port``) routes through
    ``EventBusKafka.from_bootstrap`` so the live broker is targeted without
    process-wide environment mutation; omitting it lets the Kafka bus resolve
    its bootstrap from ``KAFKA_BOOTSTRAP_SERVERS``. ``kafka_bootstrap`` is only
    meaningful for ``bus="kafka"`` and is rejected otherwise so a typo (e.g.
    passing a broker with the default in-memory bus) fails loud rather than
    silently running in-process.
    """
    if bus not in BUS_CHOICES:
        raise ValueError(
            f"Unsupported bus {bus!r}. Choose one of: {', '.join(BUS_CHOICES)}."
        )
    if bus != "kafka" and kafka_bootstrap is not None:
        raise ValueError(
            f"--kafka-bootstrap is only valid with --bus kafka (got --bus {bus})."
        )
    overrides: dict[str, str] = {"event_bus": bus}
    if kafka_bootstrap is not None:
        overrides["kafka_bootstrap"] = kafka_bootstrap
    return overrides


# Ordered keyword → task_type rules (first match wins). Lifted verbatim from
# the legacy ``delegate/prompt.md`` classification table so behavior is
# unchanged; the glue now lives in the CLI, not skill markdown.
_CLASSIFICATION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("test", "pytest", "unit test", "assert"), "test"),
    (("document", "docstring", "readme", "explain how"), "document"),
    (("refactor", "cleanup", "simplify"), "refactor"),
    (("review", "audit", "check"), "review"),
    (("reason", "think through", "decide", "compare"), "reasoning"),
    (("write", "create", "implement", "build", "generate"), "code_generation"),
)


def classify_task_type(prompt: str) -> str:
    """Classify ``prompt`` into a delegate task type via keyword match.

    First matching rule wins; falls back to :data:`DEFAULT_TASK_TYPE` when no
    keyword is present. Matching is case-insensitive against the lowercased
    prompt. This is the same table the legacy skill markdown carried — the
    glue moved into the CLI per the plan's single-command simplification.
    """
    lowered = prompt.lower()
    for keywords, task_type in _CLASSIFICATION_RULES:
        if any(keyword in lowered for keyword in keywords):
            return task_type
    return DEFAULT_TASK_TYPE


def _write_payload(
    *,
    prompt: str,
    task_type: str,
    max_tokens: int | None,
    state_root: Path,
    run_id: uuid.UUID,
    correlation_id: uuid.UUID,
) -> Path:
    """Write the delegation input payload to run_id-suffixed scratch.

    Scratch lives under ``<state-root>/tmp/`` (never ``/tmp`` —
    ``feedback_no_tmp_use_workspace``). The payload validates against the
    delegate node's input model (``ModelDelegateSkillRequest``); only the
    fields the consumer supplies are set.

    When ``max_tokens`` is ``None`` (no explicit ``--max-tokens`` override) the
    key is omitted from the payload entirely, so the delegate node resolves the
    response budget per-backend from its routing contract rather than from a
    CLI-side default.

    ``correlation_id`` (OMN-14397) is written explicitly rather than left for
    the request model's ``default_factory`` to decide implicitly: the CLI is
    the one place guaranteed to mint a fresh identity per invocation (a new
    OS process every time), so it owns this run's tracing identity end to end
    instead of delegating that responsibility downstream.
    """
    tmp_dir = state_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_dir / f"delegate-input-{run_id}.json"
    payload: dict[str, object] = {
        "prompt": prompt,
        "task_type": task_type,
        "source": DELEGATE_SOURCE,
        "correlation_id": str(correlation_id),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    payload_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return payload_path


class DelegateTimeoutExceededError(BaseException):
    """The CLI-level hard timeout backstop fired (OMN-14397).

    ``RuntimeLocal`` enforces its own timeout cooperatively via
    ``asyncio.wait_for`` (``omnibase_core.runtime.runtime_local``), which only
    preempts at an ``await`` point. A response-listener stuck in a
    synchronous, non-cooperative blocking call (e.g. a raw blocking socket
    read on the Kafka path) never yields control back to the event loop, so
    that timeout silently never fires and the CLI process hangs past its own
    declared ``--timeout``, requiring a manual ``kill`` (observed on
    ``.201``). This exception signals the ``SIGALRM``-based backstop below,
    which fires unconditionally after the grace window regardless of what the
    process is doing — including inside a blocking syscall.

    Deliberately subclasses :class:`BaseException`, not :class:`Exception`
    (round-2 fix, OMN-14397). ``run_receipt_mode`` wraps the exact call that
    hangs in a broad ``except Exception as exc:`` (``receipt_mode.py`` — logs
    and continues rather than re-raising, by design, for genuine runtime
    failures); an ``Exception``-based timeout signal fired mid-call would be
    silently swallowed there, and the CLI's ``except
    DelegateTimeoutExceededError`` in :func:`run_delegate` would never see
    it — the exact same reason :class:`KeyboardInterrupt` and
    :class:`SystemExit` are direct ``BaseException`` subclasses rather than
    ``Exception`` subclasses. A signal-driven abort must not be catchable by
    ordinary application error handling anywhere on the call path, including
    inside ``RuntimeLocal`` (``omnibase_core.runtime.runtime_local``, which
    has its own ``except Exception`` blocks around the hang) and inside
    ``run_receipt_mode``'s own follow-on artifact/telemetry I/O
    (``receipt_mode.py`` lines ~600-732, which run after the guarded call and
    would otherwise have zero timeout coverage once the one-shot
    ``signal.alarm`` had already fired and been swallowed once).
    """


# Grace window added on top of the caller's declared --timeout before the
# hard SIGALRM backstop fires. Gives RuntimeLocal's own cooperative
# asyncio.wait_for timeout the first chance to exit cleanly with a proper
# TIMEOUT receipt; the backstop only fires when that path itself failed to
# preempt — the exact defect this guards against.
_HARD_TIMEOUT_GRACE_SECONDS = 10


@contextmanager
def _hard_timeout(seconds: int) -> Iterator[None]:
    """Enforce a hard wall-clock timeout via ``SIGALRM`` (POSIX only).

    Unlike ``asyncio.wait_for``, ``SIGALRM`` interrupts blocking syscalls, so
    it aborts a hung delegation call even when the hang is inside
    non-cooperative blocking I/O (OMN-14397). No-ops on platforms without
    ``SIGALRM`` (e.g. Windows) — timeout enforcement there is best-effort via
    ``RuntimeLocal``'s internal ``asyncio.wait_for`` only.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _on_alarm(signum: int, frame: object) -> None:
        raise DelegateTimeoutExceededError(
            f"onex delegate: exceeded hard timeout of {seconds}s "
            "(declared --timeout plus grace window) — aborting hung call."
        )

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@click.command("delegate")
@click.argument("prompt")
@click.option(
    "--task-type",
    "task_type",
    type=click.Choice(TASK_TYPE_CHOICES),
    default=None,
    help=(
        "Task classification for routing. Omit to auto-classify from the "
        "prompt keywords (default fallback: research)."
    ),
)
@click.option(
    "--max-tokens",
    "max_tokens",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Optional explicit override for the delegated LLM response budget. "
        "Omit to let the delegate node resolve max_tokens per-backend from its "
        "routing contract (no CLI-side default)."
    ),
)
@click.option(
    "--bus",
    "bus",
    type=click.Choice(BUS_CHOICES),
    default=None,
    help=(
        "Event-bus backend. Omit to auto-resolve (OMN-14376): 'kafka' when "
        "KAFKA_BOOTSTRAP_SERVERS is configured and the broker probes healthy "
        "— the SAME bus the rest of the system is configured with, so the "
        "delegation lands in the shared delegation_events projection — else "
        "'inmemory'. Pass explicitly to force a specific bus regardless of "
        "the probe: 'inmemory' runs the orchestrator in-process (no broker); "
        "'kafka' publishes the typed delegate-skill command to the broker so "
        "a deployed runtime consumer dispatches it."
    ),
)
@click.option(
    "--kafka-bootstrap",
    "kafka_bootstrap",
    type=str,
    default=None,
    help=(
        "Kafka bootstrap servers (host:port) for --bus kafka. Omit to resolve "
        "from KAFKA_BOOTSTRAP_SERVERS, for example from ~/.omnibase/.env. "
        "Only valid with --bus kafka."
    ),
)
@click.option(
    "--state-root",
    type=click.Path(path_type=Path),
    default=".onex_state",
    show_default=True,
    help="Root directory for disk state, scratch payloads, and captures.",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    show_default=True,
    help="Max delegation time in seconds.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Capture DEBUG-level logging (still routed to the capture file).",
)
@click.option(
    "--emit-socket",
    "emit_socket",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Unix socket of the emit daemon for capture events (default: "
        "~/.claude/emit.sock). Unreachable daemon => events spool under "
        "<state-root>/emit_spool/ for later replay."
    ),
)
def delegate_command(
    prompt: str,
    task_type: str | None,
    max_tokens: int | None,
    bus: str | None,
    kafka_bootstrap: str | None,
    state_root: Path,
    timeout: int,
    verbose: bool,
    emit_socket: Path | None,
) -> None:
    """Delegate PROMPT to a local LLM and print exactly one typed result.

    stdout carries exactly ONE ``ModelSkillResult[ModelDelegateSkillResponse]``
    JSON — the full LLM response and metrics, never truncated. RuntimeLocal
    logs go to a capture file + the content-addressed artifact store, never to
    stdout. Exits non-zero on failure.

    \b
    Examples:
        onex delegate "explain what a calendar app needs"
        onex delegate "write a Python HTTP server" --task-type code_generation
        onex delegate "analyze the routing architecture" --max-tokens 4096
        # Publish through the configured Kafka broker so a deployed runtime dispatches it:
        onex delegate "document the router" --bus kafka --kafka-bootstrap "$KAFKA_BOOTSTRAP_SERVERS"
    """
    try:
        exit_code = run_delegate(
            prompt=prompt,
            task_type=task_type,
            max_tokens=max_tokens,
            bus=bus,
            kafka_bootstrap=kafka_bootstrap,
            state_root=state_root,
            timeout=timeout,
            verbose=verbose,
            emit_socket=emit_socket,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    sys.exit(exit_code)


def run_delegate(
    *,
    prompt: str,
    task_type: str | None,
    max_tokens: int | None,
    bus: str | None = None,
    kafka_bootstrap: str | None = None,
    state_root: Path,
    timeout: int,
    verbose: bool,
    emit_socket: Path | None,
) -> int:
    """Build the payload, resolve the contract, and dispatch in receipt mode.

    Returns the process exit code. Payload construction, node dispatch, and
    result extraction are all internal — the caller supplies a prompt and
    receives one typed receipt on stdout.

    ``bus`` selects the event-bus backend. ``None`` (the CLI default, OMN-14376)
    auto-resolves via :func:`resolve_default_bus` — ``kafka`` when
    ``KAFKA_BOOTSTRAP_SERVERS`` is configured and probes healthy, else
    ``inmemory`` — so delegation reaches the shared platform substrate BY
    DEFAULT, with no ``--bus kafka`` flag required. An explicit ``"inmemory"``
    or ``"kafka"`` is never second-guessed. ``kafka_bootstrap`` optionally
    overrides the broker when the resolved/explicit bus is ``"kafka"`` — it is
    a usage error to supply it without also explicitly requesting
    ``--bus kafka`` (a bare ``--kafka-bootstrap`` is never silently absorbed
    into the auto-resolved default). Both flow through ``backend_overrides`` to
    ``RuntimeLocal`` — the runtime is the single source of truth for the bus
    (``feedback_bus_is_the_transport``).

    ``timeout`` is enforced twice (OMN-14397): cooperatively inside
    ``RuntimeLocal`` via ``asyncio.wait_for``, and again here as a hard
    ``SIGALRM`` backstop (``timeout`` plus a fixed grace window) that fires
    even if the inner call is stuck in non-cooperative blocking I/O. A
    backstop trip returns exit code 1 with a clear stderr message instead of
    hanging indefinitely.
    """
    resolved_task_type = task_type or classify_task_type(prompt)
    if bus is None:
        if kafka_bootstrap is not None:
            raise ValueError(
                "--kafka-bootstrap is only valid with --bus kafka "
                "(got --bus unset; the auto-resolved default never accepts "
                "an explicit bootstrap override — pass --bus kafka too)."
            )
        bus, reason = resolve_default_bus()
        if bus == "kafka":
            logger.info("onex delegate: auto-resolved event bus -> kafka (%s)", reason)
        else:
            # Covers BOTH "no broker configured" and "configured but not
            # healthy" (the OMN-14380 off-box-caller / stale-broker symptom)
            # — ``reason`` (from ``resolve_default_bus``/``probe_kafka``)
            # already distinguishes the two in text. Warn unconditionally
            # (mirrors ``service_kernel.py``'s own precedent of warning when
            # KAFKA_BOOTSTRAP_SERVERS is unset) rather than silently repeating
            # the OMN-14376 data-loss default. stderr / capture log only —
            # the receipt stream on stdout stays clean.
            logger.warning(
                "onex delegate: using inmemory event bus (%s) — this "
                "delegation's evidence will land in the local SQLite fallback, "
                "NOT the shared delegation_events projection",
                reason,
            )
    backend_overrides = build_backend_overrides(
        bus=bus, kafka_bootstrap=kafka_bootstrap
    )
    run_id = uuid.uuid4()
    # OMN-14397: minted fresh per invocation — never reused/cached across runs
    # sharing a working directory or state-root — and threaded explicitly into
    # the payload so it becomes the delegate request's correlation_id rather
    # than an implicit default decided downstream. Kept a UUID object here;
    # only stringified at the JSON payload boundary in _write_payload.
    correlation_id = uuid.uuid4()
    payload_path = _write_payload(
        prompt=prompt,
        task_type=resolved_task_type,
        max_tokens=max_tokens,
        state_root=state_root,
        run_id=run_id,
        correlation_id=correlation_id,
    )
    contract_path = _resolve_packaged_contract(DELEGATE_NODE_NAME)
    try:
        with _hard_timeout(timeout + _HARD_TIMEOUT_GRACE_SECONDS):
            return run_receipt_mode(
                node_name=DELEGATE_NODE_NAME,
                contract_path=contract_path,
                input_path=payload_path,
                state_root=state_root,
                backend_overrides=backend_overrides,
                timeout=timeout,
                verbose=verbose,
                emit_socket=emit_socket or default_emit_socket_path(),
            )
    except DelegateTimeoutExceededError as exc:
        click.echo(str(exc), err=True)
        return 1
