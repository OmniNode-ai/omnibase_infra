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

Bus target (OMN-13532). By default the CLI runs the orchestrator in-process on
``EventBusInmemory`` — fully self-contained, no broker required. When a live
delegation must reach a running runtime, ``--bus kafka --kafka-bootstrap
host:port`` selects the Kafka event bus so the typed
``ModelDelegateSkillRequest`` command is published to
``onex.cmd.omnimarket.delegate-skill.v1`` on the named broker, where the
deployed ``node_delegate_skill_orchestrator`` consumer picks it up
(``feedback_bus_is_the_transport`` — the bus is THE transport). The bus
selection and bootstrap override flow straight through ``backend_overrides`` to
``RuntimeLocal``; the CLI never hardcodes the broker.

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

.. versionadded:: OMN-13096
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import click

from omnibase_infra.cli.cli_node import _resolve_packaged_contract
from omnibase_infra.cli.receipt_mode import (
    default_emit_socket_path,
    run_receipt_mode,
)

__all__ = [
    "DELEGATE_NODE_NAME",
    "DELEGATE_SOURCE",
    "DEFAULT_TASK_TYPE",
    "TASK_TYPE_CHOICES",
    "BUS_CHOICES",
    "DEFAULT_BUS",
    "build_backend_overrides",
    "classify_task_type",
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

# Default bus when ``--bus`` is omitted: in-process, fully self-contained.
DEFAULT_BUS = "inmemory"


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
    """
    tmp_dir = state_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_dir / f"delegate-input-{run_id}.json"
    payload: dict[str, object] = {
        "prompt": prompt,
        "task_type": task_type,
        "source": DELEGATE_SOURCE,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    payload_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return payload_path


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
    default=DEFAULT_BUS,
    show_default=True,
    help=(
        "Event-bus backend. 'inmemory' runs the orchestrator in-process (no "
        "broker). 'kafka' publishes the typed delegate-skill command to the "
        "broker so a deployed runtime consumer dispatches it (live path)."
    ),
)
@click.option(
    "--kafka-bootstrap",
    "kafka_bootstrap",
    type=str,
    default=None,
    help=(
        "Kafka bootstrap servers (host:port) for --bus kafka. Omit to resolve "
        "from KAFKA_BOOTSTRAP_SERVERS. The dev lane on .201 advertises "
        "localhost:19092 (probe from .201). Only valid with --bus kafka."
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
    bus: str,
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
        # Publish to the live dev broker so a deployed runtime dispatches it:
        onex delegate "document the router" --bus kafka --kafka-bootstrap localhost:19092
    """
    sys.exit(
        run_delegate(
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
    )


def run_delegate(
    *,
    prompt: str,
    task_type: str | None,
    max_tokens: int | None,
    bus: str = DEFAULT_BUS,
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

    ``bus`` selects the event-bus backend (``inmemory`` default, ``kafka`` for
    the live broker path); ``kafka_bootstrap`` optionally overrides the broker
    when ``bus="kafka"``. Both flow through ``backend_overrides`` to
    ``RuntimeLocal`` — the runtime is the single source of truth for the bus
    (``feedback_bus_is_the_transport``).
    """
    resolved_task_type = task_type or classify_task_type(prompt)
    backend_overrides = build_backend_overrides(
        bus=bus, kafka_bootstrap=kafka_bootstrap
    )
    run_id = uuid.uuid4()
    payload_path = _write_payload(
        prompt=prompt,
        task_type=resolved_task_type,
        max_tokens=max_tokens,
        state_root=state_root,
        run_id=run_id,
    )
    contract_path = _resolve_packaged_contract(DELEGATE_NODE_NAME)
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
