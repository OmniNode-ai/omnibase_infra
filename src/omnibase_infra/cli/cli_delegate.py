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

Bus target (OMN-13532; default flipped under OMN-14376; made deterministic
under OMN-16678; made CONFIG-RESOLVED under OMN-17304). ``--bus`` is OPTIONAL,
never required for a delegation to reach the shared platform substrate. When
omitted, :func:`resolve_default_bus` resolves the CLI's EMBEDDED runtime the
way every other runtime resolves — from that runtime's OWN configuration,
through the ONE shared authority
(``backends/auto_configure.py::resolve_bus_type``, also used by
``select_event_bus``), which applies a single resolution order —
**explicit ``--bus`` > configured authority > shipped ``inmemory`` default**:
the ``ONEX_CONTRACTS_DIR`` bootstrap pointer names a contracts directory whose
``runtime/runtime_config.yaml`` is the configured authority; with no pointer
(or no file) the SHIPPED tier-0 default runtime config answers — in-memory
bus, ``local`` profile — with a WARNING logged (stderr / capture, never
stdout) naming the local-SQLite consequence. ``ONEX_EVENT_BUS_TYPE`` holds NO
tier (set-and-ignored produces a warning), and a broker's reachability no
longer decides the transport: execution locus is a resolved property of
configuration, not an environmental accident.

Broker addressing (OMN-16871). ``--bus`` says WHICH KIND of transport;
``--lane`` says WHICH BROKER. The address used to be resolved by
``EventBusKafka`` from the ambient ``KAFKA_BOOTSTRAP_SERVERS``, which on the
launching Mac names the ``.201`` STABILITY-TEST lane — so every ad hoc
delegation from a developer shell published onto a governed proof lane, and
that lane recorded four such CLI terminal-run consumer groups when the finding
was re-verified. That path is DELETED, not repointed: with the resolved bus on
``kafka``, a run that named neither ``--lane`` nor ``--kafka-bootstrap`` is
refused, and the refusal lists the declared lanes. ``--lane`` resolves the
broker AND the declared transport from ``omnimarket/config/ci_bus_lanes.yaml``
under the workspace root — the same declaration the OCC publishers and the
chain canary already read, through the same ``load_lane_transport`` reader, so
no second source exists. SASL credentials are unchanged and still reach the
client through the standard ``KAFKA_SASL_*`` environment, which is where a
secret belongs; what left the environment is the ADDRESS.

**Execution locality (OMN-17295): ``--bus`` selects the TRANSPORT, never the
executor.** On both values the ``node_delegate_skill_orchestrator`` runs
IN-PROCESS in this CLI process, resolved from the local venv's installed
``omnimarket``. :func:`build_backend_overrides` is the only consumer of the
flag and it returns ``{"event_bus": <bus>}`` (plus an optional
``kafka_bootstrap``) — nothing in that map names an executor or a remote host.
With ``kafka``, the typed ``ModelDelegateSkillRequest`` command and this run's
terminal are carried over the live broker
(``feedback_bus_is_the_transport`` — the bus is THE transport) so the
projection lands in the shared ``delegation_events`` table rather than
per-machine SQLite; the accept/climb decision itself is still made by the
orchestrator code in THIS process. A ``--bus kafka`` invocation is therefore
NOT a probe of a deployed lane's behaviour, and there is no remote-execution
mode to select: building one is deliberately out of scope (a thin client, if
it is ever built, is gateway-mediated). ``--bus``/``--kafka-bootstrap`` are
explicit OVERRIDES for forcing a specific transport; both flow straight
through ``backend_overrides`` to ``RuntimeLocal``, which never hardcodes the
broker.

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
.. versionchanged:: OMN-17295
   The minted ``correlation_id`` is threaded into ``run_receipt_mode`` so the
   receipt is selected strictly by this run's identity; ``--bus`` help no
   longer claims a deployed runtime consumer dispatches the work.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import re
import signal
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.backends.auto_configure import (
    BUS_INMEMORY,
    BUS_KAFKA,
    SUPPORTED_BUS_TYPES,
    EventBusResolutionAmbiguousError,
    resolve_bus_type,
)
from omnibase_infra.cli.cli_node import _resolve_packaged_contract
from omnibase_infra.cli.delegate_lane import (
    DelegateLaneSelectionError,
    resolve_lane_target,
)
from omnibase_infra.cli.delegate_lane_credentials import (
    DelegateLaneCredentialError,
    resolve_lane_client_transport_for,
)
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusRefusedError,
    contract_terminal_topic,
    resolve_delegate_locus,
)
from omnibase_infra.cli.delegate_queue_depth import (
    observe_delegate_queue_depth,
)
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
    resolve_delegate_terminal,
)
from omnibase_infra.cli.model_delegate_locus_decision import (
    ModelDelegateLocusDecision,
)
from omnibase_infra.cli.model_delegate_queue_depth import (
    ModelDelegateQueueDepth,
)
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal
from omnibase_infra.cli.model_delegate_timeout_refusal import (
    ModelDelegateTimeoutRefusal,
)
from omnibase_infra.cli.model_delegate_transport_refusal import (
    ModelDelegateTransportRefusal,
)
from omnibase_infra.cli.omnimarket_drift_guard import (
    DRIFT_OVERRIDE_ENV,
    OmnimarketDriftError,
    check_omnimarket_drift,
)
from omnibase_infra.cli.protocol_drift_guard_verdict import (
    ProtocolDriftGuardVerdict,
)
from omnibase_infra.cli.receipt_mode import (
    default_emit_socket_path,
    run_receipt_mode,
)
from omnibase_infra.cli.task_class_selection import (
    DEFAULT_TASK_TYPE,
    EnumTaskTypeResolution,
    ModelSelectableTaskClass,
    ModelTaskClassExecutionBudget,
    ModelTaskTypeResolution,
    TaskClassContractError,
    load_selectable_task_classes,
    load_selection_fallback,
    resolve_task_class_contract_path,
    resolve_task_class_execution_budget,
    resolve_task_type,
)
from omnibase_infra.cli.workspace_reconcile import make_workspace_reconciler
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bind_lane_client_transport,
)
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy import (
    ModelKafkaConnectRetryPolicy,
)
from omnibase_infra.runtime_identity import collect_runtime_identity
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_DELEGATION_REQUEST
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

logger = logging.getLogger(__name__)

__all__ = [
    "DELEGATE_NODE_NAME",
    "DELEGATE_SOURCE",
    "DELEGATE_SOURCE_CHOICES",
    "DEFAULT_TASK_TYPE",
    "TASK_TYPE_CHOICES",
    "BUS_CHOICES",
    "DEFAULT_BUS",
    "LOCUS_CHOICES",
    "DelegateTimeoutExceededError",
    "build_backend_overrides",
    "classify_task_type",
    "load_supported_criteria",
    "resolve_default_bus",
    "resolve_task_class",
    "run_delegate",
]

# The omnimarket node that owns the consumer-facing delegation contract. It is
# registered under the ``onex.nodes`` entry-point group and resolvable from
# this CLI's environment (the delegate node ships its packaged contract.yaml).
DELEGATE_NODE_NAME = "node_delegate_skill_orchestrator"

# Default registered adapter source for ``ModelDelegateSkillRequest`` (Literal
# field) -- what the payload carries when ``--source`` is omitted. Preserves
# pre-OMN-15185 behavior for every existing caller.
DELEGATE_SOURCE = "claude-code"

# Closed choices for ``--source``, mirroring omnimarket's wire model
# (``ModelDelegateSkillRequest.source: Literal["claude-code", "codex",
# "external-client"]`` in
# ``omnimarket/models/delegation/wire/model_delegate_skill_request.py``).
#
# This CANNOT be derived by import: repo layering runs
# compat -> core -> spi -> infra, and separately omnimarket depends on
# omnibase-infra (declared in omnimarket's pyproject.toml) -- never the
# reverse. omnibase_infra importing omnimarket would be a circular/wrong-
# direction dependency, so this tuple is a manually-maintained duplicate of
# the wire Literal's args.
#
# DRIFT WARNING (OMN-15175's duplicate-alias failure class -- a hand-rolled
# ``_DelegateSource`` Literal in omnimarket silently fell out of sync with
# this exact wire model after it was widened): any future widening of
# ``ModelDelegateSkillRequest.source`` must be mirrored here by hand.
# ``tests/unit/cli/test_cli_delegate.py::TestSourceFlagDriftGuard`` asserts
# this tuple matches the live wire model's Literal args whenever omnimarket
# happens to be importable in the test env; when it is not (the normal
# omnibase_infra CI env, which has no omnimarket dependency), it instead
# asserts against the SAME documented value list stated here, so the test
# still fails the moment this comment and the tuple below disagree.
DELEGATE_SOURCE_CHOICES: tuple[str, ...] = ("claude-code", "codex", "external-client")

# The task classes the delegate contract exposes at the public Gateway --
# a hand-maintained MIRROR of the ``gateway_exposure: public`` projection of
# omnimarket's ``configs/task_class_contracts.v1.yaml``, used ONLY for the
# ``--task-type`` help text.
#
# It is not the authority and it never decides anything: an explicit
# ``--task-type`` is validated at run time against the contract itself
# (:func:`resolve_task_class`), which is what makes the CLI's selectable
# vocabulary EQUAL the contract's public set rather than merely resemble it.
# The mirror exists because repo layering forbids importing omnimarket from
# here and the ordinary omnibase_infra CI environment has no omnimarket
# installed, so there is nothing to read at import time -- the same constraint
# and the same treatment as ``DELEGATE_SOURCE_CHOICES`` above.
#
# DRIFT GUARD: ``tests/unit/cli/test_cli_delegate.py::TestTaskTypeVocabulary``
# asserts this mirror matches the stand-in contract used by infra CI. The live
# omnimarket contract has its own vocabulary pin in omnimarket's test suite;
# repo layering forbids importing that package here. Before OMN-18305 this
# tuple listed SEVEN classes against the contract's eleven, and
# ``summarization`` and ``planning`` -- the two classes an engineering standup
# actually belongs to -- were unreachable from the CLI by hand or by
# classifier.
TASK_TYPE_CHOICES = (
    "code_generation",
    "code_review",
    "complex_reasoning",
    "document",
    "planning",
    "reasoning",
    "refactor",
    "research",
    "review",
    "summarization",
    "test",
)

# Event-bus targets the CLI can select (OMN-13532). This is a TRANSPORT
# choice. It is no longer ALSO the execution-locality choice by accident:
# ``--locus`` owns that, and by default follows the transport, because a
# shared bus is the only kind another runtime can consume from
# (OMN-17295 / OMN-17304).
# These mirror ``RuntimeLocal.SUPPORTED_EVENT_BUS_VALUES`` — the runtime is the
# source of truth and rejects anything outside that set.
BUS_CHOICES = SUPPORTED_BUS_TYPES

# The absent-authority default (OMN-17304): the transport the SHIPPED tier-0
# default runtime config declares, and therefore what ``resolve_default_bus``
# returns when no configured authority answers. In-process, fully
# self-contained — the offline/standalone floor. This constant mirrors
# ``runtime/tier0_runtime_config.yaml``'s ``event_bus.type``; the tier-0
# golden tests pin the two together.
DEFAULT_BUS = BUS_INMEMORY

# Where the orchestrator makes the accept/climb decision. See
# ``EnumDelegateLocus``; the flag takes the enum's string values.
LOCUS_CHOICES: tuple[str, ...] = tuple(member.value for member in EnumDelegateLocus)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write one customer artifact atomically in its run directory."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


#: Receipt fields a summary-shaped receipt may carry this run's terminal in,
#: in the order they are tried. ``terminal_payload`` is the runtime's own
#: record; ``handler_result`` is the same object on the in-process path and
#: ``null`` on the dispatched one.
_TERMINAL_CARRIER_FIELDS: tuple[str, ...] = ("terminal_payload", "handler_result")


def _delegation_result(envelope: dict[str, object]) -> ModelDelegateTerminal | None:
    """Return this run's delegation terminal, or ``None`` if this is not a delegation.

    Two RECEIPT shapes reach here, because ``run_receipt_mode`` builds the typed
    ``ModelSkillResult[JsonValue]`` receipt ONLY when the run is success-like AND
    produced a handler result, and otherwise wraps it in a
    ``ModelReceiptRuntimeSummary``:

    * handler result present -> ``result`` IS the delegation terminal, and
      ``result_model`` names the delegate wire DTO.
    * anything else -> ``result`` is the summary and the terminal is carried in
      one of :data:`_TERMINAL_CARRIER_FIELDS`.

    Inside the summary there are then two CARRIER shapes, and that is the second
    fork OMN-18569 closes: in-process runs record the terminal bare, dispatched
    runs record the event envelope that delivered it, with the terminal under
    ``payload``. Both resolve through the one accessor
    :func:`~omnibase_infra.cli.delegate_terminal_resolver.resolve_delegate_terminal`;
    nothing here inspects a dict key to decide which shape it is holding.

    History, because each fix uncovered the next fork:

    * OMN-16999 recognised only the first RECEIPT shape, so a run that escalated
      past a failed rung -- which terminalizes ``failed`` even when a later
      attempt is ACCEPTED -- wrote nothing.
    * OMN-18569 recognised only the bare CARRIER shape, so every DISPATCHED run
      wrote nothing. Measured live 2026-09-17: correlation
      ``83aa8b6c-8189-49f0-953d-80c8f015ed0a`` exited 0 with a correct answer
      and produced no ``result.txt``, ``receipt.json`` or ``run.json``.

    Both times the writer returned quietly rather than failing, which is why
    both survived a release. So a receipt scoped to the delegate contract whose
    terminal will not resolve now RAISES. That is not a widening of the refusal:
    the scoping checks below still return ``None`` for a receipt that is not a
    delegation at all, because ``run_receipt_mode`` is shared with
    ``onex node``/``onex skill`` and a failed proof run of an unrelated node
    must be ignored, not raised on.

    Raises:
        DelegateTerminalUnresolvedError: the receipt IS a delegation and its
            terminal could not be resolved from any carrier field.
    """
    result = envelope.get("result")
    if not isinstance(result, dict):
        return None

    result_model = str(envelope.get("result_model") or "")
    if "ModelDelegateSkill" in result_model:
        return resolve_delegate_terminal(result)
    if "ModelReceiptRuntimeSummary" not in result_model:
        return None
    if DELEGATE_NODE_NAME not in str(result.get("workflow") or ""):
        return None

    refusals: list[str] = []
    for field in _TERMINAL_CARRIER_FIELDS:
        carrier = result.get(field)
        if carrier is None:
            refusals.append(f"{field}: absent")
            continue
        try:
            return resolve_delegate_terminal(carrier)
        except DelegateTerminalUnresolvedError as exc:
            refusals.append(f"{field}: {exc}")
    raise DelegateTerminalUnresolvedError(
        "delegate receipt carries no resolvable delegation terminal, so the "
        "customer artifacts cannot be written -- " + "; ".join(refusals)
    )


#: Mirrors the wire model's own pattern for a parameterised criterion slug.
#: The SLUG SET is read from the installed omnimarket (see
#: :func:`load_supported_criteria`); only this shape is spelled here, because a
#: regex cannot drift the way a copied list of names can.
_MAX_WORDS_PER_SENTENCE_RE = re.compile(r"^max_words_per_sentence_([1-9]\d*)$")


def _attempt_evidence(result: ModelDelegateTerminal) -> list[dict[str, object]]:
    """Return every rung this run attempted, in order, with its own verdict.

    The field set is :class:`~omnibase_infra.cli.model_delegate_attempt.ModelDelegateAttempt`'s own declared fields, so an
    attempt record cannot smuggle an unrelated key into the customer artifact
    (the model ignores extras), and a field the runtime stops emitting shows up
    as ``None`` instead of silently vanishing (every field is optional). Both
    properties used to be enforced by a hand-maintained tuple of field names
    beside the copy loop; they are now properties of the model itself.
    """
    return [attempt.model_dump(mode="json") for attempt in result.attempts]


def _unattributed_reason(result: ModelDelegateTerminal) -> str:
    """Say why no route was attributed, accurately for THIS run.

    OMN-18306 landed the receipt; this is its sentence. The reason used to be
    one constant asserting that "every rung this run attempted was refused",
    which states two things it never checked: that rungs were attempted at all,
    and — by naming none of them — which. Both halves were measured wrong on
    2026-09-15. One run reached two backends and the constant named neither;
    another was refused before dispatch in 265 ms with zero attempts, and the
    constant described refusals that never happened.

    Those are different failures with different fixes. "Four backends turned
    this down" is a routing or quality problem; "a guard refused this before it
    left the machine" is a request problem. A receipt that renders them
    identically cannot tell a customer which one they have.

    This NEVER names a route. The backends appear as evidence of what was
    tried; the route stays unattributed, because attributing output nobody
    accepted is the lie the refusal exists to prevent (AC3).
    """
    reached = [attempt.backend_id for attempt in result.attempts if attempt.backend_id]
    if not reached:
        return (
            "no backend was reached: this run was refused before any rung was "
            "dispatched, so there is no routing attempt to attribute. The "
            "failure reason recorded on this receipt is the guard that refused "
            "it, not a backend's verdict"
        )
    return (
        "no accepted routing attempt: this run reached "
        + ", ".join(reached)
        + " and every one of them refused, errored, or climbed, so no backend "
        "can be named as the author of this run's output. Each rung's own "
        "backend, tier and failure class is recorded under attempts"
    )


def _drift_guard_receipt_block(
    drift_guard: ProtocolDriftGuardVerdict | None,
) -> dict[str, object]:
    """The drift guard's verdict, as a receipt fragment.

    Present for either verdict the guard can hand back, and both are rendered
    through the same ``as_receipt_fields`` call so this function never learns
    which one it got:

    * the OFF-REGISTRY verdict (OMN-17255), on a machine with no canonical
      clone, where the stderr line is the only other place the fact appears
      and is gone the moment the terminal scrolls;
    * the ANCESTOR-LAG stamp (OMN-18814), on a registry machine whose
      installed omnimarket is a known ancestor of the clone head -- a run that
      proceeded while behind the tip, which a later reader has no other way to
      tell apart from a run that was AT the tip.

    ``None`` -- the guard's exact-match path -- still contributes no key at
    all, so a receipt from a converged machine stays byte-identical.
    """
    if drift_guard is None:
        return {}
    return {"drift_guard": drift_guard.as_receipt_fields()}


def _budget_outcome_receipt_block(result: ModelDelegateTerminal) -> dict[str, object]:
    """Copy only budget facts the terminal actually declared."""
    if result.budget_evidence is not None:
        return {"budget_evidence": result.budget_evidence.model_dump(mode="json")}
    if result.budget_refusal is not None:
        return {"budget_refusal": result.budget_refusal.model_dump(mode="json")}
    return {}


def _response_contract_receipt_block(
    result: ModelDelegateTerminal,
) -> dict[str, object]:
    """Copy response-contract evidence and extraction facts without inference."""
    block: dict[str, object] = {}
    if result.response_contract_evidence is not None:
        block["response_contract_evidence"] = (
            result.response_contract_evidence.model_dump(mode="json")
        )
    if result.preamble_chars is not None:
        block["preamble_chars"] = result.preamble_chars
    if result.output_refusal is not None:
        block["output_refusal"] = result.output_refusal.model_dump(mode="json")
    return block


def _require_completed_terminal_evidence(
    result: ModelDelegateTerminal,
    *,
    require_budget_evidence: bool,
    require_contract_evidence: bool,
) -> None:
    """Refuse a completed receipt that lacks evidence its request required."""
    if result.status != "completed":
        return
    missing: list[str] = []
    if require_budget_evidence and result.budget_evidence is None:
        missing.append("budget_evidence")
    if require_contract_evidence and result.response_contract_evidence is None:
        missing.append("response_contract_evidence")
    if missing:
        raise DelegateTerminalUnresolvedError(
            "completed delegation terminal omits required evidence: "
            + ", ".join(missing)
        )


def _receipt_evidence_requirements(
    *,
    response_contract: dict[str, object] | None,
) -> tuple[bool, bool]:
    """Return which completed-terminal evidence THIS request actually demanded.

    OMN-18956. The refusal below was armed with two literal ``True`` values,
    so every completed delegation was required to carry evidence no caller
    asked for and the deployed producer emits for neither. A run that
    answered, passed its quality gate at 1.0 and wrote its projection row
    still exited non-zero (dev lane, correlation
    ``0b21b5f0-fb56-4ab6-9219-24bbfd841f35``). The check was right; the
    question it was asked was not.

    * **Contract evidence is demanded by a response contract in the request.**
      A run that asked for no contract has no contract to evidence, and
      refusing it asserts a requirement nobody stated.
    * **Budget evidence is demanded by nothing a caller can pass, today.** The
      budget comparison happens only where the BACKEND declares
      ``max_grounded_input_tokens``, which is not knowable when this request
      is built, so there is no honest request-side predicate to return. It is
      returned as a value rather than dropped so the day a demand exists it
      has one place to land, and the refusal itself stays armed and proven.
    """
    return (False, response_contract is not None)


def _delegate_receipt_evidence_error(
    receipt: object,
    *,
    require_budget_evidence: bool = False,
    require_contract_evidence: bool = False,
) -> str | None:
    """Return the evidence defect that must turn a delegate receipt into failure."""
    receipt_dump = getattr(receipt, "model_dump", None)
    if not callable(receipt_dump):
        return "delegate receipt is not a serializable typed result"
    envelope = receipt_dump(mode="json")
    if not isinstance(envelope, dict):
        return "delegate receipt did not serialize to an object"
    result = _delegation_result(envelope)
    if result is None:
        return "delegate receipt carries no delegation terminal"
    try:
        _require_completed_terminal_evidence(
            result,
            require_budget_evidence=require_budget_evidence,
            require_contract_evidence=require_contract_evidence,
        )
    except DelegateTerminalUnresolvedError as exc:
        return str(exc)
    return None


def _write_unattributed_run_files(
    *,
    envelope: dict[str, object],
    result: ModelDelegateTerminal,
    state_root: Path,
    prompt: str,
    task_type: str,
    task_type_resolution: str,
    addressing: ModelDelegateRunAddressing,
    drift_guard: ProtocolDriftGuardVerdict | None = None,
) -> None:
    """Persist a terminally-failed delegation that attributed no route.

    Writes the same three files as the attributed path so a failed run is
    diagnosable at all -- ``result.txt`` (whatever content a rung managed to
    produce, usually empty), ``receipt.json``, ``run.json`` -- while stating
    outright that the route is unattributed. No route identity is written and
    none is inferred from the last attempted backend: that inference is the
    lie :func:`_write_local_run_files` exists to refuse.
    """
    run_id = str(envelope["run_id"])
    correlation_id = str(envelope["correlation_id"])
    run_dir = (state_root / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cost_usd = result.metrics.cost_usd if result.metrics is not None else None
    unattributed = _unattributed_reason(result)

    _atomic_write_text(run_dir / "result.txt", result.response)
    _atomic_write_text(
        run_dir / "receipt.json",
        json.dumps(
            {
                "receipt_id": correlation_id,
                "correlation_id": correlation_id,
                "run_id": run_id,
                "route_attributed": False,
                "route_unattributed": unattributed,
                "status": envelope.get("status"),
                "terminal_failure_cause": result.terminal_failure_cause,
                "failure_reason": result.error_message,
                "quality_gates_failed": list(result.quality_gates_failed),
                "quality_gate_passed": result.quality_gate_passed,
                "quality_score": result.quality_score,
                "cost_usd": cost_usd,
                "attempts": _attempt_evidence(result),
                "receipt": envelope,
                **_budget_outcome_receipt_block(result),
                **_response_contract_receipt_block(result),
                # OMN-18810: where a failed run RAN is the first question
                # asked about it, and route attribution being fail-closed is
                # exactly why it cannot be inferred from anything else here.
                **addressing.as_run_file_fields(),
                **_drift_guard_receipt_block(drift_guard),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    _atomic_write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "run_id": run_id,
                "correlation_id": correlation_id,
                # OMN-18810: no rung answered, so no tier is named -- the
                # same fail-closed attribution the receipt above applies.
                # This key used to be spelled ``lane``, which is now the
                # ``--lane`` value that ``addressing`` supplies.
                "routing_tier": None,
                "route_attributed": False,
                "prompt": prompt,
                "task_type": task_type,
                "task_type_resolution": task_type_resolution,
                **addressing.as_run_file_fields(),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    click.echo(
        "delegate artifacts (route UNATTRIBUTED -- "
        + unattributed
        + "): "
        + " ".join(
            str(run_dir / name) for name in ("result.txt", "receipt.json", "run.json")
        ),
        err=True,
    )


def _write_transport_refusal_run_files(
    *,
    refusal: ModelDelegateTransportRefusal,
    run_id: str,
    state_root: Path,
    prompt: str,
    task_type: str,
    task_type_resolution: str,
    addressing: ModelDelegateRunAddressing,
    envelope: dict[str, object] | None = None,
) -> None:
    """Persist the three files for a delegation that never reached the broker.

    OMN-18925 / C16. This is the same writer shape as the attributed and
    unattributed paths above -- same directory, same three filenames, same
    atomic write -- deliberately, rather than a second artifact format for a
    third kind of outcome. A caller that already knows how to read a failed
    delegation can read this one, and the invariant "there is always a
    ``receipt.json`` after a dispatch attempt" is worth more than a bespoke
    shape per failure class.

    Route attribution is fail-closed for the same reason as
    :func:`_write_unattributed_run_files`, and more strongly here: no rung
    ran, no backend was selected and no model was called, so there is nothing
    to attribute even wrongly. ``route_attributed`` is false and no backend,
    model, tier or endpoint key is written at all.

    ``terminal_failure_cause`` is written as ``None`` on purpose. All three
    members of the delegation failure enum are provider-side and a broker
    that never accepted the command says nothing about a provider -- see this
    module's transport refusal model for the full reasoning. The cause a
    reader needs is in ``transport_refusal``, typed.
    """
    run_dir = (state_root / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    correlation_id = str(refusal.correlation_id)
    refusal_fields = refusal.model_dump(mode="json")

    # Empty rather than absent: the customer-facing answer file always
    # exists, and a zero-byte one says "no answer was produced" in the same
    # place a real answer would have been.
    _atomic_write_text(run_dir / "result.txt", "")
    _atomic_write_text(
        run_dir / "receipt.json",
        json.dumps(
            {
                "receipt_id": correlation_id,
                "correlation_id": correlation_id,
                "run_id": run_id,
                "route_attributed": False,
                "route_unattributed": (
                    "transport failure: the command never reached the broker, "
                    f"so no rung ran ({refusal.reason})"
                ),
                "status": EnumSkillResultStatus.FAILED.value,
                "terminal_class": "transport",
                "terminal_failure_cause": None,
                "failure_reason": (
                    f"{refusal.transport_error_type}: {refusal.transport_error}"
                    if refusal.transport_error
                    else refusal.transport_error_type
                ),
                "transport_refusal": refusal_fields,
                "attempts": [],
                "receipt": envelope,
                **addressing.as_run_file_fields(),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    _atomic_write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "run_id": run_id,
                "correlation_id": correlation_id,
                "routing_tier": None,
                "route_attributed": False,
                "prompt": prompt,
                "task_type": task_type,
                "task_type_resolution": task_type_resolution,
                **addressing.as_run_file_fields(),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    click.echo(
        "delegate artifacts (TRANSPORT FAILURE -- "
        + refusal.reason
        + ", no rung ran): "
        + " ".join(
            str(run_dir / name) for name in ("result.txt", "receipt.json", "run.json")
        ),
        err=True,
    )


def _transport_refusal_from_receipt(
    *,
    envelope: dict[str, object],
    addressing: ModelDelegateRunAddressing,
    broker: str = "",
    command_topic: str = "",
) -> ModelDelegateTransportRefusal | None:
    """Build a transport refusal from a receipt, or ``None`` if not transport.

    The decision is read from the receipt's own typed
    ``runtime_error_is_transport`` flag, which ``run_receipt_mode`` set by
    ``isinstance`` against the infra transport error types at the moment it
    caught the exception. Nothing here re-derives the classification by
    matching a class name: a rename would silently stop matching, and the
    symptom would be the silence this whole change removes.
    """
    result = envelope.get("result")
    if not isinstance(result, dict):
        return None
    if not result.get("runtime_error_is_transport"):
        return None

    error_type = str(result.get("runtime_error_type") or "").strip()
    if not error_type:
        # The flag is only ever set beside a captured exception type, so an
        # empty one means a hand-built or truncated envelope rather than a
        # real transport failure. Refuse to write a refusal that cannot name
        # what failed; the ordinary unresolved-terminal path still reports it.
        return None

    attempts_permitted, bound_seconds = _resolve_transport_bound()
    duration_ms = envelope.get("duration_ms")

    return ModelDelegateTransportRefusal(
        reason="broker_unreachable",
        correlation_id=uuid.UUID(str(envelope["correlation_id"])),
        bus=addressing.bus,
        locus=addressing.locus.value,
        broker=broker,
        command_topic=command_topic,
        attempts_permitted=attempts_permitted,
        bound_seconds=bound_seconds,
        elapsed_seconds=(
            float(duration_ms) / 1000.0
            if isinstance(duration_ms, (int, float))
            else 0.0
        ),
        transport_error_type=error_type,
        transport_error=sanitize_error_string(str(result.get("error") or "")),
    )


def _resolve_transport_bound() -> tuple[int, float]:
    """Resolve the connect retry policy the delegate's own transport reads.

    Returns ``(attempts_permitted, bound_seconds)`` from the SAME declaration
    :class:`EventBusKafka` resolves at connect time, rather than from a
    literal duplicated here -- which is the drift AC-3 exists to prevent, and
    would put a number in the refusal that was never in force.
    """
    bus_config = ModelKafkaEventBusConfig.default()
    policy = ModelKafkaConnectRetryPolicy.from_bus_config(
        bus_config,
        attempt_timeout_seconds=float(bus_config.timeout_seconds),
    )
    return policy.total_attempts, policy.total_bound_seconds


def _write_local_run_files(
    *,
    receipt: object,
    state_root: Path,
    prompt: str,
    task_type: str,
    addressing: ModelDelegateRunAddressing,
    task_type_resolution: str | None = None,
    drift_guard: ProtocolDriftGuardVerdict | None = None,
    require_budget_evidence: bool = False,
    require_contract_evidence: bool = False,
    broker: str = "",
    command_topic: str = "",
) -> None:
    """Persist local delegation output and the accepted route evidence.

    Route identity is accepted only from the accepted attempt. Synthesizing a
    route from the last attempted backend would make a failed or escalated run
    look like a truthful answer.

    Three outcomes, and each is distinguishable from the other two (OMN-18569):

    * the receipt is a delegation with an accepted rung -> three attributed
      files, and a stderr line naming them;
    * the receipt is a delegation with no accepted rung -> three UNattributed
      files naming no route, and a stderr line saying so (OMN-18306);
    * the receipt is not a delegation at all -> nothing written, silently,
      because ``run_receipt_mode`` is shared and another node's proof run is
      not this writer's business.

    What is NOT an outcome any more is the fourth one: a delegation whose
    terminal cannot be resolved returning quietly. :func:`_delegation_result`
    raises instead, ``run_receipt_mode`` reports the failure on stderr and
    folds it into a non-zero exit, and the receipt still reaches stdout. A
    customer who gets no files now learns that from the process, not from
    listing a directory later.
    """
    if task_type_resolution is None:
        raise ValueError(
            "task_type_resolution is required; refusing to fabricate provenance"
        )
    receipt_dump = getattr(receipt, "model_dump", None)
    if not callable(receipt_dump):
        raise ValueError("delegate receipt is not a serializable typed result")
    envelope = receipt_dump(mode="json")
    if not isinstance(envelope, dict):
        raise ValueError("delegate receipt did not serialize to an object")

    # OMN-18925 / C16: a transport-class failure is written, not raised.
    #
    # This sits BEFORE _delegation_result deliberately. A run whose broker
    # was never reached has no terminal to resolve, so the resolver below
    # would raise DelegateTerminalUnresolvedError -- correctly, on its own
    # terms -- and run_receipt_mode would catch that, print "receipt callback
    # failed", and leave the caller with no run directory at all. That is the
    # measured 2026-09-21 failure: every lane is told to read the terminal
    # from receipt.json and on this path the file did not exist.
    #
    # The refusal is not a substitute for a terminal and invents nothing. It
    # records that the command never reached the broker, which transport
    # stage stopped it, and the deadline that was in force.
    transport_refusal = _transport_refusal_from_receipt(
        envelope=envelope,
        addressing=addressing,
        broker=broker,
        command_topic=command_topic,
    )
    if transport_refusal is not None:
        _write_transport_refusal_run_files(
            refusal=transport_refusal,
            run_id=str(envelope["run_id"]),
            state_root=state_root,
            prompt=prompt,
            task_type=task_type,
            task_type_resolution=task_type_resolution,
            addressing=addressing,
            envelope=envelope,
        )
        return

    # Scope this writer by the receipt's declared concrete type instead of
    # treating a generic fixture (or another node's result) as a malformed
    # delegation. A genuine delegation still fails closed below when route
    # evidence is absent.
    result = _delegation_result(envelope)
    if result is None:
        return
    _require_completed_terminal_evidence(
        result,
        require_budget_evidence=require_budget_evidence,
        require_contract_evidence=require_contract_evidence,
    )
    accepted = result.accepted_attempt
    if accepted is None:
        # OMN-18306: a run with no accepted attempt is still a run the customer
        # paid for and is owed an account of. Route attribution stays
        # fail-closed -- no backend, model, tier or endpoint is written, and
        # nothing is synthesised from the last attempted rung -- but the
        # terminal failure cause, the failure reason, every rung attempted, and
        # the cost incurred are written down, because the alternative (what
        # this raise used to do) was to tell the customer nothing at all.
        _write_unattributed_run_files(
            envelope=envelope,
            result=result,
            state_root=state_root,
            prompt=prompt,
            task_type=task_type,
            task_type_resolution=task_type_resolution,
            addressing=addressing,
            drift_guard=drift_guard,
        )
        return

    # Route identity comes from the fields the wire contract actually declares
    # -- ``ModelDelegateSkillAttemptRecord`` for the rung, ``provider`` for the
    # endpoint. The pre-OMN-18569 spellings this used to fall back through
    # (``model_used``, ``routing_decision_id``, ``tier_name``, ``endpoint_url``,
    # ``delegated_to``) are fields of OTHER delegation models -- the routing
    # decision and the projection row -- and never appear on a terminal, so
    # they could not fire. They are deleted rather than carried: a fallback
    # that cannot fire is indistinguishable, to the next reader, from one that
    # protects a live path.
    model = (accepted.model_id or result.model_name or "").strip()
    backend_id = (accepted.backend_id or "").strip()
    routing_tier = (accepted.tier or "").strip()
    endpoint = (result.provider or "").strip()
    if not all((model, backend_id, routing_tier, endpoint)):
        raise ValueError(
            "delegate receipt accepted attempt is missing backend, model, tier, "
            "or endpoint identity"
        )

    run_id = str(envelope["run_id"])
    correlation_id = str(envelope["correlation_id"])
    run_dir = (state_root / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(run_dir / "result.txt", result.response)
    _atomic_write_text(
        run_dir / "receipt.json",
        json.dumps(
            {
                "receipt_id": correlation_id,
                "correlation_id": correlation_id,
                "run_id": run_id,
                "backend_id": backend_id,
                "model": model,
                "endpoint": endpoint,
                "routing_tier": routing_tier,
                "status": envelope.get("status"),
                "receipt": envelope,
                **_budget_outcome_receipt_block(result),
                **_response_contract_receipt_block(result),
                # OMN-18810: the rung that answered is not the machine that
                # ran it. Both files carry the same four addressing keys so
                # neither can be read against the other.
                **addressing.as_run_file_fields(),
                **_drift_guard_receipt_block(drift_guard),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    _atomic_write_text(
        run_dir / "run.json",
        json.dumps(
            {
                "run_id": run_id,
                "correlation_id": correlation_id,
                # OMN-18810: this key was spelled ``lane`` and held the
                # accepted rung's TIER, so a run dispatched to the ``dev``
                # lane wrote ``lane: "local"`` and was read as a silent
                # local fallback. The tier now uses the spelling
                # ``receipt.json`` already gave it, and ``lane`` below is
                # the ``--lane`` value and nothing else.
                "routing_tier": routing_tier,
                "prompt": prompt,
                "task_type": task_type,
                "task_type_resolution": task_type_resolution,
                **addressing.as_run_file_fields(),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    click.echo(
        "delegate artifacts: "
        + " ".join(
            str(run_dir / name) for name in ("result.txt", "receipt.json", "run.json")
        ),
        err=True,
    )


def resolve_default_bus(*, kafka_bootstrap: str | None = None) -> tuple[str, str]:
    """Resolve the bus ``--bus`` defaults to when the flag is omitted (OMN-17304).

    The CLI hosts a runtime instance, and per the OMN-17304 operator ruling a
    runtime resolves its transport from its OWN configuration through the ONE
    shared authority — the same
    ``backends/auto_configure.py::resolve_bus_type`` seam the runtime kernel
    calls (``service_kernel.py::_resolve_event_bus_transport``). There is no
    CLI-specific config surface and no CLI-specific ladder:

    1. The embedded runtime's configuration is resolved by
       :func:`omnibase_infra.runtime.service_kernel.resolve_embedded_runtime_config`:
       the ``ONEX_CONTRACTS_DIR`` BOOTSTRAP pointer names a contracts
       directory whose ``runtime/runtime_config.yaml`` is the configured
       authority; with no pointer (or no file), the SHIPPED tier-0 default
       runtime config answers — in-memory bus, ``local`` profile. An
       unconfigured install is still config-resolved.
    2. ``config.event_bus.type`` from that configuration is passed as
       ``config_bus=`` — the tier the pre-ruling CLI skipped, which is what
       made ``~/.zshrc`` the transport authority.

    Because a config answer ALWAYS exists on this path, the broker-probe tier
    is structurally unreachable: a reachable (or unreachable) broker no longer
    decides the transport, and neither does ``ONEX_EVENT_BUS_TYPE`` — that env
    var holds NO tier any more (set-and-ignored produces a warning from the
    shared authority; see :func:`resolve_bus_type`). A configured ``kafka``
    with an unreachable broker now fails loudly downstream instead of
    silently degrading to a transport the config did not declare.

    This module still reads no environment itself (the ``check-env-reads`` CI
    gate blocks raw environment reads outside the approved config surfaces):
    the ``ONEX_CONTRACTS_DIR`` bootstrap read lives in ``service_kernel.py``
    and the set-and-ignored warning read stays in ``auto_configure.py`` — the
    files that already own those boundaries.

    Only called when ``--bus`` is NOT explicitly supplied — an explicit
    ``--bus`` (kafka or inmemory) is tier 1 of the shared order and is never
    second-guessed.

    ``authority_topic=SUFFIX_DELEGATION_REQUEST`` is still threaded through
    (OMN-16529) so the probe tier — should a future caller ever reach it with
    no config — keys AUTHORITATIVE on live consumer-group liveness for the
    exact delegation topic; on the current path it is inert by construction.

    Returns:
        ``(bus_type, reason)`` — the resolved transport and provenance naming
        WHICH authority answered (the config file path, or the shipped tier-0
        default), for the capture log and receipts.

    Raises:
        ProtocolConfigurationError: the resolved runtime config exists but is
            invalid — including a lane-profile config declaring the in-memory
            bus (the OMN-17304 profile axis).
    """
    from omnibase_infra.runtime.service_kernel import resolve_embedded_runtime_config

    config, config_source = resolve_embedded_runtime_config()
    return resolve_bus_type(
        config_bus=str(config.event_bus.type),
        config_source=config_source,
        kafka_bootstrap=kafka_bootstrap,
        authority_topic=SUFFIX_DELEGATION_REQUEST,
    )


@contextmanager
def _bind_lane_transport(
    lane_transport: ModelLaneClientTransport | None,
) -> Iterator[None]:
    """Make a resolved lane transport visible to every client this run builds.

    OMN-18432. ``RuntimeLocal`` constructs the bus itself and can be handed an
    address and nothing else, so a process that resolved a lane's declared
    protocol, its mechanism and its own identity has no argument to put them
    in. It states them for the scope of the dispatch instead, address-matched,
    and the bus factory and the pre-flight probe both read them back for that
    one broker.

    ``None`` -- an explicit ``--kafka-bootstrap``, or a surface whose
    environment already carries the credential -- binds nothing, and every
    client built inside this scope is assembled exactly as it is today.
    """
    if lane_transport is None:
        yield
        return
    logger.info(
        "onex delegate: lane %s transport bound for %s (%s/%s) as principal "
        "%s, credential resolved by reference from this machine's client "
        "store, declared in %s",
        lane_transport.lane,
        lane_transport.bootstrap_servers,
        lane_transport.security_protocol,
        lane_transport.sasl_mechanism or "no-sasl",
        lane_transport.sasl_username or "anonymous",
        lane_transport.declared_in,
    )
    with bind_lane_client_transport(lane_transport):
        yield


def build_backend_overrides(*, bus: str, kafka_bootstrap: str | None) -> dict[str, str]:
    """Build the ``backend_overrides`` map for ``run_receipt_mode``/``RuntimeLocal``.

    ``bus`` selects the event-bus backend (``inmemory`` or ``kafka``). For
    ``kafka``, ``kafka_bootstrap`` (``host:port``) is REQUIRED and routes
    through ``EventBusKafka.from_bootstrap`` so the live broker is targeted
    without process-wide environment mutation. Omitting it used to let the
    Kafka bus resolve its own bootstrap from ``KAFKA_BOOTSTRAP_SERVERS``;
    OMN-16871 removed that path, because on the launching host that variable
    names the governed stability-test lane and every ad hoc delegation from a
    developer shell therefore published onto a proof lane. There is now no
    combination of arguments that produces a kafka bus with an address this
    process did not resolve explicitly. ``kafka_bootstrap`` is only meaningful
    for ``bus="kafka"`` and is rejected otherwise so a typo (e.g. passing a
    broker with the default in-memory bus) fails loud rather than silently
    running in-process.
    """
    if bus not in BUS_CHOICES:
        raise ValueError(
            f"Unsupported bus {bus!r}. Choose one of: {', '.join(BUS_CHOICES)}."
        )
    if bus != "kafka" and kafka_bootstrap is not None:
        raise ValueError(
            f"--kafka-bootstrap is only valid with --bus kafka (got --bus {bus})."
        )
    if bus == "kafka" and kafka_bootstrap is None:
        raise ValueError(
            "the kafka bus requires a broker address: pass --lane <lane id> "
            "to resolve it from the lane declaration, or --kafka-bootstrap "
            "<host:port> to state a lane-internal address directly. It is no "
            "longer resolved from KAFKA_BOOTSTRAP_SERVERS (OMN-16871) -- on "
            "the launching host that variable names the governed "
            "stability-test lane, so an ambient value silently selected a "
            "proof lane."
        )
    overrides: dict[str, str] = {"event_bus": bus}
    if kafka_bootstrap is not None:
        overrides["kafka_bootstrap"] = kafka_bootstrap
    return overrides


def classify_task_type(
    prompt: str,
    *,
    classes: tuple[ModelSelectableTaskClass, ...] | None = None,
) -> str:
    """Resolve ``prompt`` to a task class declared by the task-class contract.

    OMN-18305 removed the hardcoded keyword table this used to be. Selection
    rules — word boundaries, presence-not-frequency, and shape gating — are
    declared per class in omnimarket's ``task_class_contracts.v1.yaml`` and
    evaluated by :mod:`omnibase_infra.cli.task_class_selection`; see that
    module for what the table got wrong and why.

    ``classes`` is for callers that already resolved the contract (the CLI
    resolves it once per run) and for tests pointing at a probe contract.
    """
    return resolve_task_class(prompt, explicit=None, classes=classes).task_type


def resolve_task_class(
    prompt: str,
    *,
    explicit: str | None,
    classes: tuple[ModelSelectableTaskClass, ...] | None = None,
) -> ModelTaskTypeResolution:
    """Resolve this run's task class and carry HOW it was resolved with it.

    An explicit ``--task-type`` is validated against the RESOLVED contract, never
    against ``TASK_TYPE_CHOICES`` alone (OMN-18342). ``TASK_TYPE_CHOICES`` is a
    documentation/help-text mirror pinned equal to the contract by
    ``TestTaskTypeVocabulary`` -- it must never decide a live path. When the
    contract cannot be resolved, this fails closed (propagates
    ``TaskClassContractError``) rather than falling back to the mirror.
    """
    if classes is not None:
        return resolve_task_type(prompt, explicit=explicit, classes=classes)
    contract_path = resolve_task_class_contract_path()
    return resolve_task_type(
        prompt,
        explicit=explicit,
        classes=load_selectable_task_classes(contract_path),
        # OMN-18305 residual: the fallback is a GRADING decision, so the
        # contract owns it. A contract that declares none yields the module
        # default, whose docstring records the two properties any fallback
        # has to satisfy.
        fallback=load_selection_fallback(contract_path),
    )


def load_supported_criteria() -> frozenset[str] | None:
    """Return the closed acceptance-criterion vocabulary, or ``None`` if unreadable.

    ``acceptance_criteria`` is NOT free text. The delegation wire model
    validates every entry against a closed slug set, plus the pattern
    ``max_words_per_sentence_<N>``, and refuses the whole request otherwise.
    Measured live 2026-09-15: three free-text criteria produced a 265 ms
    ``ValidationError`` with a pydantic traceback, zero rungs attempted, and no
    mention of which command-line flag the caller had got wrong.

    So the vocabulary is resolved HERE, at the flag, from the same installed
    omnimarket the contract itself is read from — one source, not a copy that
    can drift. ``None`` means omnimarket is unresolvable, in which case this
    command cannot dispatch at all for unrelated reasons and the criteria are
    passed through to be validated where they always were.
    """
    try:
        module = importlib.import_module(
            "omnimarket.models.delegation.wire.model_delegation_request"
        )
        supported = module.SUPPORTED_ACCEPTANCE_CRITERIA
    except (ImportError, AttributeError):
        return None
    return frozenset(str(item) for item in supported)


def _validate_criteria(criteria: tuple[str, ...]) -> tuple[str, ...]:
    """Refuse an unknown criterion here, naming the flag and the vocabulary."""
    if not criteria:
        return criteria
    supported = load_supported_criteria()
    if supported is None:
        return criteria
    unsupported = sorted(
        item
        for item in criteria
        if item not in supported and not _MAX_WORDS_PER_SENTENCE_RE.match(item)
    )
    if unsupported:
        raise ValueError(
            "--criteria takes declared criterion slugs, not free text. "
            f"Unsupported: {', '.join(repr(item) for item in unsupported)}. "
            f"Allowed: {', '.join(sorted(supported))}, or "
            "max_words_per_sentence_<N>."
        )
    return criteria


def _resolve_task_class_flag(
    task_type: str | None, task_class_alias: str | None
) -> str | None:
    """Collapse ``--task-type`` and its ``--task-class`` alias into one value.

    Passing both is a usage error rather than a precedence rule: a silent
    winner between two spellings of the same flag is how a caller ends up
    graded against a class it thought it had overridden.
    """
    if task_type is not None and task_class_alias is not None:
        raise ValueError(
            "--task-type and --task-class are two spellings of the same flag; "
            "pass one, not both"
        )
    return task_type if task_type is not None else task_class_alias


def _load_response_contract(raw: str | None) -> dict[str, object] | None:
    """Read ``--response-contract`` as inline JSON or as a path to a JSON file.

    Refuses anything that is not a JSON object: the field is threaded to the
    quality gate as a JSON-Schema-shaped contract, and a list or a bare scalar
    would be accepted here and rejected far downstream with no mention of this
    flag.
    """
    if raw is None:
        return None
    candidate = Path(raw)
    if candidate.suffix == ".json":
        if not candidate.is_file():
            raise ValueError(
                f"--response-contract names {raw!r}, which ends in .json but is "
                "not a readable file"
            )
        text = candidate.read_text(encoding="utf-8")
    else:
        text = raw
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--response-contract is neither a readable .json file nor valid "
            f"inline JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            "--response-contract must be a JSON object describing the expected "
            f"response structure, not a {type(parsed).__name__}"
        )
    return parsed


def _write_payload(
    *,
    prompt: str,
    task_type: str,
    source: str,
    max_tokens: int | None,
    state_root: Path,
    run_id: uuid.UUID,
    correlation_id: uuid.UUID,
    acceptance_criteria: tuple[str, ...] = (),
    quality_contract_mode: str | None = None,
    response_contract: dict[str, object] | None = None,
    system_prompt: str | None = None,
    requested_timeout_seconds: int | None = None,
) -> Path:
    """Write the delegation input payload to run_id-suffixed scratch.

    Scratch lives under ``<state-root>/tmp/`` (never ``/tmp`` —
    ``feedback_no_tmp_use_workspace``). The payload validates against the
    delegate node's input model (``ModelDelegateSkillRequest``); only the
    fields the consumer supplies are set.

    ``source`` (OMN-15185) is the caller-resolved registered adapter source —
    the caller (:func:`run_delegate`) has already applied the
    :data:`DELEGATE_SOURCE` default when ``--source`` was omitted, so this
    function never reads the module constant itself; it only writes whatever
    value it is handed.

    When ``max_tokens`` is ``None`` (no explicit ``--max-tokens`` override) the
    key is omitted from the payload entirely, so the delegate node resolves the
    response budget per-backend from its routing contract rather than from a
    CLI-side default.

    ``correlation_id`` (OMN-14397) is written explicitly rather than left for
    the request model's ``default_factory`` to decide implicitly: the CLI is
    the one place guaranteed to mint a fresh identity per invocation (a new
    OS process every time), so it owns this run's tracing identity end to end
    instead of delegating that responsibility downstream.

    THE FOUR CALLER-STATED FIELDS (OMN-18305 residual, measured 2026-09-15).
    ``ModelDelegateSkillRequest`` has carried ``acceptance_criteria``,
    ``quality_contract_mode``, ``response_contract`` and ``system_prompt``
    since OMN-15193/OMN-15482, and the quality-gate reducer already branches on
    ``replace_task_class`` — but this CLI wrote none of them, so the ONLY
    rubric a ``onex delegate`` caller could be graded against was the one the
    task class declares. That is the rubric that was wrong in both failures
    measured on 2026-09-15: a drafting prompt refused on ``planning``'s
    ``covers_dependencies`` and another refused on ``research``'s
    ``cites_sources``. Each field is omitted entirely when unset, so no
    existing caller changes shape.
    """
    tmp_dir = state_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_dir / f"delegate-input-{run_id}.json"
    payload: dict[str, object] = {
        "prompt": prompt,
        "task_type": task_type,
        "source": source,
        "correlation_id": str(correlation_id),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if acceptance_criteria:
        payload["acceptance_criteria"] = list(acceptance_criteria)
    if quality_contract_mode is not None:
        payload["quality_contract_mode"] = quality_contract_mode
    if response_contract is not None:
        payload["response_contract"] = response_contract
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if requested_timeout_seconds is not None:
        payload["requested_timeout_seconds"] = requested_timeout_seconds
    payload_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return payload_path


def _terminal_wait_seconds(
    *,
    requested_timeout_seconds: int | None,
    execution_budget: ModelTaskClassExecutionBudget,
) -> int:
    """Return the effective execution window plus its declared delivery margin."""
    execution_seconds = (
        execution_budget.task_class_timeout_ceiling_seconds
        if requested_timeout_seconds is None
        else min(
            requested_timeout_seconds,
            execution_budget.task_class_timeout_ceiling_seconds,
        )
    )
    return execution_seconds + execution_budget.terminal_delivery_margin_seconds


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


def _timeout_receipt(
    *,
    correlation_id: uuid.UUID,
    run_id: uuid.UUID,
    declared_timeout: int,
    elapsed_seconds: float,
    bus: str,
    locus_decision: ModelDelegateLocusDecision,
    contract_path: Path,
    queue_depth: ModelDelegateQueueDepth,
) -> ModelSkillResult[ModelDelegateTimeoutRefusal]:
    """Build the one typed result a timed-out delegation puts on stdout (OMN-17516).

    ``onex delegate`` documents exactly ONE ``ModelSkillResult`` on stdout. The
    hard-timeout backstop was the single path that did not honour it: exit 1, a
    prose line on stderr, and nothing at all on stdout. A caller parsing the
    documented contract could not distinguish that from a run still in flight,
    which is precisely the 2026-08-26 report this closes.

    The receipt is ``FAILED`` with ``exit_code=1``. It is a report of a
    delegation that did not terminalize, never a substitute for one: no
    synthesized answer, no widened bound, nothing rescued.

    Every value is one this invocation already resolved -- its own minted ids,
    the transport and locus it actually ran on (OMN-17295/OMN-17304), and the
    topics read from the contract it dispatched. Nothing is re-derived here, so
    the refusal cannot describe a run other than the one that produced it.
    """
    refusal = ModelDelegateTimeoutRefusal(
        correlation_id=correlation_id,
        declared_timeout_seconds=declared_timeout,
        grace_seconds=_HARD_TIMEOUT_GRACE_SECONDS,
        elapsed_seconds=elapsed_seconds,
        bus=bus,
        locus=locus_decision.locus.value,
        terminal_topic=contract_terminal_topic(contract_path),
        command_topic=locus_decision.command_topic,
        broker=locus_decision.broker,
        queue_depth=queue_depth,
    )
    return ModelSkillResult[ModelDelegateTimeoutRefusal](
        skill_name="delegate",
        node_name=DELEGATE_NODE_NAME,
        status=EnumSkillResultStatus.FAILED,
        correlation_id=correlation_id,
        run_id=run_id,
        exit_code=1,
        duration_ms=int(elapsed_seconds * 1000),
        result=refusal,
        result_model=(
            "omnibase_infra.cli.model_delegate_timeout_refusal."
            "ModelDelegateTimeoutRefusal"
        ),
        runtime_identity=collect_runtime_identity(config_source=str(contract_path)),
    )


@click.command("delegate")
@click.argument("prompt")
@click.option(
    "--task-type",
    "task_type",
    type=str,
    default=None,
    help=(
        "Task class for routing, validated at run time against the task-class "
        "contract's public projection ("
        + ", ".join(TASK_TYPE_CHOICES)
        + "). Omit to resolve it from the contract's declared selection "
        f"predicates; the fallback when none claims the prompt is "
        f"{DEFAULT_TASK_TYPE}. The chosen class and how it was chosen are "
        "printed on stderr and recorded in the run artifacts."
    ),
)
@click.option(
    "--task-class",
    "task_class_alias",
    type=str,
    default=None,
    help=(
        "Alias for --task-type. The contract calls these TASK CLASSES, so the "
        "flag that selects one may be spelled either way; passing both is a "
        "usage error rather than a silent precedence rule."
    ),
)
@click.option(
    "--criteria",
    "criteria",
    type=str,
    multiple=True,
    help=(
        "An acceptance criterion this answer must meet, repeatable. Stating "
        "your own criteria is how you stop being graded against a rubric you "
        "did not ask for: on 2026-09-15 a drafting prompt was refused on every "
        "rung for missing source citations, because the task class it landed "
        "on grades research. With --criteria-mode replace-task-class these "
        "criteria BECOME the bar; by default they are added to it."
    ),
)
@click.option(
    "--criteria-mode",
    "criteria_mode",
    type=click.Choice(["extend-task-class", "replace-task-class"]),
    default=None,
    help=(
        "Whether --criteria are added to the task class's own definition of "
        "done (the default) or REPLACE it. Only meaningful with --criteria. "
        "'replace-task-class' is the escape hatch from a shape floor that does "
        "not apply to your task."
    ),
)
@click.option(
    "--response-contract",
    "response_contract",
    type=str,
    default=None,
    help=(
        "A JSON-Schema-shaped contract describing the response structure you "
        "expect, as inline JSON or a path to a .json file. When set, the "
        "quality gate validates the response STRUCTURALLY against this schema "
        "instead of the task class's keyword heuristics."
    ),
)
@click.option(
    "--system-prompt",
    "system_prompt",
    type=str,
    default=None,
    help=(
        "A system message sent as a distinct role alongside the prompt, rather "
        "than concatenated into it. Omit to use the task class default."
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
    "--source",
    "source",
    type=click.Choice(DELEGATE_SOURCE_CHOICES),
    default=None,
    help=(
        "Registered adapter source stamped into the delegation request's "
        "'source' field (must match the wire model's ModelDelegateSkillRequest "
        f".source Literal). Omit to use the default: {DELEGATE_SOURCE!r} — "
        "unchanged pre-OMN-15185 behavior for every existing caller."
    ),
)
@click.option(
    "--bus",
    "bus",
    type=click.Choice(BUS_CHOICES),
    default=None,
    help=(
        "Event-bus backend — event TRANSPORT only. Where the work RUNS is "
        "--locus, which by default follows this: 'kafka' means a shared "
        "broker other runtimes consume from, so the deployed orchestrator "
        "can decide and this CLI publishes and waits; 'inmemory' exists only "
        "inside this process, so this CLI decides. Omit to resolve from the "
        "runtime's configured authority (OMN-17304): the runtime config named "
        "by the ONEX_CONTRACTS_DIR bootstrap pointer, else the shipped tier-0 "
        "default (inmemory). ONEX_EVENT_BUS_TYPE and broker reachability "
        "play no part. Pass explicitly to override the configured authority."
    ),
)
@click.option(
    "--locus",
    "locus",
    type=click.Choice(LOCUS_CHOICES),
    default=EnumDelegateLocus.AUTO.value,
    help=(
        "Where the delegate orchestrator makes the accept/climb decision "
        "(OMN-17295). 'auto' (default) follows the resolved transport: a "
        "shared bus dispatches to the deployed runtime consuming the command "
        "topic, an in-process bus runs here. 'deployed-lane' publishes the "
        "typed command and awaits this run's own correlated terminal, hosting "
        "nothing — it REFUSES if no live consumer group is bound to that "
        "topic, rather than quietly running here and reporting the result as "
        "the lane's. 'in-process' runs the orchestrator in this CLI from the "
        "local venv and says so in the record. The resolved locus and its "
        "evidence go into the run's capture log and receipt."
    ),
)
@click.option(
    "--lane",
    "lane",
    type=str,
    default=None,
    help=(
        "Lane this delegation is addressed to, e.g. 'dev'. Its broker and "
        "transport are read from the checked-in lane declaration "
        "omnimarket/config/ci_bus_lanes.yaml under the workspace root "
        "(--omni-home / $OMNI_HOME); the refusal lists the declared lanes. "
        "Required whenever the resolved bus is kafka, unless you state a "
        "lane-internal address with --kafka-bootstrap. The broker address is "
        "NOT read from KAFKA_BOOTSTRAP_SERVERS (OMN-16871): on the launching "
        "host that variable names the governed stability-test lane, so an "
        "ambient value silently selected a proof lane."
    ),
)
@click.option(
    "--kafka-bootstrap",
    "kafka_bootstrap",
    type=str,
    default=None,
    help=(
        "Broker address (host:port) stated directly, for a lane-internal "
        "address no declaration can carry -- e.g. 'redpanda:9092' from a "
        "container on the lane's own compose network. Prefer --lane, which "
        "resolves the address AND the transport from the lane declaration. "
        "Mutually exclusive with --lane; only valid with --bus kafka."
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
    type=click.IntRange(min=1),
    default=None,
    help="Requested execution time in seconds; omitted uses the task-class ceiling.",
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
@click.option(
    "--omni-home",
    type=click.Path(path_type=Path),
    envvar="OMNIBASE_PATH",
    default=None,
    help=(
        "Canonical OmniNode workspace root for the local omnimarket drift "
        "check (OMN-13930). Bound to $OMNIBASE_PATH, the product name for "
        "that root (OMN-16855/OMN-16852) -- the envvar binding is "
        "load-bearing: without it the guard silently receives "
        "omni_home=None and the canonical-clone check never fires, because "
        "callers never pass this flag explicitly."
    ),
)
@click.option(
    "--allow-omnimarket-drift",
    "allow_omnimarket_drift",
    is_flag=True,
    envvar=DRIFT_OVERRIDE_ENV,
    default=False,
    help=(
        "Dispatch even when the omnimarket co-install has drifted from the "
        "canonical clone (OMN-13930). Refusal is the DEFAULT; this is the "
        "only supported way past it, and it is named in the refusal message. "
        f"Bound to ${DRIFT_OVERRIDE_ENV}. Results produced under an override "
        "come from an UNVERIFIED build and are not evidence."
    ),
)
def delegate_command(
    prompt: str,
    task_type: str | None,
    task_class_alias: str | None,
    criteria: tuple[str, ...],
    criteria_mode: str | None,
    response_contract: str | None,
    system_prompt: str | None,
    max_tokens: int | None,
    source: str | None,
    bus: str | None,
    locus: str,
    lane: str | None,
    kafka_bootstrap: str | None,
    state_root: Path,
    timeout: int | None,
    verbose: bool,
    emit_socket: Path | None,
    omni_home: Path | None,
    allow_omnimarket_drift: bool,
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
        onex delegate "hand off from the external client" --source external-client
        # Dispatch to the deployed orchestrator; refuses if nothing consumes the topic.
        # --lane names the broker: since OMN-16871 a shared-bus run that names none
        # is refused rather than reading an ambient address, so the lane selector is
        # part of the command, not an optional extra.
        onex delegate "document the router" --bus kafka --lane dev --locus deployed-lane
        # Run it here on purpose, and say so in the record:
        onex delegate "document the router" --bus kafka --lane dev --locus in-process
    """
    try:
        exit_code = run_delegate(
            prompt=prompt,
            task_type=_resolve_task_class_flag(task_type, task_class_alias),
            acceptance_criteria=_validate_criteria(tuple(criteria)),
            criteria_mode=criteria_mode,
            response_contract=_load_response_contract(response_contract),
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            source=source,
            bus=bus,
            locus=EnumDelegateLocus(locus),
            lane=lane,
            kafka_bootstrap=kafka_bootstrap,
            state_root=state_root,
            timeout=timeout,
            verbose=verbose,
            emit_socket=emit_socket,
            omni_home=omni_home,
            allow_drift=allow_omnimarket_drift,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    sys.exit(exit_code)


def run_delegate(
    *,
    prompt: str,
    task_type: str | None,
    acceptance_criteria: tuple[str, ...] = (),
    criteria_mode: str | None = None,
    response_contract: dict[str, object] | None = None,
    system_prompt: str | None = None,
    max_tokens: int | None,
    source: str | None = None,
    bus: str | None = None,
    locus: EnumDelegateLocus = EnumDelegateLocus.AUTO,
    lane: str | None = None,
    kafka_bootstrap: str | None = None,
    state_root: Path,
    timeout: int | None,
    verbose: bool,
    emit_socket: Path | None,
    omni_home: Path | None = None,
    allow_drift: bool = False,
) -> int:
    """Build the payload, resolve the contract, and dispatch in receipt mode.

    Returns the process exit code. Payload construction, node dispatch, and
    result extraction are all internal — the caller supplies a prompt and
    receives one typed receipt on stdout.

    ``source`` (OMN-15185) is the registered adapter source stamped into the
    delegation request's ``source`` field. ``None`` (the CLI default) resolves
    to :data:`DELEGATE_SOURCE` (``"claude-code"``) — unchanged pre-OMN-15185
    behavior for every existing caller. An explicit value must be one of
    :data:`DELEGATE_SOURCE_CHOICES`; the CLI's ``click.Choice`` enforces this
    at the flag boundary, and this function does not re-validate it.

    ``bus`` selects the event-bus backend. ``None`` (the CLI default,
    OMN-17304) resolves via :func:`resolve_default_bus` from the embedded
    runtime's OWN configuration — the runtime config named by the
    ``ONEX_CONTRACTS_DIR`` bootstrap pointer, else the shipped tier-0 default
    (``inmemory``) — so a configured install reaches the shared platform
    substrate BY DEFAULT, with no ``--bus kafka`` flag required, and an
    unconfigured one stays fully offline. An explicit ``"inmemory"``
    or ``"kafka"`` is never second-guessed.

    ``lane`` names the broker for a kafka run and is resolved through the lane
    declaration (OMN-16871); ``kafka_bootstrap`` states an address directly,
    for a lane-internal endpoint no declaration can carry. Exactly one of them
    is required whenever the resolved bus is ``"kafka"`` — neither is a
    refusal, and so is both. It remains a usage error to supply
    ``kafka_bootstrap`` without also explicitly requesting ``--bus kafka`` (a
    bare ``--kafka-bootstrap`` is never silently absorbed into the
    auto-resolved default). The resolved address flows through
    ``backend_overrides`` to ``RuntimeLocal`` — the runtime is the single
    source of truth for the bus (``feedback_bus_is_the_transport``).

    ``timeout`` is enforced twice (OMN-14397): cooperatively inside
    ``RuntimeLocal`` via ``asyncio.wait_for``, and again here as a hard
    ``SIGALRM`` backstop (``timeout`` plus a fixed grace window) that fires
    even if the inner call is stuck in non-cooperative blocking I/O. A
    backstop trip returns exit code 1 with a clear stderr message instead of
    hanging indefinitely.
    """
    # OMN-13930: ``DELEGATE_NODE_NAME`` is an omnimarket-provided node, so
    # this surface carries the same stale/absent co-install exposure as
    # ``onex skill`` and ``onex node`` -- it was simply the one of the three
    # never wired to the guard, and a drifted venv surfaced here as a bare
    # contract-resolution failure with no pointer to the repair command.
    # Runs FIRST, before any bus probe or payload write, so a drifted venv
    # never produces a receipt that could be mistaken for evidence.
    try:
        # OMN-17255: the return value is the OFF-REGISTRY verdict, present only
        # on a machine with no canonical clone. It is carried into the receipt
        # so a customer can read after the fact which check this run got, and
        # is None on a registry machine -- where the commit comparison ran and
        # the receipt is unchanged.
        drift_guard_check = check_omnimarket_drift(
            omni_home=str(omni_home) if omni_home else None,
            allow_drift=allow_drift,
            # OMN-17190: heal in-flight instead of handing a human a command to
            # type. Bound here rather than defaulted inside the guard so the
            # guard stays a pure function for every non-CLI caller.
            reconcile=make_workspace_reconciler(str(omni_home) if omni_home else None),
        )
    except OmnimarketDriftError as exc:
        raise click.ClickException(str(exc)) from exc

    # OMN-18305: resolve the class from the CONTRACT, and say out loud which
    # class was chosen and how. A class chosen silently is how a prose task
    # ended up filed as `test`, with the prose quality checks disarmed and the
    # customer never told.
    try:
        task_class = resolve_task_class(prompt, explicit=task_type)
        execution_budget = resolve_task_class_execution_budget(
            resolve_task_class_contract_path(), task_type=task_class.task_type
        )
    except TaskClassContractError as exc:
        raise click.ClickException(str(exc)) from exc
    resolved_task_type = task_class.task_type
    click.echo(
        f"task class: {resolved_task_type} "
        f"({task_class.resolution.value} — {task_class.reason})",
        err=True,
    )
    resolved_source = source or DELEGATE_SOURCE
    if bus is None:
        if kafka_bootstrap is not None:
            raise ValueError(
                "--kafka-bootstrap is only valid with --bus kafka "
                "(got --bus unset; the auto-resolved default never accepts "
                "an explicit bootstrap override — pass --bus kafka too)."
            )
        try:
            bus, reason = resolve_default_bus()
        except EventBusResolutionAmbiguousError as exc:
            # OMN-16678: an indeterminate probe is a REFUSAL, not a fallback.
            # Surfaced as a ClickException so the caller gets the ambiguity and
            # both remedies on stderr with a non-zero exit, instead of a
            # traceback or a silently coin-flipped transport.
            raise click.ClickException(str(exc)) from exc
        if bus == "kafka":
            logger.info("onex delegate: auto-resolved event bus -> kafka (%s)", reason)
        else:
            # Covers BOTH "shipped tier-0 default answered (no configured
            # authority)" and "the configured authority itself declares
            # inmemory" — ``reason`` (from ``resolve_default_bus``) already
            # names which in text. Warn unconditionally (mirrors
            # ``service_kernel.py``'s own precedent of warning when
            # KAFKA_BOOTSTRAP_SERVERS is unset) rather than silently repeating
            # the OMN-14376 data-loss default. stderr / capture log only —
            # the receipt stream on stdout stays clean.
            logger.warning(
                "onex delegate: using inmemory event bus (%s) — this "
                "delegation's evidence will land in the local SQLite fallback, "
                "NOT the shared delegation_events projection",
                reason,
            )
    else:
        # OMN-17304: an explicit --bus is tier 1 of the shared resolution
        # authority (``auto_configure.resolve_bus_type``) — it short-circuits
        # every other tier, so NO resolution happens on this run at all. Say
        # that out loud. Until now every provenance line lived inside the
        # ``bus is None`` branch above, so an explicit flag produced no record
        # whatsoever and nothing downstream (capture file, receipt, an
        # operator reading stderr) could tell "the configured authority chose
        # kafka" apart from "a human typed --bus kafka and the authority was
        # never consulted". Those are different kinds of evidence, and a
        # lane-probe receipt that conflates them is the same class of
        # instrument defect as OMN-17295.
        logger.info(
            "onex delegate: explicit --bus %s OVERRIDES the configured "
            "transport authority — tier 1 short-circuits resolution, so no "
            "config surface, env override, or broker probe was consulted",
            bus,
        )
    # OMN-16871: the broker ADDRESS comes from the lane the caller selected,
    # read out of the checked-in lane declaration. It is never taken from
    # ``KAFKA_BOOTSTRAP_SERVERS`` -- on the launching host that variable names
    # the governed stability-test lane, so an ambient value silently addressed
    # a proof lane. A kafka bus with no lane and no explicit broker is a
    # REFUSAL here, not a fallback.
    try:
        lane_target = resolve_lane_target(
            bus=bus,
            lane=lane,
            kafka_bootstrap=kafka_bootstrap,
            omni_home=omni_home,
        )
    except DelegateLaneSelectionError as exc:
        raise click.ClickException(str(exc)) from exc

    if lane_target is not None:
        # Name the declaration that answered, not merely the address: a
        # receipt that records only ``host:port`` cannot distinguish a
        # declared lane from a value somebody typed.
        logger.info(
            "onex delegate: --lane %s resolves to broker %s over %s, declared in %s",
            lane_target.lane,
            lane_target.bootstrap_servers,
            lane_target.security_protocol,
            lane_target.declared_in,
        )
        resolved_bootstrap: str | None = lane_target.bootstrap_servers
    else:
        resolved_bootstrap = kafka_bootstrap
        if resolved_bootstrap is not None:
            logger.info(
                "onex delegate: explicit --kafka-bootstrap %s states the "
                "broker address directly; no lane declaration was consulted",
                resolved_bootstrap,
            )
    backend_overrides = build_backend_overrides(
        bus=bus, kafka_bootstrap=resolved_bootstrap
    )
    # OMN-18432: AS WHOM, now that OMN-16871 settled WHICH BROKER. The lane
    # declaration carries the protocol and the mechanism and the CLI has been
    # logging both and forwarding neither, so a client addressed at the
    # authenticated dev-lane listener was still assembled out of whatever the
    # shell exported. Resolve the whole transport once, here, and refuse now
    # if this machine holds no identity for a lane that declares one -- the
    # alternative surfaces minutes later as a handshake error naming nothing
    # an operator can act on.
    try:
        lane_transport = resolve_lane_client_transport_for(
            lane_target=lane_target,
            onex_home=Path.home() / ".onex",
            environ=os.environ,
        )
    except DelegateLaneCredentialError as exc:
        raise click.ClickException(str(exc)) from exc

    # The binding covers the PRE-FLIGHT PROBE as well as the publish. The
    # locus probe runs first and builds its own admin client; on a SASL lane a
    # probe that authenticated differently from the publish would refuse
    # before the publish was ever attempted, and would report the wrong cause.
    with _bind_lane_transport(lane_transport):
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
            source=resolved_source,
            acceptance_criteria=acceptance_criteria,
            # The wire enum spells these with underscores; the flag spells them
            # with dashes, as every other choice flag on this command does.
            quality_contract_mode=(
                None if criteria_mode is None else criteria_mode.replace("-", "_")
            ),
            response_contract=response_contract,
            system_prompt=system_prompt,
            requested_timeout_seconds=timeout,
            max_tokens=max_tokens,
            state_root=state_root,
            run_id=run_id,
            correlation_id=correlation_id,
        )
        contract_path = _resolve_packaged_contract(DELEGATE_NODE_NAME)
        # OMN-17295 / OMN-17304: decide WHERE the orchestrator runs, and — for a
        # dispatched run — prove a deployed one is actually consuming the command
        # topic BEFORE anything is published. A refusal here is the point: the
        # defect being closed is an invocation that silently ran in-process and
        # was then read as evidence about a lane it never reached, so this path
        # never degrades, it stops.
        locus_probe_started = time.monotonic()
        try:
            locus_decision = resolve_delegate_locus(
                requested=locus,
                bus=bus,
                # OMN-16871: the RESOLVED address, not the raw flag. The
                # deployed-lane probe asks whether a live consumer group is bound
                # to the command topic; asking that of one broker and then
                # publishing to another is a probe of a lane the run never
                # reaches, which is the OMN-17295 instrument defect in a second
                # place.
                kafka_bootstrap=resolved_bootstrap,
                contract_path=contract_path,
                shared_bus_value=BUS_KAFKA,
            )
        except DelegateLocusRefusedError as exc:
            # OMN-18925 / C16: the SECOND transport exit, and the one a
            # genuinely unreachable broker takes. The probe refuses here
            # before anything is published, which is correct and stays --
            # what was wrong is that it left no artifact, so the negative
            # control ("broker is down") and the positive one ("broker
            # stalled mid-dispatch") produced different evidence shapes for
            # the same class of problem, and the cheaper of the two produced
            # none at all.
            #
            # The refusal is written against the CLI's own minted run_id
            # because no receipt exists yet: nothing has run. Its reason is
            # locus_probe_refused rather than broker_unreachable, because a
            # probe that finds no live consumer group may be reporting a lane
            # that is simply not running rather than a sick broker, and those
            # two send a reader to different places.
            attempts_permitted, bound_seconds = _resolve_transport_bound()
            _write_transport_refusal_run_files(
                refusal=ModelDelegateTransportRefusal(
                    reason="locus_probe_refused",
                    correlation_id=correlation_id,
                    bus=bus,
                    locus=(locus or EnumDelegateLocus.DEPLOYED_LANE.value),
                    broker=resolved_bootstrap or "",
                    attempts_permitted=attempts_permitted,
                    bound_seconds=bound_seconds,
                    elapsed_seconds=time.monotonic() - locus_probe_started,
                    transport_error_type=type(exc).__name__,
                    transport_error=sanitize_error_string(str(exc)),
                ),
                run_id=str(run_id),
                state_root=state_root,
                prompt=prompt,
                task_type=resolved_task_type,
                task_type_resolution=task_class.resolution.value,
                addressing=ModelDelegateRunAddressing(
                    locus=EnumDelegateLocus.DEPLOYED_LANE,
                    bus=bus,
                    lane=lane_target.lane if lane_target is not None else None,
                    dispatch_target=None,
                ),
            )
            raise click.ClickException(str(exc)) from exc

        # OMN-18810: the four addressing facts the two written files record,
        # built from the decision that was just PROVEN viable rather than
        # from the raw flags. ``--lane dev`` that resolved to no broker never
        # reaches here (``resolve_delegate_locus`` refuses first), so a file
        # can never name a lane the run did not actually address.
        addressing = ModelDelegateRunAddressing(
            locus=locus_decision.locus,
            bus=bus,
            lane=lane_target.lane if lane_target is not None else None,
            dispatch_target=(
                f"{locus_decision.command_topic} via {locus_decision.broker}"
                if locus_decision.locus is EnumDelegateLocus.DEPLOYED_LANE
                else None
            ),
        )

        # OMN-18956: derive ONCE, before the receipt layer is wired, so the
        # request and the requirement cannot drift apart between them.
        receipt_evidence_demanded = _receipt_evidence_requirements(
            response_contract=response_contract,
        )
        try:
            # OMN-17516: the refusal reports the wall time actually served,
            # which routinely exceeds declared + grace because a SIGALRM
            # raised inside a blocking call only propagates at the next
            # bytecode boundary. Measured, never assumed from the bound.
            dispatch_started = time.monotonic()
            terminal_wait_seconds = _terminal_wait_seconds(
                requested_timeout_seconds=timeout,
                execution_budget=execution_budget,
            )
            with _hard_timeout(terminal_wait_seconds + _HARD_TIMEOUT_GRACE_SECONDS):
                return run_receipt_mode(
                    node_name=DELEGATE_NODE_NAME,
                    contract_path=contract_path,
                    input_path=payload_path,
                    state_root=state_root,
                    backend_overrides=backend_overrides,
                    timeout=terminal_wait_seconds,
                    verbose=verbose,
                    emit_socket=emit_socket or default_emit_socket_path(),
                    # OMN-17304: a dispatched run hosts NOTHING. Without this the
                    # CLI subscribes the entry handler to the command topic it is
                    # publishing to and executes a backlog command out of its own
                    # venv, while the lane executes the real one — two runs, and
                    # the receipt was the wrong one's.
                    host_handlers=locus_decision.locus is EnumDelegateLocus.IN_PROCESS,
                    locus_decision=locus_decision,
                    # OMN-17295 / OMN-14872: the receipt layer cannot select by an
                    # identity it was never told. Handing it the id this CLI just
                    # minted is what lets it refuse another run's terminal
                    # envelope instead of printing it as ours.
                    expected_correlation_id=correlation_id,
                    # OMN-18956: ask only for the evidence THIS request
                    # demanded. Bound here because this is the only place the
                    # request and the receipt are both in scope.
                    receipt_validator=functools.partial(
                        _delegate_receipt_evidence_error,
                        require_budget_evidence=receipt_evidence_demanded[0],
                        require_contract_evidence=receipt_evidence_demanded[1],
                    ),
                    receipt_callback=lambda receipt: _write_local_run_files(
                        receipt=receipt,
                        state_root=state_root,
                        prompt=prompt,
                        task_type=resolved_task_type,
                        task_type_resolution=task_class.resolution.value,
                        addressing=addressing,
                        drift_guard=drift_guard_check,
                        # OMN-18956 residual: this is the SECOND site that
                        # arms the same refusal, and the first fix moved only
                        # the validator. The writer runs inside the receipt
                        # CALLBACK, so with literals here a run still exited
                        # non-zero after the validator had accepted it --
                        # which is why proving the leaf was not proof of the
                        # entry. Both now read one derivation.
                        require_budget_evidence=receipt_evidence_demanded[0],
                        require_contract_evidence=receipt_evidence_demanded[1],
                        # OMN-18925: the two facts a transport refusal needs
                        # that addressing does not carry separately. Taken
                        # from the decision that was PROVEN viable, so a
                        # refusal names the broker the run actually addressed
                        # rather than the flag as typed.
                        broker=locus_decision.broker,
                        command_topic=locus_decision.command_topic,
                    ),
                )
        except DelegateTimeoutExceededError as exc:
            # OMN-17516. The human-facing line stays (OMN-14397 added it and a
            # test pins it), and the caller that parses stdout now gets the one
            # typed result this command has always documented. Before this, the
            # backstop returned 1 with EMPTY stdout, which is byte-identical to
            # a run still in progress -- the whole of the 2026-08-26 "no result
            # at all" report. Nothing here invents a terminal: the receipt says
            # FAILED, and says what was awaited and for how long.
            # OMN-18852: a bare timeout does not tell the caller whether
            # the run was slow or merely behind, and on the .201 dev lane on
            # 2026-09-19 it was ALWAYS the second -- queue wait grew 3s ->
            # 445s across nine delegations while a control run's own
            # inference took 1.559s. The depth is OBSERVED from the broker
            # here, never supplied or assumed, and an unresolvable one says
            # so rather than printing a zero that would read as an idle queue.
            queue_depth = observe_delegate_queue_depth(
                bus=bus,
                broker=locus_decision.broker,
                command_topic=locus_decision.command_topic,
                consumer_groups=locus_decision.lane_consumer_groups,
            )
            click.echo(f"{exc} Queue: {queue_depth.describe()}.", err=True)
            click.echo(
                _timeout_receipt(
                    correlation_id=correlation_id,
                    run_id=run_id,
                    declared_timeout=terminal_wait_seconds,
                    elapsed_seconds=time.monotonic() - dispatch_started,
                    bus=bus,
                    locus_decision=locus_decision,
                    contract_path=contract_path,
                    queue_depth=queue_depth,
                ).model_dump_json()
            )
            return 1
