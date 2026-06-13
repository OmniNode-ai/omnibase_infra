# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Receipt-mode execution for ``onex node`` / ``onex run`` (OMN-13094).

Layer A of the skill-output-suppression slice
(``docs/plans/2026-06-12-skill-output-suppression-plan.md``, Phase 2 item 1):

- ALL runtime logging routes to a run_id-suffixed capture file under the
  node's state root — console handlers are never installed, which kills the
  25-50-line RuntimeLocal INFO stream at the source.
- After the run, the full capture log and the full handler result are written
  to the content-addressed artifact store (omnibase_core, OMN-13093),
  ``artifact.captured`` + ``tool.output.captured`` events are emitted via the
  emit daemon socket, and stdout receives exactly ONE typed
  :class:`~omnibase_core.models.dispatch.model_skill_result.ModelSkillResult`
  JSON object carrying the FULL result.

Failure asymmetry (capture vs telemetry):

- Artifact write failure ⇒ the FULL output is printed instead of the receipt
  (no hidden loss — parent invariant 1; no silent fallback).
- Artifact write success but event emission failure ⇒ the receipt still
  prints; the event is spooled to a local outbox under the state root for
  later replay. A transient emit-daemon outage never re-floods the caller
  when the artifact exists.

.. versionadded:: OMN-13094
"""

from __future__ import annotations

import json
import logging
import socket
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import click
from pydantic import JsonValue, ValidationError

from omnibase_core.artifacts.artifact_store import ArtifactStore
from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
from omnibase_core.models.artifacts.model_artifact_ref import ModelArtifactRef
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli.model_receipt_runtime_summary import ModelReceiptRuntimeSummary
from omnibase_infra.runtime.runtime_local import RuntimeLocal

__all__ = [
    "CAPTURE_DIR_NAME",
    "SPOOL_DIR_NAME",
    "default_emit_socket_path",
    "run_receipt_mode",
]

logger = logging.getLogger(__name__)

# Capture logs and the emission spool live under the node's state root
# (matching the workflow_result.json convention — plan Open Question 4).
CAPTURE_DIR_NAME = "captures"
SPOOL_DIR_NAME = "emit_spool"

# Socket timeout for emit-daemon calls. Emission is telemetry, never the
# critical path — a slow daemon must not stall the dispatch.
_EMIT_TIMEOUT_SECONDS = 2.0

_MAX_EMIT_RESPONSE_BYTES = 65536

_WORKFLOW_TO_STATUS: dict[EnumWorkflowResult, EnumSkillResultStatus] = {
    EnumWorkflowResult.COMPLETED: EnumSkillResultStatus.SUCCESS,
    EnumWorkflowResult.PARTIAL: EnumSkillResultStatus.PARTIAL,
    EnumWorkflowResult.FAILED: EnumSkillResultStatus.FAILED,
    EnumWorkflowResult.TIMEOUT: EnumSkillResultStatus.FAILED,
}


def default_emit_socket_path() -> Path:
    """Deterministic address of the live emit daemon's Unix domain socket.

    ``~/.claude/emit.sock`` — the same default as omniclaude's emit client
    wrapper. This is the daemon's published address; nothing is ever written
    there by this module (emission failures spool under the node's state
    root instead). Override per-invocation via ``onex node --emit-socket``.
    """
    return Path.home() / ".claude" / "emit.sock"


def _emit_via_socket(
    socket_path: Path, event_type: str, payload: dict[str, JsonValue]
) -> None:
    """Send one event to the emit daemon (newline-delimited JSON protocol).

    Raises on any failure — the caller decides whether to spool.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(_EMIT_TIMEOUT_SECONDS)
    try:
        sock.connect(str(socket_path))
        request = {"event_type": event_type, "payload": payload}
        sock.sendall(json.dumps(request).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_EMIT_RESPONSE_BYTES:
                raise ConnectionError(
                    f"emit daemon response exceeded {_MAX_EMIT_RESPONSE_BYTES} bytes"
                )
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        raw = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8").strip()
        if not raw:
            raise ConnectionError("emit daemon closed connection without responding")
        response: object = json.loads(raw)
        if not isinstance(response, dict) or response.get("status") != "queued":
            raise RuntimeError(f"emit daemon rejected {event_type}: {response!r}")
    finally:
        sock.close()


def _emit_or_spool(
    event_type: str,
    payload: dict[str, JsonValue],
    *,
    spool_dir: Path,
    spool_stem: str,
    socket_path: Path,
) -> None:
    """Emit ``event_type`` via the daemon socket; spool locally on failure.

    Telemetry asymmetry (plan Phase 2 item 1): once the artifact exists, an
    emission failure must never re-flood the caller — the event is recorded
    to a local outbox file for later replay and the receipt still prints.
    """
    try:
        _emit_via_socket(socket_path, event_type, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        # OSError covers socket/connection/timeout failures; ValueError covers
        # malformed daemon JSON; RuntimeError covers daemon rejections.
        # Spool-on-failure is the contract here — never raise past this point.
        spool_dir.mkdir(parents=True, exist_ok=True)
        spool_path = spool_dir / f"{event_type.replace('.', '-')}-{spool_stem}.json"
        spool_record = {
            "event_type": event_type,
            "payload": payload,
            "spooled_at_utc": datetime.now(UTC).isoformat(),
            "spool_reason": f"{type(exc).__name__}: {exc}",
        }
        spool_path.write_text(json.dumps(spool_record, indent=2), encoding="utf-8")
        logger.warning(
            "receipt_mode: emit of %s failed (%s); spooled to %s",
            event_type,
            type(exc).__name__,
            spool_path,
        )


def _configure_capture_logging(capture_path: Path, *, verbose: bool) -> None:
    """Route ALL logging to ``capture_path``; install no console handlers.

    ``force=True`` removes any pre-existing root handlers so the runtime
    INFO stream can never leak to stdout/stderr in receipt mode (F7).
    """
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(capture_path, encoding="utf-8")],
        force=True,
    )


def _close_capture_logging() -> None:
    """Flush and close all root handlers, then install a ``NullHandler``.

    The ``NullHandler`` prevents Python's ``lastResort`` handler from leaking
    post-capture log records (e.g. spool warnings) to stderr — receipt mode
    allows nothing but the receipt on stdout/stderr. Spool records carry
    their own ``spool_reason``, so the dropped log line loses nothing.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.flush()
        handler.close()
        root.removeHandler(handler)
    root.addHandler(logging.NullHandler())


def _fully_qualified_name(obj: object) -> str:
    """Return ``module.QualName`` for the concrete type of ``obj``."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _extract_correlation_id(workflow_data: dict[str, JsonValue]) -> uuid.UUID:
    """Extract the run's correlation_id from workflow state, else mint one."""
    for key in ("handler_result", "terminal_payload"):
        candidate = workflow_data.get(key)
        if isinstance(candidate, dict):
            raw = candidate.get("correlation_id")
            if isinstance(raw, str):
                try:
                    return uuid.UUID(raw)
                except ValueError:
                    continue
    return uuid.uuid4()


def _load_workflow_data(state_root: Path) -> dict[str, JsonValue]:
    """Read ``workflow_result.json`` if the runtime wrote one."""
    result_path = state_root / "workflow_result.json"
    if not result_path.exists():
        return {}
    parsed: object = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        return {}
    return cast("dict[str, JsonValue]", parsed)


def _print_full_output_on_capture_failure(
    *,
    reason: str,
    capture_text: str,
    workflow_data: dict[str, JsonValue],
    runtime_error: str,
) -> None:
    """No-hidden-loss path: artifact write failed, so print EVERYTHING.

    Parent invariant 1 / no-silent-fallback: when durable capture is not
    possible, suppression is not allowed — the full output passes through.
    """
    click.echo(
        f"receipt mode: artifact capture failed ({reason}); "
        "printing full output instead of a receipt (no hidden loss).",
        err=True,
    )
    if capture_text:
        click.echo(capture_text, nl=False)
    if runtime_error:
        click.echo(runtime_error, nl=False)
    if workflow_data:
        click.echo(json.dumps(workflow_data, indent=2))


def run_receipt_mode(
    *,
    node_name: str,
    contract_path: Path,
    input_path: Path | None,
    state_root: Path,
    backend_overrides: dict[str, str],
    timeout: int,
    verbose: bool,
    emit_socket: Path,
) -> int:
    """Execute the node and print exactly one ``ModelSkillResult`` JSON.

    Returns the process exit code (the runtime's exit code; 1 when the
    runtime raised before producing a workflow result).
    """
    run_id = uuid.uuid4()
    capture_path = state_root / CAPTURE_DIR_NAME / f"{node_name}-{run_id}.log"
    spool_dir = state_root / SPOOL_DIR_NAME
    _configure_capture_logging(capture_path, verbose=verbose)

    started = time.monotonic()
    runtime_error = ""
    workflow_result: EnumWorkflowResult | None = None
    exit_code = 1
    handler_result_obj: object | None = None
    try:
        runtime = RuntimeLocal(
            workflow_path=contract_path,
            state_root=state_root,
            backend_overrides=backend_overrides,
            input_path=input_path,
            timeout=timeout,
        )
        workflow_result = runtime.run()
        exit_code = runtime.exit_code
        handler_result_obj = runtime.handler_result
    except Exception:
        # RuntimeLocal.run() records execution failures itself; reaching here
        # means contract load / bus construction raised. Errors are never
        # hidden: the traceback goes to the capture AND inline in the result.
        runtime_error = traceback.format_exc()
        logger.exception("receipt_mode: runtime raised")
    duration_ms = int((time.monotonic() - started) * 1000)

    _close_capture_logging()
    capture_text = (
        capture_path.read_text(encoding="utf-8") if capture_path.exists() else ""
    )
    workflow_data = _load_workflow_data(state_root)
    correlation_id = _extract_correlation_id(workflow_data)

    if runtime_error:
        status = EnumSkillResultStatus.ERROR
    else:
        # workflow_result is always set when runtime_error is empty.
        status = _WORKFLOW_TO_STATUS[
            workflow_result
            if workflow_result is not None
            else EnumWorkflowResult.FAILED
        ]

    # --- Layer B: durable capture (parent invariant 1) -------------------
    handler_result_json = workflow_data.get("handler_result")
    artifact_payloads: list[dict[str, JsonValue]] = []
    artifact_refs: list[ModelArtifactRef] = []
    try:
        store = ArtifactStore()
        captures: list[tuple[bytes, str, str]] = [
            (capture_text.encode("utf-8"), "text/plain", "runtime_capture_log"),
        ]
        if handler_result_json is not None:
            captures.append(
                (
                    json.dumps(handler_result_json).encode("utf-8"),
                    "application/json",
                    "handler_result",
                )
            )
        for data, media_type, artifact_kind in captures:
            ref = store.write_blob(
                data,
                media_type=media_type,
                artifact_kind=artifact_kind,
                source_system="onex_cli",
                scope_ref=node_name,
                correlation_id=str(correlation_id),
            )
            artifact_refs.append(ref)
            artifact_payloads.append(
                {
                    "artifact_ref": ref.ref,
                    "artifact_hash": ref.hex_digest,
                    "artifact_size_bytes": len(data),
                    "artifact_media_type": media_type,
                    "artifact_kind": artifact_kind,
                    "source_system": "onex_cli",
                    "scope_ref": node_name,
                    "correlation_id": str(correlation_id),
                    "run_id": str(run_id),
                    "redaction_state": "raw",
                }
            )
    except (KeyError, OSError, ValueError) as exc:
        # Artifact write failed ⇒ NO receipt, FULL output (invariant 1).
        reason = f"{type(exc).__name__}: {exc}"
        _print_full_output_on_capture_failure(
            reason=reason,
            capture_text=capture_text,
            workflow_data=workflow_data,
            runtime_error=runtime_error,
        )
        return exit_code

    # --- Telemetry: emit or spool (never re-floods the caller) -----------
    socket_path = emit_socket
    for index, event_payload in enumerate(artifact_payloads):
        _emit_or_spool(
            "artifact.captured",
            event_payload,
            spool_dir=spool_dir,
            spool_stem=f"{run_id}-{index}",
            socket_path=socket_path,
        )
    _emit_or_spool(
        "tool.output.captured",
        {
            "tool_name": "onex_node",
            "node_name": node_name,
            "suppression_decision": "receipt_mode",
            "correlation_id": str(correlation_id),
            "run_id": str(run_id),
            "exit_code": exit_code,
            "status": status.value,
            "artifact_refs": [ref.ref for ref in artifact_refs],
            "capture_log_bytes": len(capture_text.encode("utf-8")),
        },
        spool_dir=spool_dir,
        spool_stem=str(run_id),
        socket_path=socket_path,
    )

    # --- Layer A: exactly one typed receipt on stdout ---------------------
    metrics: dict[str, float] = {
        "capture_log_bytes": float(len(capture_text.encode("utf-8"))),
    }
    receipt: ModelSkillResult[JsonValue] | ModelSkillResult[ModelReceiptRuntimeSummary]
    if (
        status.is_success_like
        and handler_result_json is not None
        and handler_result_obj is not None
    ):
        receipt = ModelSkillResult[JsonValue](
            skill_name=node_name,
            node_name=node_name,
            status=status,
            correlation_id=correlation_id,
            run_id=run_id,
            exit_code=exit_code,
            duration_ms=duration_ms,
            result=handler_result_json,
            result_model=_fully_qualified_name(handler_result_obj),
            metrics=metrics,
            artifact_refs=artifact_refs,
        )
    else:
        summary = ModelReceiptRuntimeSummary(
            workflow_result=(
                workflow_result.value if workflow_result is not None else "error"
            ),
            exit_code=exit_code,
            workflow=str(contract_path),
            terminal_payload=workflow_data.get("terminal_payload"),
            handler_result=handler_result_json,
            error=runtime_error,
            # Errors are never hidden: non-success inlines the FULL capture
            # log; success keeps it behind the artifact ref.
            capture_log="" if status.is_success_like else capture_text,
        )
        receipt = ModelSkillResult[ModelReceiptRuntimeSummary](
            skill_name=node_name,
            node_name=node_name,
            status=status,
            correlation_id=correlation_id,
            run_id=run_id,
            exit_code=exit_code,
            duration_ms=duration_ms,
            result=summary,
            result_model=_fully_qualified_name(summary),
            metrics=metrics,
            artifact_refs=artifact_refs,
        )

    try:
        click.echo(receipt.model_dump_json())
    except ValidationError as exc:  # pragma: no cover - construction validates
        click.echo(f"receipt mode: receipt serialization failed: {exc}", err=True)
        return 1
    return exit_code
