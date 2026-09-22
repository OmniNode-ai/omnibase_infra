#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Execute one planned K1-K6 delegation and bind its terminal proof.

This command is the only executable composition root for final K1-K6 capture.
It accepts one write-once JSON plan, validates it before constructing a bus,
uses the normal Kafka event-bus constructor and RuntimeDelegationDispatchPort,
and records request, caller output, raw terminal/cursor, and projection proof
from that same dispatch.  It has no transport fallback and never reads a prior
CLI receipt as first-stage evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from finalize_delegation_projection_evidence import (
    build_final_k1_k6_terminal_evidence_sink,
)

from omnibase_infra.backends.auto_configure import select_event_bus
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

_ROUTE_ALIAS = "delegation.orchestrate"
_CAPTURE_KIND = "runtime_delegation_port"


def _object_from_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_finalization_plan(
    path: Path, *, run_id: str, correlation_id: str
) -> None:
    """Reject a malformed finalizer before the command can issue work."""
    plan, _raw = _object_from_file(path, "finalization execution plan")
    if _string(plan.get("captured_run_id"), "finalization captured_run_id") != run_id:
        raise ValueError("finalization plan run_id does not match dispatch plan")
    if (
        _string(plan.get("correlation_id"), "finalization correlation_id")
        != correlation_id
    ):
        raise ValueError(
            "finalization plan correlation_id does not match dispatch plan"
        )
    _positive_int(
        plan.get("readback_timeout_seconds", 60), "finalization readback timeout"
    )
    for key in (
        "projection_readback_command",
        "issued_lane_tenant_identity_command",
    ):
        command = plan.get(key)
        if not isinstance(command, list) or not command:
            raise ValueError(f"finalization plan lacks non-empty {key}")
        if any(not isinstance(item, str) or not item for item in command):
            raise ValueError(
                f"finalization plan {key} must contain only non-empty strings"
            )


def _strict_plan(plan_path: Path) -> dict[str, Any]:
    plan, _raw = _object_from_file(plan_path, "final K1-K6 dispatch plan")
    _string(plan.get("run_id"), "plan run_id")
    correlation_id = _string(plan.get("correlation_id"), "plan correlation_id")
    try:
        UUID(correlation_id)
    except ValueError as exc:
        raise ValueError("plan correlation_id must be a UUID") from exc
    evidence_dir = Path(_string(plan.get("evidence_dir"), "plan evidence_dir"))
    if evidence_dir.exists():
        raise ValueError("plan evidence_dir must not already exist")
    finalization_plan = Path(
        _string(plan.get("finalization_execution_plan"), "finalization plan")
    )
    if not finalization_plan.is_file():
        raise ValueError("finalization_execution_plan must name an existing file")
    _validate_finalization_plan(
        finalization_plan,
        run_id=_string(plan.get("run_id"), "plan run_id"),
        correlation_id=correlation_id,
    )
    transport = plan.get("transport")
    if not isinstance(transport, dict):
        raise ValueError("plan transport must be an object")
    if _string(transport.get("bus_type"), "transport bus_type") != "kafka":
        raise ValueError("final K1-K6 capture requires explicit kafka transport")
    bootstrap = _string(transport.get("kafka_bootstrap_servers"), "kafka bootstrap")
    if "localhost" in bootstrap or "127.0.0.1" in bootstrap:
        raise ValueError("final K1-K6 capture forbids local Kafka fallback")
    _string(transport.get("environment"), "transport environment")
    _string(transport.get("consumer_group"), "transport consumer_group")
    dispatch = plan.get("dispatch")
    if not isinstance(dispatch, dict):
        raise ValueError("plan dispatch must be an object")
    if _string(dispatch.get("route_alias"), "dispatch route_alias") != _ROUTE_ALIAS:
        raise ValueError(
            "dispatch route_alias does not name the canonical delegation route"
        )
    if (
        _string(dispatch.get("correlation_id"), "dispatch correlation_id")
        != correlation_id
    ):
        raise ValueError("dispatch correlation_id does not match plan correlation_id")
    if _string(dispatch.get("tenant_id"), "dispatch tenant_id") != _string(
        plan.get("tenant_id"), "plan tenant_id"
    ):
        raise ValueError("dispatch tenant_id does not match planned lane tenant")
    _string(dispatch.get("prompt"), "dispatch prompt")
    _string(dispatch.get("task_type"), "dispatch task_type")
    _positive_int(dispatch.get("max_tokens"), "dispatch max_tokens")
    _positive_int(dispatch.get("execution_timeout_seconds"), "dispatch timeout")
    _positive_int(
        dispatch.get("terminal_delivery_margin_seconds"), "dispatch terminal margin"
    )
    if dispatch.get("wait") is not True:
        raise ValueError("final K1-K6 proof dispatch requires wait=true")
    return plan


def _write_private(path: Path, payload: bytes) -> dict[str, Any]:
    with path.open("xb") as handle:
        handle.write(payload)
    path.chmod(0o600)
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_first_stage(
    *,
    evidence_dir: Path,
    plan_path: Path,
    plan: dict[str, Any],
    response: dict[str, object],
) -> None:
    """Record bytes emitted by this command after the actual port dispatch."""
    terminal_dir = evidence_dir / "actual-dispatch-terminal"
    terminal_path = terminal_dir / "terminal-envelope.json"
    cursor_path = terminal_dir / "transport-cursor.json"
    if not terminal_path.is_file() or not cursor_path.is_file():
        raise ValueError("dispatch returned without a retained raw terminal/cursor")
    dispatch = plan["dispatch"]
    assert isinstance(dispatch, dict)
    if not evidence_dir.is_dir():
        raise ValueError("evidence directory was not initialized by the composition")
    request_bytes = (
        json.dumps(dispatch, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    # This is the exact caller-visible representation printed by main after
    # finalization.  It is intentionally labelled derived JSON, not broker wire.
    response_bytes = (
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    plan_bytes = plan_path.read_bytes()
    terminal_bytes = terminal_path.read_bytes()
    cursor_bytes = cursor_path.read_bytes()
    evidence = {
        "request": _write_private(evidence_dir / "request.json", request_bytes),
        "caller_response": _write_private(
            evidence_dir / "caller-response.json", response_bytes
        ),
        "terminal_envelope": {
            "path": "actual-dispatch-terminal/terminal-envelope.json",
            "bytes": len(terminal_bytes),
            "sha256": hashlib.sha256(terminal_bytes).hexdigest(),
        },
        "terminal_cursor": {
            "path": "actual-dispatch-terminal/transport-cursor.json",
            "bytes": len(cursor_bytes),
            "sha256": hashlib.sha256(cursor_bytes).hexdigest(),
        },
        "dispatch_plan": _write_private(
            evidence_dir / "dispatch-plan.json", plan_bytes
        ),
    }
    manifest = {
        "schema_version": "1.0.0",
        "capture_kind": _CAPTURE_KIND,
        "run_id": plan["run_id"],
        "correlation_id": plan["correlation_id"],
        "tenant_proof": {"state": "pending_authoritative_projection_readback"},
        "caller_response": {"representation": "derived_json_emitted_by_this_command"},
        "evidence": evidence,
    }
    with (evidence_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (evidence_dir / "manifest.json").chmod(0o600)


async def run(
    plan_path: Path,
    *,
    event_bus_factory: Callable[..., object] = select_event_bus,
    port_factory: Callable[
        ..., RuntimeDelegationDispatchPort
    ] = RuntimeDelegationDispatchPort,
) -> dict[str, object]:
    """Run the strict final composition; injection exists only for unit tests."""
    plan = _strict_plan(plan_path)
    transport = plan["transport"]
    dispatch = plan["dispatch"]
    assert isinstance(transport, dict) and isinstance(dispatch, dict)
    evidence_dir = Path(plan["evidence_dir"])
    # Strict validation is complete; initialize the write-once root before the
    # broker can retain its terminal.  Rejected plans never construct a bus.
    evidence_dir.mkdir(mode=0o700)
    # Build only after strict plan validation; no bus exists for rejected plans.
    event_bus = event_bus_factory(
        bus_type="kafka",
        kafka_bootstrap_servers=transport["kafka_bootstrap_servers"],
        environment=transport["environment"],
        consumer_group=transport["consumer_group"],
    )
    sink = build_final_k1_k6_terminal_evidence_sink(
        evidence_dir, Path(plan["finalization_execution_plan"])
    )
    constructed = True
    try:
        start = getattr(event_bus, "start", None)
        if not callable(start):
            raise ValueError("canonical Kafka transport does not expose start")
        await start()
        port = port_factory(event_bus, terminal_evidence_sink=sink)
        response = await port.dispatch(
            prompt=dispatch["prompt"],
            task_type=dispatch["task_type"],
            source_session_id=dispatch.get("source_session_id"),
            source_file_path=dispatch.get("source_file_path"),
            correlation_id=UUID(dispatch["correlation_id"]),
            max_tokens=dispatch["max_tokens"],
            wait=dispatch["wait"],
            execution_timeout_seconds=dispatch["execution_timeout_seconds"],
            terminal_delivery_margin_seconds=dispatch[
                "terminal_delivery_margin_seconds"
            ],
            tenant_id=dispatch["tenant_id"],
        )
        _write_first_stage(
            evidence_dir=evidence_dir, plan_path=plan_path, plan=plan, response=response
        )
        await sink.finalize_after_dispatch()
        return response
    except Exception:
        if (
            evidence_dir.exists()
            and not (evidence_dir / "composition-failure.json").exists()
        ):
            with (evidence_dir / "composition-failure.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump({"schema_version": "1.0.0", "status": "failed"}, handle)
                handle.write("\n")
            (evidence_dir / "composition-failure.json").chmod(0o600)
        raise
    finally:
        if constructed:
            close = getattr(event_bus, "close", None)
            if callable(close):
                await close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    try:
        response = asyncio.run(run(args.plan))
    except (ValueError, RuntimeError) as exc:
        print(f"FINAL_K1_K6_EVIDENCE_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
