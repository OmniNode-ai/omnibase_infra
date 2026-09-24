#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Run the final tenant readbacks and bind them to one issued delegation run.

This is the post-dispatch stage of the K1-K6 evidence runner.  It accepts an
already-captured CLI evidence directory and a write-once execution plan.  The
normal form reads the terminal envelope, correlation-scoped projection row, and
issued lane tenant identity through direct command arrays.  The actual dispatch
port form receives the exact terminal envelope at the broker boundary and
passes that retained path here; only the projection and issued-lane commands
then run.  Commands never pass through a shell.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from bind_delegation_projection_evidence import (
    _nonempty_string,
    _read_json_bytes,
    _terminal_identity,
    bind,
)

from omnibase_infra.runtime.models.model_delegation_terminal_evidence import (
    ModelDelegationTerminalEvidence,
)

_REQUIRED_READBACK_COMMAND_KEYS = (
    "projection_readback_command",
    "issued_lane_tenant_identity_command",
)
_DEFAULT_READBACK_TIMEOUT_SECONDS = 60


def _command(plan: dict[str, Any], key: str, correlation_id: str) -> list[str]:
    value = plan.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"execution plan lacks non-empty {key}")
    command: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"execution plan {key} must contain only non-empty strings"
            )
        command.append(item.replace("{correlation_id}", correlation_id))
    return command


def _readback_timeout_seconds(plan: dict[str, Any]) -> int:
    value = plan.get("readback_timeout_seconds", _DEFAULT_READBACK_TIMEOUT_SECONDS)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("execution plan readback_timeout_seconds must be positive")
    return value


def _run_readback(
    command: list[str], destination: Path, label: str, *, timeout_seconds: int
) -> None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"{label} command exceeded {timeout_seconds}-second readback timeout"
        ) from exc
    if completed.returncode != 0:
        raise ValueError(f"{label} command failed with exit {completed.returncode}")
    if not completed.stdout:
        raise ValueError(f"{label} command returned empty stdout")
    with destination.open("xb") as handle:
        handle.write(completed.stdout)
    destination.chmod(0o600)


def _copy_readback(source: Path, destination: Path, label: str) -> None:
    _value, raw_bytes = _read_json_bytes(source, label)
    with destination.open("xb") as handle:
        handle.write(raw_bytes)
    destination.chmod(0o600)


def finalize(
    evidence_dir: Path,
    execution_plan_path: Path,
    *,
    terminal_envelope_path: Path | None = None,
) -> Path:
    manifest, _ = _read_json_bytes(evidence_dir / "manifest.json", "capture manifest")
    plan, _ = _read_json_bytes(execution_plan_path, "finalization execution plan")
    run_id = _nonempty_string(manifest.get("run_id"), "manifest run_id")
    correlation_id = _nonempty_string(
        manifest.get("correlation_id"), "manifest correlation_id"
    )
    if _nonempty_string(plan.get("captured_run_id"), "plan captured_run_id") != run_id:
        raise ValueError("execution plan run_id does not match captured run")
    if (
        _nonempty_string(plan.get("correlation_id"), "plan correlation_id")
        != correlation_id
    ):
        raise ValueError("execution plan correlation_id does not match captured run")
    timeout_seconds = _readback_timeout_seconds(plan)
    for key in _REQUIRED_READBACK_COMMAND_KEYS:
        _command(plan, key, correlation_id)
    if terminal_envelope_path is None:
        _command(plan, "terminal_readback_command", correlation_id)

    staging = evidence_dir / ".tenant-readbacks.tmp"
    if staging.exists():
        raise ValueError(f"refusing to reuse readback staging directory {staging}")
    staging.mkdir(mode=0o700)
    try:
        terminal_path = staging / "terminal-envelope.json"
        projection_path = staging / "projection-readback.json"
        identity_path = staging / "issued-lane-tenant-identity.json"
        if terminal_envelope_path is None:
            _run_readback(
                _command(plan, "terminal_readback_command", correlation_id),
                terminal_path,
                "terminal readback",
                timeout_seconds=timeout_seconds,
            )
        else:
            _copy_readback(
                terminal_envelope_path,
                terminal_path,
                "actual dispatch terminal envelope",
            )
        _run_readback(
            _command(plan, "projection_readback_command", correlation_id),
            projection_path,
            "projection readback",
            timeout_seconds=timeout_seconds,
        )
        _run_readback(
            _command(plan, "issued_lane_tenant_identity_command", correlation_id),
            identity_path,
            "issued lane tenant identity",
            timeout_seconds=timeout_seconds,
        )
        return bind(
            evidence_dir,
            terminal_path,
            projection_path,
            identity_path,
            execution_plan_path=execution_plan_path,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class FinalK1K6TerminalEvidenceSink:
    """Bind the actual dispatch port's raw terminal to final tenant proof.

    This is the final-run-only sink passed to
    ``RuntimeDelegationDispatchPort(terminal_evidence_sink=...)``.  It retains
    the exact broker-consumed terminal envelope and cursor before it invokes
    the post-dispatch projection readbacks.  A normal port has no sink and
    never enters this code path.
    """

    def __init__(self, evidence_dir: Path, execution_plan_path: Path) -> None:
        self._evidence_dir = evidence_dir
        self._execution_plan_path = execution_plan_path

    async def __call__(self, evidence: ModelDelegationTerminalEvidence) -> None:
        try:
            decoded = evidence.raw_envelope.decode(evidence.encoding)
            terminal = json.loads(decoded)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "broker terminal evidence is not valid encoded JSON"
            ) from exc
        if not isinstance(terminal, dict):
            raise ValueError("broker terminal evidence must be a JSON object")
        terminal_correlation, terminal_tenant = _terminal_identity(terminal)
        if terminal_correlation != str(evidence.correlation_id):
            raise ValueError("broker terminal correlation does not match dispatch")
        if terminal_tenant != evidence.tenant_id:
            raise ValueError("broker terminal tenant does not match dispatch tenant")

        evidence_dir = self._evidence_dir.resolve()
        terminal_dir = evidence_dir / "actual-dispatch-terminal"
        staging = evidence_dir / f".{terminal_dir.name}.tmp"
        if terminal_dir.exists():
            raise ValueError(f"refusing to overwrite retained terminal {terminal_dir}")
        if staging.exists():
            raise ValueError(f"refusing to reuse terminal staging directory {staging}")
        staging.mkdir(mode=0o700)
        try:
            terminal_path = staging / "terminal-envelope.json"
            with terminal_path.open("xb") as handle:
                handle.write(evidence.raw_envelope)
            terminal_path.chmod(0o600)
            cursor_path = staging / "transport-cursor.json"
            with cursor_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": "1.0.0",
                        "topic": evidence.topic,
                        "partition": evidence.partition,
                        "offset": evidence.offset,
                        "encoding": evidence.encoding,
                        "bytes": len(evidence.raw_envelope),
                        "sha256": hashlib.sha256(evidence.raw_envelope).hexdigest(),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            cursor_path.chmod(0o600)
            staging.replace(terminal_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        # Retention is deliberately the only work in the broker observer.
        # Projection readbacks run after the caller-visible port result is
        # captured by the final-run composition command.  Waiting for a
        # projection here could deadlock terminal delivery or consumer progress.

    async def finalize_after_dispatch(self) -> Path:
        """Run bounded projection readbacks after the port has returned.

        The launcher invokes this only after it has retained the request and
        caller-visible result from the same dispatch.  A failure leaves the raw
        terminal intact and writes a durable failure marker; it never converts a
        completed delegation into a claimed tenant proof.
        """
        terminal_dir = self._evidence_dir.resolve() / "actual-dispatch-terminal"
        terminal_path = terminal_dir / "terminal-envelope.json"
        if not terminal_path.is_file():
            raise ValueError("actual dispatch terminal was not retained")
        try:
            return await asyncio.to_thread(
                finalize,
                self._evidence_dir.resolve(),
                self._execution_plan_path,
                terminal_envelope_path=terminal_path,
            )
        except Exception as exc:
            failure_path = terminal_dir / "finalization-failure.json"
            if not failure_path.exists():
                with failure_path.open("x", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "schema_version": "1.0.0",
                            "status": "failed",
                            "reason": str(exc),
                        },
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                    handle.write("\n")
                failure_path.chmod(0o600)
            raise


def build_final_k1_k6_terminal_evidence_sink(
    evidence_dir: Path,
    execution_plan_path: Path,
) -> FinalK1K6TerminalEvidenceSink:
    """Create the explicit post-terminal sink for the final K1-K6 run."""

    return FinalK1K6TerminalEvidenceSink(evidence_dir, execution_plan_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("execution_plan", type=Path)
    args = parser.parse_args(argv)
    try:
        print(finalize(args.evidence_dir, args.execution_plan))
    except ValueError as exc:
        print(f"FINAL_TENANT_PROOF_REJECTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
