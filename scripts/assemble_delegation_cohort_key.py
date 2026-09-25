#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Assemble one delegation cohort key from a terminal receipt and a runtime readback.

OMN-18930 (K3 of OMN-18925). Two inputs, each carrying only what it can prove:

* ``--receipt`` -- the ``onex delegate`` receipt holding the correlation's
  terminal payload. The prompt hash, the resolved task type, the response
  contract hash, the first attempt's route labels and the effective deadline
  are read from the terminal payload bytes, never from the caller's request.
* ``--runtime-readback`` -- a JSON object captured from the lane runtime: the
  consumer that actually handled the correlation (a typed node identity), that
  consumer's build identity, the provider-policy source hashes it resolved, the
  declared retry bounds and the first attempt's provider (or explicit null).

The readback must name the same correlation and lane as the receipt and must
have been captured within 300 seconds of the terminal envelope, the freshness
bound of the plan's event-to-projection proof. Anything missing, mismatched or
stale is refused (exit 2, ``COHORT_KEY_REFUSED: <reason>`` on stderr): a
dimension the evidence did not prove is never filled in from current state.

On success the key is printed to stdout as JSON (exit 0), ready for
``scripts/validate_delegation_cohort_keys.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from omnibase_infra.models.delegation import ModelDelegationCohortKey

FRESHNESS_BOUND_SECONDS = 300.0


class CohortKeyRefusedError(ValueError):
    """The evidence cannot support a complete cohort key."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortKeyRefusedError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CohortKeyRefusedError(f"{label} must be a JSON object: {path}")
    return value


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CohortKeyRefusedError(f"{name} is absent or not an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CohortKeyRefusedError(f"{name} is absent or empty")
    return value


def _timestamp(value: object, name: str) -> datetime:
    text = _string(value, name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CohortKeyRefusedError(f"{name} is not an ISO-8601 time: {text}") from exc
    if parsed.tzinfo is None:
        raise CohortKeyRefusedError(f"{name} carries no timezone: {text}")
    return parsed


def _terminal(receipt: dict[str, Any]) -> dict[str, Any]:
    inner = _object(receipt.get("receipt"), "receipt.receipt")
    result = _object(inner.get("result"), "receipt.result")
    return _object(result.get("terminal_payload"), "receipt.result.terminal_payload")


def _terminal_fields(terminal: dict[str, Any]) -> dict[str, Any]:
    payload = _object(terminal.get("payload"), "terminal payload")
    if terminal.get("correlation_id") != payload.get("correlation_id"):
        raise CohortKeyRefusedError(
            "terminal envelope and payload disagree on correlation_id"
        )

    prompt = _string(payload.get("prompt_text"), "terminal payload prompt_text")

    if "response_contract_evidence" not in payload:
        raise CohortKeyRefusedError(
            "terminal payload carries no response_contract_evidence"
        )
    contract_evidence = payload["response_contract_evidence"]
    if not isinstance(contract_evidence, dict) or (
        "contract_sha256" not in contract_evidence
    ):
        raise CohortKeyRefusedError(
            "terminal response_contract_evidence states neither a contract hash "
            "nor an explicit absence"
        )

    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise CohortKeyRefusedError("terminal payload carries no attempts")
    first = _object(attempts[0], "terminal payload attempts[0]")

    budget = payload.get("budget_evidence")
    timeout = (
        budget.get("execution_timeout_seconds") if isinstance(budget, dict) else None
    )
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise CohortKeyRefusedError(
            "terminal budget_evidence carries no execution_timeout_seconds"
        )

    return {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "resolved_task_type": _string(
            payload.get("task_type"), "terminal payload task_type"
        ),
        "response_contract_sha256": contract_evidence["contract_sha256"],
        "first_hop": {
            "backend": _string(first.get("backend_id"), "attempts[0].backend_id"),
            "model": _string(first.get("model_id"), "attempts[0].model_id"),
            "tier": _string(first.get("tier"), "attempts[0].tier"),
        },
        "deadline_seconds": float(timeout),
    }


def assemble(
    receipt: dict[str, Any], readback: dict[str, Any]
) -> ModelDelegationCohortKey:
    """Return the complete key, or raise ``CohortKeyRefusedError``."""
    terminal = _terminal(receipt)
    correlation_id = _string(terminal.get("correlation_id"), "terminal correlation_id")
    if receipt.get("correlation_id") != correlation_id:
        raise CohortKeyRefusedError("receipt and terminal disagree on correlation_id")
    if readback.get("correlation_id") != correlation_id:
        raise CohortKeyRefusedError(
            "runtime readback correlation_id does not match the terminal"
        )

    lane = _string(receipt.get("lane"), "receipt lane")
    if readback.get("lane") != lane:
        raise CohortKeyRefusedError(
            f"runtime readback lane {readback.get('lane')!r} is not the receipt lane "
            f"{lane!r}"
        )

    terminal_at = _timestamp(
        terminal.get("envelope_timestamp"), "terminal envelope_timestamp"
    )
    captured_at = _timestamp(readback.get("captured_at"), "readback captured_at")
    skew = abs((captured_at - terminal_at).total_seconds())
    if skew > FRESHNESS_BOUND_SECONDS:
        raise CohortKeyRefusedError(
            f"readback captured_at is {skew:.0f}s from the terminal; the bound is "
            f"{FRESHNESS_BOUND_SECONDS:.0f}s"
        )

    fields = _terminal_fields(terminal)
    if "first_hop_provider" not in readback:
        raise CohortKeyRefusedError(
            "runtime readback states no first_hop_provider (explicit null allowed)"
        )
    for name in (
        "consumer_identity",
        "build_identity",
        "provider_policy",
        "retry_bounds",
    ):
        _object(readback.get(name), f"runtime readback {name}")

    candidate = {
        "prompt_sha256": fields["prompt_sha256"],
        "resolved_task_type": fields["resolved_task_type"],
        "response_contract_sha256": fields["response_contract_sha256"],
        "lane": lane,
        "build_identity": readback["build_identity"],
        "consumer_identity": readback["consumer_identity"],
        "first_hop_identity": {
            **fields["first_hop"],
            "provider": readback["first_hop_provider"],
        },
        "provider_policy": readback["provider_policy"],
        "deadline_seconds": fields["deadline_seconds"],
        "retry_bounds": readback["retry_bounds"],
    }
    try:
        return ModelDelegationCohortKey.model_validate_json(json.dumps(candidate))
    except ValidationError as exc:
        raise CohortKeyRefusedError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--runtime-readback", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        key = assemble(
            _read_object(args.receipt, "receipt"),
            _read_object(args.runtime_readback, "runtime readback"),
        )
    except CohortKeyRefusedError as exc:
        print(f"COHORT_KEY_REFUSED: {exc}", file=sys.stderr)
        return 2
    print(key.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
