# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime canary: boot a candidate alone, wake every dormant contract, write a receipt (OMN-26463).

Bound from ``nodes/node_runtime_canary_effect/contract.yaml``. The canary is one
runtime container on its own broker namespace and database, deployed through the
deploy agent's instance lane. A run judges a candidate (omnibase_infra,
omnimarket and omnibase_core refs plus an overlay revision) on three checks:

1. ``/ready`` inside the ready budget (5 minutes);
2. every dormant contract woken once with a synthetic first request, then the
   container log read for the first failing line (the failure of 2026-11-22
   16:59:29Z came on a wake, after a boot that had wired 54 contracts);
3. the heartbeat healthy for the hold (15 minutes).

It writes a PASS or FAIL receipt named by the candidate's key. The dev roll's
gate (``HandlerRuntimeCanaryGate``) reads that file and refuses a candidate with
no receipt or a FAIL receipt, so a boot failure cannot end as a SUCCESS job.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums.enum_runtime_canary_verdict import EnumRuntimeCanaryVerdict
from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
)
from omnibase_infra.models.model_runtime_canary_gate_decision import (
    ModelRuntimeCanaryGateDecision,
)
from omnibase_infra.models.model_runtime_canary_receipt import (
    ModelRuntimeCanaryReceipt,
)
from omnibase_infra.models.model_runtime_canary_request import (
    ModelRuntimeCanaryRequest,
)
from omnibase_infra.protocols.protocol_runtime_canary_probe import (
    ProtocolRuntimeCanaryProbe,
)

__all__: list[str] = [
    "RECEIPT_DIR_NAME",
    "HandlerRuntimeCanary",
    "HandlerRuntimeCanaryGate",
    "RuntimeCanaryRefusedError",
    "first_failing_line",
    "judge_candidate",
    "require_for_roll",
    "require_pass",
    "receipt_path",
]

_EXCEPTION_LINE: Final = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit)\b.*$")
_LEVEL_LINE: Final = re.compile(r"\b(?:ERROR|CRITICAL|FATAL)\b")
_TRACEBACK: Final = "Traceback (most recent call last)"
#: A traceback this close under an ERROR line is that line's own.
_TRACEBACK_LOOKAHEAD: Final = 3


def _exception_line_after(lines: list[str], start: int) -> str | None:
    """The unindented exception line that ends the traceback opened at ``start``."""
    for line in lines[start + 1 :]:
        stripped = line.rstrip()
        if stripped and not stripped[0].isspace() and _EXCEPTION_LINE.match(stripped):
            return stripped
    return None


def first_failing_line(log: str) -> str | None:
    """The first failing line of a container log, or None when it holds none.

    A traceback is named by its exception line (``TypeError: ...``), which is the
    line that says what failed; an ERROR, CRITICAL or FATAL log line with no
    traceback under it is named as it stands.
    """
    lines = log.splitlines()
    for index, line in enumerate(lines):
        if _TRACEBACK in line:
            return _exception_line_after(lines, index) or line.strip()
        if _LEVEL_LINE.search(line):
            window = lines[index + 1 : index + 1 + _TRACEBACK_LOOKAHEAD]
            for offset, following in enumerate(window, start=1):
                if _TRACEBACK in following:
                    return _exception_line_after(lines, index + offset) or line.strip()
            return line.strip()
        stripped = line.strip()
        if _EXCEPTION_LINE.match(stripped):
            return stripped
    return None


def receipt_path(receipt_dir: Path, candidate: ModelRuntimeCanaryCandidate) -> Path:
    return receipt_dir / f"{candidate.key}.json"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def judge_candidate(
    request: ModelRuntimeCanaryRequest,
    probe: ProtocolRuntimeCanaryProbe,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log_path: str | None = None,
) -> ModelRuntimeCanaryReceipt:
    """Judge one candidate against a probe, write its receipt and return it."""
    receipt = _judge(request, probe, clock, sleep, log_path)
    request.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path(request.receipt_dir, request.candidate).write_text(
        receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return receipt


class HandlerRuntimeCanary:
    """Run one canary against the lane's probe and write its receipt."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    def handle(self, request: ModelRuntimeCanaryRequest) -> ModelRuntimeCanaryReceipt:
        key: type[object] = ProtocolRuntimeCanaryProbe
        probe = cast("ProtocolRuntimeCanaryProbe", self._container.get_service(key))
        return judge_candidate(request, probe)


def _judge(
    request: ModelRuntimeCanaryRequest,
    probe: ProtocolRuntimeCanaryProbe,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    log_path: str | None,
) -> ModelRuntimeCanaryReceipt:
    started_at = _now()
    t0 = clock()
    facts: dict[str, object] = {
        "candidate": request.candidate,
        "started_at": started_at,
        "log_path": log_path,
    }

    def fail(check: str, fallback: str, **more: object) -> ModelRuntimeCanaryReceipt:
        # The container's own first failing line outranks the canary's words.
        line = first_failing_line(probe.log_text()) or fallback
        return ModelRuntimeCanaryReceipt.model_validate(
            {
                "verdict": EnumRuntimeCanaryVerdict.FAIL,
                "finished_at": _now(),
                "first_failing_line": line,
                "failed_check": check,
                **facts,
                **more,
            }
        )

    # 1. /ready inside the budget.
    while not probe.ready():
        waited = clock() - t0
        if waited >= request.ready_budget_seconds:
            return fail(
                "ready",
                f"/ready not ready within {request.ready_budget_seconds:.0f}s",
                ready_seconds=waited,
            )
        sleep(request.poll_seconds)
    ready_seconds = clock() - t0

    # 2. Every dormant contract woken once, then the log read.
    t_wake = clock()
    dormant = probe.dormant_contracts()
    woken: list[str] = []
    for contract in dormant:
        if probe.wake(contract):
            woken.append(contract)
    unwoken = tuple(c for c in dormant if c not in woken)
    wake_seconds = clock() - t_wake
    wake_facts = {
        "ready_seconds": ready_seconds,
        "wake_seconds": wake_seconds,
        "contracts_dormant": tuple(dormant),
        "contracts_woken": tuple(woken),
        "contracts_unwoken": unwoken,
    }
    if first_failing_line(probe.log_text()) is not None or unwoken:
        return fail(
            "wake",
            f"dormant contract(s) not woken: {', '.join(unwoken)}",
            **wake_facts,
        )

    # 3. The heartbeat healthy for the hold.
    t_hold = clock()
    while clock() - t_hold < request.heartbeat_hold_seconds:
        if not probe.heartbeat_ok():
            held = clock() - t_hold
            return fail(
                "heartbeat",
                f"heartbeat unhealthy after {held:.0f}s of the "
                f"{request.heartbeat_hold_seconds:.0f}s hold",
                heartbeat_seconds=held,
                **wake_facts,
            )
        sleep(request.poll_seconds)
    heartbeat_seconds = clock() - t_hold
    if first_failing_line(probe.log_text()) is not None:
        return fail(
            "heartbeat",
            "the log failed during the hold",
            heartbeat_seconds=heartbeat_seconds,
            **wake_facts,
        )
    return ModelRuntimeCanaryReceipt(
        verdict=EnumRuntimeCanaryVerdict.PASS,
        finished_at=_now(),
        heartbeat_seconds=heartbeat_seconds,
        **{**facts, **wake_facts},
    )


class HandlerRuntimeCanaryGate:
    """The dev roll's read of a candidate's canary receipt."""

    def handle(
        self, request: ModelRuntimeCanaryRequest
    ) -> ModelRuntimeCanaryGateDecision:
        path = receipt_path(request.receipt_dir, request.candidate)
        where = str(path)
        if not path.is_file():
            return ModelRuntimeCanaryGateDecision(
                allowed=False,
                reason=f"no canary receipt for candidate {request.candidate.key}",
                receipt_path=where,
            )
        try:
            receipt = ModelRuntimeCanaryReceipt.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            return ModelRuntimeCanaryGateDecision(
                allowed=False,
                reason=f"canary receipt unreadable: {type(exc).__name__}",
                receipt_path=where,
            )
        if receipt.candidate != request.candidate:
            return ModelRuntimeCanaryGateDecision(
                allowed=False,
                reason="canary receipt names a different candidate",
                receipt_path=where,
            )
        if receipt.verdict is not EnumRuntimeCanaryVerdict.PASS:
            return ModelRuntimeCanaryGateDecision(
                allowed=False,
                reason=f"canary receipt is {receipt.verdict.value}: "
                f"{receipt.first_failing_line}",
                receipt_path=where,
            )
        return ModelRuntimeCanaryGateDecision(
            allowed=True, reason="canary receipt is PASS", receipt_path=where
        )


#: Where each candidate ref comes from in the command's sibling tag map.
_OMNIMARKET: Final = "omnimarket"
_CORE: Final = "omnibase_core"
#: The directory under the agent's state directory the canary lane's receipts are read
#: from. Absent, the agent does not require a receipt: an operator turns the gate on for
#: a lane by creating it, once the canary lane's receipts are delivered there.
RECEIPT_DIR_NAME: Final = "canary-receipts"
#: A sibling tag is ``dev-<UTC stamp>-<12 hex of the commit>``; the commit is the ref.
_TAG_COMMIT: Final = re.compile(r"^dev-[0-9]{14}-([0-9a-f]{12})$")


def _tag_ref(tags: Mapping[str, str], repository: str) -> str:
    match = _TAG_COMMIT.match(tags.get(repository, ""))
    return match.group(1) if match else ""


class RuntimeCanaryRefusedError(RuntimeError):
    """The dev roll may not build this candidate: no PASS canary receipt for it."""


def require_pass(
    receipt_dir: Path,
    *,
    git_ref: str,
    sibling_tags: Mapping[str, str],
    overlay_ref: str,
) -> Path:
    """The path of the candidate's PASS receipt, or :class:`RuntimeCanaryRefusedError`.

    The candidate is the four refs the roll will build: the command's own
    ``git_ref``, the commits of the omnimarket and omnibase_core dev tags it pins and
    its overlay revision. A ref the command leaves to follow a branch tag names no commit, so a
    receipt cannot be for it, and the roll is refused as not pinned.
    """
    gate = HandlerRuntimeCanaryGate()
    refs = {
        "infra_ref": git_ref,
        "omnimarket_ref": _tag_ref(sibling_tags, _OMNIMARKET),
        "core_ref": _tag_ref(sibling_tags, _CORE),
        "overlay_rev": overlay_ref,
    }
    try:
        candidate = ModelRuntimeCanaryCandidate(**refs)
    except ValueError as exc:
        raise RuntimeCanaryRefusedError(
            f"the candidate is not pinned to four commit shas ({refs}): {exc}"
        ) from exc
    decision = gate.handle(
        ModelRuntimeCanaryRequest(candidate=candidate, receipt_dir=receipt_dir)
    )
    if not decision.allowed:
        raise RuntimeCanaryRefusedError(f"{decision.reason} ({decision.receipt_path})")
    return Path(decision.receipt_path)


def require_for_roll(command: Mapping[str, object], state_dir: Path) -> Path | None:
    """The dev roll's receipt read: the PASS receipt's path, or None when no gate is set.

    ``command`` is the roll's rebuild command as the boot guard gets it and
    ``state_dir`` is the agent's state directory. A refusal raises
    :class:`RuntimeCanaryRefusedError` before any build or recreate.
    """
    directory = state_dir / RECEIPT_DIR_NAME
    if not directory.is_dir():
        return None
    tags = command.get("sibling_tags")
    return require_pass(
        directory,
        git_ref=str(command.get("git_ref") or ""),
        sibling_tags=tags if isinstance(tags, Mapping) else {},
        overlay_ref=str(command.get("overlay_ref") or ""),
    )
