# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure C28 grading and the injected-target effect handler (OMN-19931).

Counters are per-window counts, not cumulative offsets. The audit-only kind
can report STALLED while counting input and publishing nothing; this matches
C28's recorded PASS and is intentionally not a new grading criterion.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import TypeGuard

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    APPLIED_TOPIC,
    AST_GATE_TEST,
    BRANCHES,
    KINDS,
    LIVE_WINDOW_MINUTES,
    RUNTIME_CONTAINERS,
    SEAM_DLQ,
    WIRING_MODULE,
)
from omnibase_infra.nodes.node_board_probe_effect.models import (
    EnumBoardCheckId,
    EnumBoardCheckSurfaceClass,
    EnumBoardProbeOutcome,
    ModelBoardProbeResult,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_check import (
    ModelConsumerFlowCheck,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_observation import (
    ModelConsumerFlowObservation,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)
from omnibase_infra.nodes.node_board_probe_effect.models.typed_dict_consumer_flow import (
    TypedDictConsumerFlowBoot,
    TypedDictConsumerFlowCursor,
    TypedDictConsumerFlowInjection,
    TypedDictConsumerFlowNatural,
    TypedDictConsumerFlowNegative,
    TypedDictConsumerFlowPage,
    TypedDictConsumerFlowRow,
    TypedDictConsumerFlowRun,
)
from omnibase_infra.nodes.node_board_probe_effect.protocols.protocol_consumer_flow_target import (
    ProtocolConsumerFlowTarget,
)


def _add(
    rec: list[ModelConsumerFlowCheck], clause: str, name: str, ok: bool, evidence: str
) -> None:
    rec.append(
        ModelConsumerFlowCheck(clause=clause, name=name, ok=bool(ok), evidence=evidence)
    )


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def kind_rows(
    rows: Sequence[TypedDictConsumerFlowRow], prefix: str, topic: str
) -> list[TypedDictConsumerFlowRow]:
    return [
        r
        for r in rows
        if isinstance(r, dict)
        and str(r.get("consumer_group", "")).startswith(prefix)
        and r.get("topic") == topic
    ]


def grade_kinds(
    rec: list[ModelConsumerFlowCheck],
    samples: Sequence[Sequence[TypedDictConsumerFlowRow]],
) -> None:
    """CLAUSE 1 over ``samples``: each is every row one full walk returned."""
    _add(
        rec,
        "kinds",
        "sampled_more_than_once",
        len(samples) >= 2,
        f"{len(samples)} sample(s); one sample cannot show a window advancing",
    )
    for kind, prefix, topic, counter in KINDS:
        per_sample = [kind_rows(s, prefix, topic) for s in samples]
        present = [bool(rows) for rows in per_sample]
        _add(
            rec,
            "kinds",
            f"{kind}_present_in_every_sample",
            bool(per_sample) and all(present),
            f"{prefix}* on {topic}: present per sample {present}",
        )
        flat = [r for rows in per_sample for r in rows]
        bad = [
            (
                r.get("consumer_group"),
                r.get("flow_state"),
                r.get("messages_in"),
                r.get("messages_out"),
            )
            for r in flat
            if not (
                _is_int(r.get("messages_in"))
                and _is_int(r.get("messages_out"))
                and r.get("flow_state") not in (None, "UNKNOWN")
            )
        ]
        _add(
            rec,
            "kinds",
            f"{kind}_instrumented",
            bool(flat) and not bad,
            "every row has integer in/out counters and a state other than UNKNOWN"
            if flat and not bad
            else f"uninstrumented rows (group, state, in, out): {bad[:5]} of {len(flat)}",
        )
        starts = sorted(
            {str(r.get("window_start")) for r in flat if r.get("window_start")}
        )
        _add(
            rec,
            "kinds",
            f"{kind}_window_advances",
            len(starts) >= 2,
            f"{len(starts)} distinct window_start value(s) across the samples: {starts[:6]}",
        )
        moved = [value for r in flat if _is_int(value := r.get(counter))]
        _add(
            rec,
            "kinds",
            f"{kind}_{counter}_counted",
            any(v > 0 for v in moved),
            f"{counter} per row across the samples: {moved[:12]}",
        )


def _param(name: str) -> str | None:
    m = re.search(r"\[([^\]]+)\]$", name)
    return m.group(1) if m else None


def grade_negative(
    rec: list[ModelConsumerFlowCheck], neg: TypedDictConsumerFlowNegative
) -> None:
    """CLAUSE 2 over the clean run and one run per mutated branch."""
    clean: TypedDictConsumerFlowRun = neg.get("clean") or {}
    outcomes: dict[str, str] = clean.get("outcomes") or {}
    not_passed = sorted(n for n, o in outcomes.items() if o != "passed")
    _add(
        rec,
        "negative",
        "tests_pass_on_this_checkout",
        bool(outcomes) and not not_passed and clean.get("returncode") == 0,
        f"{len(outcomes)} collected, not passed: {not_passed}, pytest exit {clean.get('returncode')}",
    )
    params = {_param(n) for n in outcomes}
    missing_branches = sorted(b for b in BRANCHES if b not in params)
    _add(
        rec,
        "negative",
        "both_branches_are_parametrized",
        not missing_branches,
        f"parametrized ids seen {sorted(p for p in params if p)}; missing {missing_branches}",
    )
    for branch in BRANCHES:
        run: TypedDictConsumerFlowRun = (neg.get("mutations") or {}).get(branch) or {}
        mo: dict[str, str] = run.get("outcomes") or {}
        failed = sorted(n for n, o in mo.items() if o in ("failed", "error"))
        own = [n for n in failed if _param(n) == branch]
        foreign = [n for n in failed if _param(n) not in (None, branch)]
        others_passing = [
            n
            for n, o in mo.items()
            if _param(n) not in (None, branch) and o == "passed"
        ]
        _add(
            rec,
            "negative",
            f"mutation_{branch}_bites_its_own_branch",
            bool(run.get("applied")) and bool(own),
            f"removed the registration in {BRANCHES[branch]} (applied={run.get('applied')}); "
            f"failing [{branch}] cases: {own}",
        )
        _add(
            rec,
            "negative",
            f"mutation_{branch}_is_branch_specific",
            bool(run.get("applied")) and not foreign and bool(others_passing),
            f"failing cases of the other branch: {foreign}; other-branch cases still "
            f"passing: {len(others_passing)} (zero would mean the module did not run)",
        )
        _add(
            rec,
            "negative",
            f"mutation_{branch}_fails_the_ast_gate",
            mo.get(AST_GATE_TEST) in ("failed", "error"),
            f"{AST_GATE_TEST}: {mo.get(AST_GATE_TEST)}",
        )
        _add(
            rec,
            "negative",
            f"mutation_{branch}_restored",
            bool(run.get("restored")),
            f"{WIRING_MODULE} digest after restore equals the original: {run.get('restored')}",
        )


def grade_cursor(
    rec: list[ModelConsumerFlowCheck], cur: TypedDictConsumerFlowCursor
) -> None:
    """CLAUSE 3 over one full walk and the live window read before it."""
    pages: list[TypedDictConsumerFlowPage] = cur.get("pages") or []
    _add(
        rec,
        "cursor",
        "walk_terminates",
        bool(pages) and bool(cur.get("terminated")),
        f"{len(pages)} page(s); last next_cursor "
        f"{pages[-1].get('next_cursor') if pages else None!r}; terminated={cur.get('terminated')}",
    )
    # OMN-19812: the fixed 2000-row served window (omnimarket#3119, OMN-20152)
    # makes an exactly-full final page normal, so the end is measured, not assumed.
    silent_truncation = [
        i
        for i, p in enumerate(pages)
        if _is_int(count := p.get("row_count"))
        and _is_int(limit := p.get("row_limit"))
        and count >= limit
        and not p.get("next_cursor")
        and not (
            isinstance(end_proof := p.get("end_proof"), dict)
            and end_proof.get("proven") is True
        )
    ]
    proof = pages[-1].get("end_proof") if pages else None
    _add(
        rec,
        "cursor",
        "truncated_pages_carry_a_cursor",
        bool(pages) and not silent_truncation,
        f"pages at row_limit with a null next_cursor: {silent_truncation}"
        f"; full final page proven the end by a since= read past it: {proof}",
    )
    advancing = cur.get("second_page_differs")
    _add(
        rec,
        "cursor",
        "since_advances",
        bool(pages) and (len(pages) < 2 or advancing is True),
        "single page, nothing to advance past"
        if len(pages) < 2
        else f"page 2 via since= differs from page 1: {advancing}",
    )
    live = set(cur.get("live_groups") or [])
    walked = set(cur.get("walked_groups") or [])
    _add(
        rec,
        "cursor",
        "live_window_is_non_empty",
        bool(live),
        f"{len(live)} distinct consumer groups in the last {LIVE_WINDOW_MINUTES} minutes "
        "(an empty window would make the comparison vacuous)",
    )
    unreachable = sorted(live - walked)
    _add(
        rec,
        "cursor",
        "every_live_group_reachable",
        bool(live) and not unreachable,
        f"live {len(live)}, walked {len(walked)}, unreachable {len(unreachable)}: {unreachable[:10]}",
    )


def grade_boot(
    rec: list[ModelConsumerFlowCheck], boot: TypedDictConsumerFlowBoot
) -> None:
    before, after = boot.get("applied_hwm_before"), boot.get("applied_hwm_after")
    _add(
        rec,
        "boot",
        "applied_event_on_this_boot",
        isinstance(before, int) and isinstance(after, int) and after > before,
        f"{APPLIED_TOPIC} high-watermark {before} -> {after} over "
        f"{boot.get('applied_window_seconds')}s, before this probe published anything",
    )
    logs: dict[str, TypedDictConsumerFlowNatural] = boot.get("natural") or {}
    _add(
        rec,
        "boot",
        "runtime_logs_read",
        set(logs) == set(RUNTIME_CONTAINERS)
        and all(_is_int(lines := v.get("lines")) and lines > 0 for v in logs.values()),
        "log lines read per container: "
        + ", ".join(f"{c}={v.get('lines')}" for c, v in sorted(logs.items())),
    )
    natural = {c: v.get("natural") for c, v in sorted(logs.items())}
    _add(
        rec,
        "boot",
        "zero_natural_stall_alert_validation_errors",
        bool(logs) and all(v == 0 for v in natural.values()),
        f"natural errors per container {natural} (total, this probe's: "
        + ", ".join(
            f"{c}={v.get('total')}/{v.get('probe')}" for c, v in sorted(logs.items())
        )
        + ")",
    )
    inj: TypedDictConsumerFlowInjection = boot.get("injection") or {}
    _add(
        rec,
        "boot",
        "injected_payload_published",
        _is_int(inj.get("offset")),
        f"offset {inj.get('offset')} on {APPLIED_TOPIC}, correlation {inj.get('correlation_id')}",
    )
    _add(
        rec,
        "boot",
        "seam_raised_a_validation_error_after_the_publish",
        _is_int(validation_errors_after := inj.get("validation_errors_after"))
        and validation_errors_after > 0,
        f"stall-alert validation errors after the publish: {inj.get('validation_errors_after')} "
        "(the counter that must read 0 above is shown able to read non-zero)",
    )
    _add(
        rec,
        "boot",
        "seam_dead_lettered_the_injected_correlation",
        _is_int(boundary_lines := inj.get("boundary_lines")) and boundary_lines > 0,
        f"boundary_swallow_prevented dlq_routed=true lines naming the injected "
        f"correlation id: {inj.get('boundary_lines')}",
    )
    _add(
        rec,
        "boot",
        "injected_marker_durably_on_the_dlq",
        _is_int(dlq_copies := inj.get("dlq_copies")) and dlq_copies > 0,
        f"{SEAM_DLQ} messages carrying the marker: {inj.get('dlq_copies')}",
    )


def grade_consumer_flow(
    request: ModelConsumerFlowRequest, observation: ModelConsumerFlowObservation
) -> ModelBoardProbeResult:
    """Grade every script expectation; unreadable or empty is indeterminate."""
    checks: list[ModelConsumerFlowCheck] = []
    reasons: tuple[str, ...]
    if not observation.read_ok:
        outcome = EnumBoardProbeOutcome.INDETERMINATE
        reasons = (f"consumer flow unreadable: {observation.read_error}",)
    elif not any(
        (observation.kinds, observation.negative, observation.cursor, observation.boot)
    ):
        outcome = EnumBoardProbeOutcome.INDETERMINATE
        reasons = ("consumer flow observation is empty",)
    else:
        grade_kinds(checks, observation.kinds.get("samples") or [])
        grade_negative(checks, observation.negative)
        grade_cursor(checks, observation.cursor)
        grade_boot(checks, observation.boot)
        failures = tuple(
            f"{c.clause}/{c.name}: {c.evidence}" for c in checks if not c.ok
        )
        if not checks:
            outcome = EnumBoardProbeOutcome.INDETERMINATE
            reasons = ("consumer flow produced no checks",)
        else:
            outcome = (
                EnumBoardProbeOutcome.FAIL if failures else EnumBoardProbeOutcome.PASS
            )
            reasons = failures or (f"all {len(checks)} consumer-flow checks held",)
    return ModelBoardProbeResult(
        check_id=EnumBoardCheckId.CONSUMER_FLOW,
        surface_class=EnumBoardCheckSurfaceClass.LAB_HARDWARE,
        subject=request.subject_lane,
        outcome=outcome,
        reasons=reasons,
        evidence_items=tuple(f"{c.clause}/{c.name}: {c.evidence}" for c in checks),
        observed_at=observation.observed_at,
    )


class HandlerConsumerFlow:
    """Observe through an injected target, then apply the pure C28 grader."""

    def __init__(self, target: ProtocolConsumerFlowTarget | None = None) -> None:
        from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
            HandlerDockerConsumerFlowTarget,
        )

        self._target = (
            target if target is not None else HandlerDockerConsumerFlowTarget()
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: host infrastructure I/O."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Effectful observation followed by pure grading."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelConsumerFlowRequest) -> ModelBoardProbeResult:
        """Collect and grade once."""
        observation = await self._target.observe(request)
        result = grade_consumer_flow(request, observation)
        if request.record is not None:
            payload = (
                json.dumps(
                    {
                        "version": 1,
                        "criterion": "C28",
                        "result": result.model_dump(mode="json"),
                        "observation": observation.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            await asyncio.to_thread(
                request.record.write_text, payload, encoding="utf-8"
            )
        return result
