# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28 collection and branch-specific negative controls, ported unchanged.

The sole lane write is the marked malformed trigger. Negative controls
restore the checkout's wiring source byte for byte even if pytest fails.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import re
import subprocess
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path

from omnibase_infra.nodes.node_board_probe_effect.handlers.consumer_flow_constants import (
    APPLIED_TOPIC,
    BOOT_CONTAINERS,
    BOUNDARY_RE,
    BRANCHES,
    GENERIC_DLQ,
    KINDS,
    NEGATIVE_TESTS,
    PAIRING_LOOKAHEAD,
    PROBE_CID_PREFIX,
    RUNTIME_CONTAINERS,
    SEAM_DLQ,
    VALIDATION_ERROR_RE,
    WIRING_MODULE,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.consumer_flow_lane import (
    ConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.error_consumer_flow_boot_changed import (
    ConsumerFlowBootChangedError,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.error_consumer_flow_input import (
    ConsumerFlowInputError,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    kind_rows,
)
from omnibase_infra.nodes.node_board_probe_effect.models.typed_dict_consumer_flow import (
    TypedDictConsumerFlowCollected,
    TypedDictConsumerFlowInjection,
    TypedDictConsumerFlowNatural,
    TypedDictConsumerFlowNegative,
    TypedDictConsumerFlowPage,
    TypedDictConsumerFlowRow,
    TypedDictConsumerFlowRun,
    TypedDictConsumerFlowWalk,
)


def natural_validation_errors(
    lines: Sequence[str],
) -> tuple[int, int, int]:
    """(total, caused by this probe, natural) stall-alert validation errors.

    An error is this probe's when the next applied-topic boundary line after it,
    in the same container's log, carries a correlation id with the probe
    prefix. An error with no such line within ``PAIRING_LOOKAHEAD`` lines is
    natural: an unattributed error is never excused.
    """
    total = probe = 0
    for i, line in enumerate(lines):
        if not VALIDATION_ERROR_RE.search(line):
            continue
        total += 1
        for nxt in lines[i + 1 : i + 1 + PAIRING_LOOKAHEAD]:
            m = BOUNDARY_RE.search(nxt)
            if m:
                if m.group("cid").startswith(PROBE_CID_PREFIX):
                    probe += 1
                break
    return total, probe, total - probe


def junit_outcomes(xml_text: str) -> dict[str, str]:
    """Map ``name`` (with its ``[param]`` id) to passed / failed / error / skipped."""
    try:
        root = ET.fromstring(xml_text)  # noqa: S314 — JUnit XML this job's own pytest wrote
    except ET.ParseError as exc:
        raise ConsumerFlowInputError(
            f"pytest wrote unparseable junit xml: {exc}"
        ) from exc
    out: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = case.get("name") or ""
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[
                    child.tag
                ]
                break
        out[name] = outcome
    return out


def walk(lane: ConsumerFlowLane, max_pages: int = 200) -> TypedDictConsumerFlowWalk:
    pages: list[TypedDictConsumerFlowPage] = []
    rows: list[TypedDictConsumerFlowRow] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    first_rows: list[TypedDictConsumerFlowRow] = []
    second_rows: list[TypedDictConsumerFlowRow] = []
    terminated = False
    while len(pages) < max_pages:
        body = lane.page({"since": cursor} if cursor else {})
        page_rows = body["rows"]
        if not pages:
            first_rows = page_rows[:3]
        elif len(pages) == 1:
            second_rows = page_rows[:3]
        pages.append(
            {
                "row_count": body.get("row_count"),
                "row_limit": body.get("row_limit"),
                "next_cursor": body.get("next_cursor"),
                "backing": body.get("backing"),
                "data_freshness": body.get("data_freshness"),
            }
        )
        rows.extend(page_rows)
        nxt = body.get("next_cursor")
        if not nxt:
            terminated = True
            break
        if str(nxt) in seen_cursors:
            break
        seen_cursors.add(str(nxt))
        cursor = str(nxt)
    return {
        "pages": pages,
        "rows": rows,
        "terminated": terminated,
        "second_page_differs": (second_rows != first_rows) if len(pages) > 1 else None,
    }


def _mutated_source(source: str, factory: str) -> str | None:
    """``source`` with the one ``flow_counters.register(...)`` in ``factory`` replaced by ``pass``."""
    tree = ast.parse(source)
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name == factory
        ),
        None,
    )
    if fn is None:
        return None
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "register"
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "flow_counters"
    ]
    end = calls[0].end_lineno if len(calls) == 1 else None
    if end is None:
        return None
    lines = source.splitlines(keepends=True)
    first, last = calls[0].lineno - 1, end - 1
    indent = lines[first][: len(lines[first]) - len(lines[first].lstrip())]
    lines[first : last + 1] = [f"{indent}pass  # C28 probe mutation\n"]
    return "".join(lines)


def run_negative(
    repo: Path,
    pytest_cmd: Sequence[str],
    scratch: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> TypedDictConsumerFlowNegative:
    def run_pytest_pass(tag: str) -> TypedDictConsumerFlowRun:
        xml = scratch / f"c28-negative-{tag}.xml"
        xml.unlink(missing_ok=True)
        cmd = [
            *pytest_cmd,
            *NEGATIVE_TESTS,
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml}",
        ]
        try:
            proc = runner(
                cmd, cwd=repo, capture_output=True, text=True, timeout=900, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ConsumerFlowInputError(
                f"pytest ({tag}) could not run: {type(exc).__name__}"
            ) from exc
        if not xml.exists():
            raise ConsumerFlowInputError(
                f"pytest ({tag}) wrote no junit xml, exit {proc.returncode}: {proc.stdout[-400:]}"
            )
        return {
            "returncode": proc.returncode,
            "outcomes": junit_outcomes(xml.read_text("utf-8")),
        }

    target = repo / WIRING_MODULE
    original = target.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    result: TypedDictConsumerFlowNegative = {
        "clean": run_pytest_pass("clean"),
        "mutations": {},
        "module_sha256": digest,
    }
    for branch, factory in BRANCHES.items():
        mutated = _mutated_source(original.decode("utf-8"), factory)
        entry: TypedDictConsumerFlowRun = {
            "factory": factory,
            "applied": mutated is not None,
        }
        if mutated is not None:
            try:
                target.write_text(mutated, encoding="utf-8")
                entry.update(run_pytest_pass(branch))
            finally:
                target.write_bytes(original)
        entry["restored"] = hashlib.sha256(target.read_bytes()).hexdigest() == digest
        result["mutations"][branch] = entry
    return result


def probe_correlation_id() -> str:
    return str(uuid.UUID(hex="c28c28c2c28c" + uuid.uuid4().hex[12:]))


def inject(lane: ConsumerFlowLane) -> TypedDictConsumerFlowInjection:
    """Publish one malformed trigger, envelope copied from the newest real one."""
    newest = lane.rpk("topic", "consume", APPLIED_TOPIC, "-o", "-1", "-n", "1")
    try:
        message = json.loads(newest)
        envelope = json.loads(message["value"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ConsumerFlowInputError(
            f"newest {APPLIED_TOPIC} message is not an envelope"
        ) from exc
    marker = f"c28-probe-{uuid.uuid4().hex}"
    cid = probe_correlation_id()
    eid = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.UTC)
    envelope.update(
        payload={"c28_probe_marker": marker},
        envelope_id=eid,
        correlation_id=cid,
        envelope_timestamp=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    headers = {h["key"]: h["value"] for h in message.get("headers") or [] if "key" in h}
    headers.update(message_id=eid, correlation_id=cid, timestamp=now.isoformat())
    args = ["topic", "produce", APPLIED_TOPIC, "-z", "none"]
    for key, value in headers.items():
        args += ["-H", f"{key}:{value}"]
    out = lane.rpk(*args, stdin=json.dumps(envelope, separators=(",", ":")) + "\n")
    m = re.search(r"at offset (\d+)", out)
    return {
        "marker": marker,
        "correlation_id": cid,
        "published_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "offset": int(m.group(1)) if m else None,
        "envelope_copied_from_offset": message.get("offset"),
    }


def observe_injection(
    lane: ConsumerFlowLane,
    inj: TypedDictConsumerFlowInjection,
    dlq_before: int,
    wait_seconds: float,
) -> TypedDictConsumerFlowInjection:
    deadline = lane.monotonic() + wait_seconds
    since = inj["published_at"]
    # A second of slack before the publish, so a line stamped in the same second counts.
    since_dt = datetime.datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.UTC
    ) - datetime.timedelta(seconds=1)
    since_arg = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    while True:
        errors = boundaries = 0
        for c in RUNTIME_CONTAINERS:
            lines = lane.logs(c, since=since_arg)
            errors += sum(1 for ln in lines if VALIDATION_ERROR_RE.search(ln))
            boundaries += sum(
                1
                for ln in lines
                if (m := BOUNDARY_RE.search(ln))
                and m.group("cid") == inj["correlation_id"]
            )
        dlq_after = lane.high_watermark(GENERIC_DLQ)
        copies = 0
        if dlq_after > dlq_before:
            out = lane.rpk(
                "topic",
                "consume",
                GENERIC_DLQ,
                "-o",
                f"{dlq_before}:{dlq_after}",
                "-f",
                "%o\\t%v\\n",
                timeout=120,
            )
            copies = sum(1 for ln in out.splitlines() if inj["marker"] in ln)
        if (errors and boundaries and copies) or lane.monotonic() >= deadline:
            return {
                **inj,
                "validation_errors_after": errors,
                "boundary_lines": boundaries,
                "dlq_copies": copies,
                "generic_dlq_before": dlq_before,
                "generic_dlq_after": dlq_after,
            }
        lane.sleep(10)


def observe_lane(
    lane: ConsumerFlowLane,
    *,
    samples: int,
    interval: float,
    settle_seconds: float,
    injection_wait: float,
) -> TypedDictConsumerFlowCollected:
    ident_before = lane.wait_settled(settle_seconds)

    natural: dict[str, TypedDictConsumerFlowNatural] = {}
    for c in RUNTIME_CONTAINERS:
        lines = lane.logs(c)
        total, probe, nat = natural_validation_errors(lines)
        natural[c] = {
            "lines": len(lines),
            "total": total,
            "probe": probe,
            "natural": nat,
            "first_line": lines[0][:120] if lines else None,
        }

    hwm_before = lane.high_watermark(APPLIED_TOPIC)
    t0 = lane.monotonic()
    kind_samples: list[list[TypedDictConsumerFlowRow]] = []
    kind_view: list[dict[str, list[TypedDictConsumerFlowRow]]] = []
    for i in range(samples):
        if i:
            lane.sleep(interval)
        rows = walk(lane)["rows"]
        kind_samples.append(rows)
        kind_view.append({k: kind_rows(rows, p, t) for k, p, t, _c in KINDS})
    # At least a minute on the applied topic, as the readback took it.
    remaining = 60.0 - (lane.monotonic() - t0)
    if remaining > 0:
        lane.sleep(remaining)
    hwm_after = lane.high_watermark(APPLIED_TOPIC)
    applied_window = round(lane.monotonic() - t0)

    live = lane.live_groups()
    w = walk(lane)
    control_page1 = lane.page({})
    control_cursor = (
        lane.page({"cursor": str(control_page1["next_cursor"])})
        if control_page1.get("next_cursor")
        else None
    )

    dlq_before = lane.high_watermark(GENERIC_DLQ)
    seam_dlq_before = lane.high_watermark(SEAM_DLQ)
    inj = observe_injection(lane, inject(lane), dlq_before, injection_wait)
    seam_dlq_after = lane.high_watermark(SEAM_DLQ)

    ident_after = lane.identity()
    changed = {
        n: (ident_before[n]["id"], ident_after.get(n, {}).get("id"))
        for n in BOOT_CONTAINERS
        if ident_before[n]["id"] != ident_after.get(n, {}).get("id")
        or ident_before[n]["started_at"] != ident_after.get(n, {}).get("started_at")
    }
    if changed:
        raise ConsumerFlowBootChangedError(
            f"lane containers replaced during the run: {changed}"
        )

    walked_rows = w["rows"]
    return {
        "boot_identity": ident_before,
        "kinds": {"samples": kind_samples, "view": kind_view},
        "cursor": {
            "pages": w["pages"],
            "terminated": w["terminated"],
            "second_page_differs": w["second_page_differs"],
            "rows": len(walked_rows),
            "walked_groups": sorted(
                {str(r.get("consumer_group")) for r in walked_rows}
            ),
            "walked_pairs": len(
                {(r.get("consumer_group"), r.get("topic")) for r in walked_rows}
            ),
            "live_groups": live,
            "undeclared_cursor_control": None
            if control_cursor is None
            else {
                "returns_page_1_again": control_cursor["rows"][:3]
                == control_page1["rows"][:3],
            },
        },
        "boot": {
            "applied_hwm_before": hwm_before,
            "applied_hwm_after": hwm_after,
            "applied_window_seconds": applied_window,
            "natural": natural,
            "injection": inj,
            "seam_dlq_hwm": [seam_dlq_before, seam_dlq_after],
        },
    }
