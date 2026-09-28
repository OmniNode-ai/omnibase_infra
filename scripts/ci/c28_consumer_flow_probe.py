# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19812 -- the C28 producer: consumer-flow observability, re-measured.

WHAT THIS IS
    Beta criterion C28 reads "Consumer-flow observability on the runtime:
    payload validation failures on the consumer-flow seam are observed rather
    than discarded, every subscription kind registers a flow counter, and the
    flow exposure serves every live consumer group by paging with a cursor".
    Until this producer existed it rested on one hand READBACK, taken on the
    .201 dev lane at 2026-09-19T12:20Z and never re-taken. This script takes
    every clause of that readback again, on the lane as it is deployed now,
    and its EXIT CODE is the verdict -- the C11 / C12 / C15 shape, so the board
    reads a workflow conclusion and needs no second grader.

    ``0``  every check held
    ``1``  at least one check failed; the record names each one and why
    ``2``  the probe could not look (a lane container not running or not
           healthy, the boot changed under the run, docker, the broker, the
           database or the exposure refused). Deliberately distinct from
           ``1``: "I could not look" is not "the criterion is broken". Both
           fail the job.

WHAT IS MEASURED, CLAUSE BY CLAUSE -- the readback's own clauses
    kinds    CLAUSE 1. The consumer-flow exposure, walked in full each time, is
             sampled ``--samples`` times ``--sample-interval`` apart. The two
             subscription kinds the proof sentence names by hand must each
             have an INSTRUMENTED row in every sample: the audit-purpose
             projection consumer (``node_gateway_link_health_projection_compute``,
             whose contract declares ``consumer_purpose: audit`` -- the kind
             that produced no row at all when OMN-17214 was filed) and the
             publishing reducer (``projection_consumer_flow``). Instrumented
             means integer in/out counters and a state other than UNKNOWN. The
             window must ADVANCE across the samples, so a row retained from
             an earlier boot cannot pass, and each counter the kind is about
             must move: the audit consumer's ``messages_in`` and the
             reducer's ``messages_out`` (reducer publishes attributed, the
             half of OMN-17214 that read 76 of 78 rows as falsely STALLED).
             Consumer groups are matched on the node name up to ``consume.``,
             not on the trailing contract version, so a version bump does not
             read as a missing kind; the matched groups are recorded.

    negative CLAUSE 2. ``tests/ci/test_omn17214_subscription_flow_counter_gate.py``
             and ``tests/unit/runtime/auto_wiring/test_omn17214_raw_projection_flow_seam.py``
             must pass on this checkout, and must BITE, symmetrically and
             branch-specifically: with ``flow_counters.register(...)`` removed
             from ``_make_event_bus_callback`` only the ``[event_bus]``
             parametrized cases (and the AST gate) fail, and with it removed
             from ``_make_raw_event_projection_callback`` only the
             ``[raw_event_projection]`` ones do. The other branch's cases must
             still PASS under each mutation, which is what shows the mutated
             module imported and ran rather than erroring everything. The
             source is restored byte for byte after each mutation and the
             restore is checked by digest.

    cursor   CLAUSE 3. The live window is read first, straight from
             ``omnidash_analytics.omninode_internal.consumer_flow_windows``
             (distinct groups with a window in the last 10 minutes), then the
             exposure is walked with its DECLARED cursor parameter, ``since``,
             until ``next_cursor`` is null. The walk must terminate, a
             truncated page must carry a cursor, and no live group may be
             unreachable through the exposure. The order matters: a group the
             database has is already in the snapshot the exposure serves, so
             reading the database first cannot manufacture a miss. A superset
             is legitimate (the snapshot retains a pair whose last window
             predates the 10-minute cut) and is recorded, not failed. The
             readback's negative control -- the undeclared ``?cursor=``
             returning page 1 again -- is re-taken and recorded.

    boot     THE BOOT PREMISE clause 3 is taken across. At least one applied
             event: ``onex.evt.omnimarket.projection-consumer-flow-applied.v1``
             must advance on its own over the sampling window, measured before
             this probe publishes anything. Zero stall-alert payload-validation
             errors in the runtime containers' logs for this boot, counting
             BOTH model names the seam has had (``...Request``, the name the
             OMN-16875 AC2 grep used, and ``...Trigger``, the declared input
             model since OMN-16778), and excluding only errors this probe
             itself caused on an earlier run (see below). A zero and a dead
             probe read alike, so the zero is then PROVEN LIVE: one malformed
             payload is published to the trigger topic, its envelope copied
             from the newest real message so that only the payload contract is
             under test, and the seam must OBSERVE it -- a stall-alert
             validation error after the publish, a ``boundary_swallow_prevented
             dlq_routed=true`` line carrying the injected correlation id, and
             the injected marker present on the generic DLQ.

    Every run records the containers' ids and start times before and after.
    If the boot changed under the run the clauses were not taken across one
    boot, and the run is repeated once and then reported as could-not-look.

THE ONE WRITE
    The malformed payload is the only thing this probe writes to the lane,
    exactly as the readback's was. Its correlation id carries the fixed prefix
    ``c28c28c2-c28c-`` so a later run on the same boot can tell the errors this
    probe caused from a natural one. Everything else is ``docker inspect``,
    ``docker logs``, ``rpk topic describe/consume``, a ``psql`` select and HTTP
    GETs. The broker credential is read by ``rpk`` inside the broker container
    from that container's own environment; it never reaches this process.

WHAT IS RECORDED, NOT GRADED
    Two findings the readback surfaced and put on OMN-16875: refusals go to the
    generic infra DLQ while the per-seam DLQ the effect's contract declares
    stays at high-watermark 0, and one publish produces several DLQ copies.
    The per-seam high-watermark and the copy count are in every record. Grading
    them would be a new criterion, not a re-measurement of this one.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ---- the subject, as the readback named it ---------------------------------
EXPOSURE_TOPIC: Final[str] = "onex.snapshot.projection.consumer-flow.v1"
DEFAULT_BASE_URL: Final[str] = "http://host.docker.internal:3002"
LIVE_WINDOW_MINUTES: Final[int] = 10
LIVE_WINDOW_SQL: Final[str] = (
    "select distinct consumer_group from omninode_internal.consumer_flow_windows "
    "where window_start > now() - interval '10 minutes'"
)

#: (kind, consumer-group prefix, subscribed topic, the counter that must move)
KINDS: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "audit_projection_consumer",
        "local.omnibase_infra.node_gateway_link_health_projection_compute.consume.",
        "onex.evt.omnibase-infra.gateway-heartbeat.v1",
        "messages_in",
    ),
    (
        "publishing_reducer",
        "local.omnimarket.projection_consumer_flow.consume.",
        "onex.evt.platform.node-heartbeat.v1",
        "messages_out",
    ),
)

APPLIED_TOPIC: Final[str] = "onex.evt.omnimarket.projection-consumer-flow-applied.v1"
GENERIC_DLQ: Final[str] = "onex.dlq.omnibase-infra.events.v1"
SEAM_DLQ: Final[str] = "onex.dlq.omnimarket.consumer-flow-stall-alert-malformed.v1"

RUNTIME_CONTAINERS: Final[tuple[str, ...]] = (
    "omninode-runtime",
    "omninode-runtime-effects",
)
BOOT_CONTAINERS: Final[tuple[str, ...]] = (
    *RUNTIME_CONTAINERS,
    "omnimarket-projection-api",
)
BROKER_CONTAINER: Final[str] = "omnibase-infra-redpanda"
PG_CONTAINER: Final[str] = "omnibase-infra-postgres"
ANALYTICS_DB: Final[str] = "omnidash_analytics"

#: Both names the stall-alert seam's input model has carried.
VALIDATION_ERROR_RE: Final[re.Pattern[str]] = re.compile(
    r"validation errors? for ModelConsumerFlowStallAlert(?:Request|Trigger)\b"
)
BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(
    r"metric_name=boundary_swallow_prevented dlq_routed=true .*?"
    r"topic=" + re.escape(APPLIED_TOPIC) + r"\b.*?correlation_id=(?P<cid>[0-9a-f-]{36})"
)
#: How far after a validation error its boundary line may sit in the same log.
PAIRING_LOOKAHEAD: Final[int] = 60
PROBE_CID_PREFIX: Final[str] = "c28c28c2-c28c-"

# ---- clause 2 ---------------------------------------------------------------
NEGATIVE_TESTS: Final[tuple[str, ...]] = (
    "tests/ci/test_omn17214_subscription_flow_counter_gate.py",
    "tests/unit/runtime/auto_wiring/test_omn17214_raw_projection_flow_seam.py",
)
WIRING_MODULE: Final[str] = "src/omnibase_infra/runtime/auto_wiring/handler_wiring.py"
#: parametrize id -> the factory whose registration the mutation removes
BRANCHES: Final[dict[str, str]] = {
    "event_bus": "_make_event_bus_callback",
    "raw_event_projection": "_make_raw_event_projection_callback",
}
AST_GATE_TEST: Final[str] = (
    "test_every_selected_subscription_factory_registers_a_flow_counter"
)


class ProbeInputError(RuntimeError):
    """The probe could not look. Exit 2, never exit 1."""


class BootChangedError(ProbeInputError):
    """A lane container was replaced while the run was measuring."""


@dataclass(frozen=True)
class Check:
    name: str
    clause: str
    ok: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "clause": self.clause,
            "ok": self.ok,
            "evidence": self.evidence,
        }


@dataclass
class Record:
    checks: list[Check] = field(default_factory=list)

    def add(self, clause: str, name: str, ok: bool, evidence: str) -> None:
        self.checks.append(
            Check(name=name, clause=clause, ok=bool(ok), evidence=evidence)
        )

    @property
    def failures(self) -> list[str]:
        return [f"{c.clause}/{c.name}: {c.evidence}" for c in self.checks if not c.ok]

    @property
    def verdict(self) -> str:
        return "pass" if self.checks and not self.failures else "fail"

    @property
    def exit_code(self) -> int:
        return EXIT_OK if self.verdict == "pass" else EXIT_FINDINGS

    @property
    def detail(self) -> str:
        if self.verdict == "pass":
            return (
                f"all {len(self.checks)} checks held: both named subscription kinds "
                "carry live instrumented rows, the OMN-17214 negative tests pass and "
                "bite per branch, the since-cursor walk reaches every live consumer "
                "group, and across one boot with applied events the seam had no "
                "natural validation error and observed the injected one"
            )
        return f"{len(self.failures)} of {len(self.checks)} checks failed"


# =============================================================================
# Grading -- pure. Every expectation is declared in this file.
# =============================================================================


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def kind_rows(
    rows: Sequence[dict[str, Any]], prefix: str, topic: str
) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if isinstance(r, dict)
        and str(r.get("consumer_group", "")).startswith(prefix)
        and r.get("topic") == topic
    ]


def grade_kinds(rec: Record, samples: Sequence[Sequence[dict[str, Any]]]) -> None:
    """CLAUSE 1 over ``samples``: each is every row one full walk returned."""
    rec.add(
        "kinds",
        "sampled_more_than_once",
        len(samples) >= 2,
        f"{len(samples)} sample(s); one sample cannot show a window advancing",
    )
    for kind, prefix, topic, counter in KINDS:
        per_sample = [kind_rows(s, prefix, topic) for s in samples]
        present = [bool(rows) for rows in per_sample]
        rec.add(
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
        rec.add(
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
        rec.add(
            "kinds",
            f"{kind}_window_advances",
            len(starts) >= 2,
            f"{len(starts)} distinct window_start value(s) across the samples: {starts[:6]}",
        )
        moved: list[int] = [r[counter] for r in flat if _is_int(r.get(counter))]
        rec.add(
            "kinds",
            f"{kind}_{counter}_counted",
            any(v > 0 for v in moved),
            f"{counter} per row across the samples: {moved[:12]}",
        )


def junit_outcomes(xml_text: str) -> dict[str, str]:
    """Map ``name`` (with its ``[param]`` id) to passed / failed / error / skipped."""
    try:
        root = ET.fromstring(xml_text)  # noqa: S314 — JUnit XML this job's own pytest wrote
    except ET.ParseError as exc:
        raise ProbeInputError(f"pytest wrote unparseable junit xml: {exc}") from exc
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


def _param(name: str) -> str | None:
    m = re.search(r"\[([^\]]+)\]$", name)
    return m.group(1) if m else None


def grade_negative(rec: Record, neg: dict[str, Any]) -> None:
    """CLAUSE 2 over the clean run and one run per mutated branch."""
    clean = neg.get("clean") or {}
    outcomes: dict[str, str] = clean.get("outcomes") or {}
    not_passed = sorted(n for n, o in outcomes.items() if o != "passed")
    rec.add(
        "negative",
        "tests_pass_on_this_checkout",
        bool(outcomes) and not not_passed and clean.get("returncode") == 0,
        f"{len(outcomes)} collected, not passed: {not_passed}, pytest exit {clean.get('returncode')}",
    )
    params = {_param(n) for n in outcomes}
    missing_branches = sorted(b for b in BRANCHES if b not in params)
    rec.add(
        "negative",
        "both_branches_are_parametrized",
        not missing_branches,
        f"parametrized ids seen {sorted(p for p in params if p)}; missing {missing_branches}",
    )
    for branch in BRANCHES:
        run = (neg.get("mutations") or {}).get(branch) or {}
        mo: dict[str, str] = run.get("outcomes") or {}
        failed = sorted(n for n, o in mo.items() if o in ("failed", "error"))
        own = [n for n in failed if _param(n) == branch]
        foreign = [n for n in failed if _param(n) not in (None, branch)]
        others_passing = [
            n
            for n, o in mo.items()
            if _param(n) not in (None, branch) and o == "passed"
        ]
        rec.add(
            "negative",
            f"mutation_{branch}_bites_its_own_branch",
            bool(run.get("applied")) and bool(own),
            f"removed the registration in {BRANCHES[branch]} (applied={run.get('applied')}); "
            f"failing [{branch}] cases: {own}",
        )
        rec.add(
            "negative",
            f"mutation_{branch}_is_branch_specific",
            bool(run.get("applied")) and not foreign and bool(others_passing),
            f"failing cases of the other branch: {foreign}; other-branch cases still "
            f"passing: {len(others_passing)} (zero would mean the module did not run)",
        )
        rec.add(
            "negative",
            f"mutation_{branch}_fails_the_ast_gate",
            mo.get(AST_GATE_TEST) in ("failed", "error"),
            f"{AST_GATE_TEST}: {mo.get(AST_GATE_TEST)}",
        )
        rec.add(
            "negative",
            f"mutation_{branch}_restored",
            bool(run.get("restored")),
            f"{WIRING_MODULE} digest after restore equals the original: {run.get('restored')}",
        )


def grade_cursor(rec: Record, cur: dict[str, Any]) -> None:
    """CLAUSE 3 over one full walk and the live window read before it."""
    pages: list[dict[str, Any]] = cur.get("pages") or []
    rec.add(
        "cursor",
        "walk_terminates",
        bool(pages) and bool(cur.get("terminated")),
        f"{len(pages)} page(s); last next_cursor "
        f"{pages[-1].get('next_cursor') if pages else None!r}; terminated={cur.get('terminated')}",
    )
    silent_truncation = [
        i
        for i, p in enumerate(pages)
        if _is_int(p.get("row_count"))
        and _is_int(p.get("row_limit"))
        and p["row_count"] >= p["row_limit"]
        and not p.get("next_cursor")
    ]
    rec.add(
        "cursor",
        "truncated_pages_carry_a_cursor",
        bool(pages) and not silent_truncation,
        f"pages at row_limit with a null next_cursor: {silent_truncation}",
    )
    advancing = cur.get("second_page_differs")
    rec.add(
        "cursor",
        "since_advances",
        bool(pages) and (len(pages) < 2 or advancing is True),
        "single page, nothing to advance past"
        if len(pages) < 2
        else f"page 2 via since= differs from page 1: {advancing}",
    )
    live = set(cur.get("live_groups") or [])
    walked = set(cur.get("walked_groups") or [])
    rec.add(
        "cursor",
        "live_window_is_non_empty",
        bool(live),
        f"{len(live)} distinct consumer groups in the last {LIVE_WINDOW_MINUTES} minutes "
        "(an empty window would make the comparison vacuous)",
    )
    unreachable = sorted(live - walked)
    rec.add(
        "cursor",
        "every_live_group_reachable",
        bool(live) and not unreachable,
        f"live {len(live)}, walked {len(walked)}, unreachable {len(unreachable)}: {unreachable[:10]}",
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


def grade_boot(rec: Record, boot: dict[str, Any]) -> None:
    before, after = boot.get("applied_hwm_before"), boot.get("applied_hwm_after")
    rec.add(
        "boot",
        "applied_event_on_this_boot",
        isinstance(before, int) and isinstance(after, int) and after > before,
        f"{APPLIED_TOPIC} high-watermark {before} -> {after} over "
        f"{boot.get('applied_window_seconds')}s, before this probe published anything",
    )
    logs: dict[str, dict[str, Any]] = boot.get("natural") or {}
    rec.add(
        "boot",
        "runtime_logs_read",
        set(logs) == set(RUNTIME_CONTAINERS)
        and all(_is_int(v.get("lines")) and v["lines"] > 0 for v in logs.values()),
        "log lines read per container: "
        + ", ".join(f"{c}={v.get('lines')}" for c, v in sorted(logs.items())),
    )
    natural = {c: v.get("natural") for c, v in sorted(logs.items())}
    rec.add(
        "boot",
        "zero_natural_stall_alert_validation_errors",
        bool(logs) and all(v == 0 for v in natural.values()),
        f"natural errors per container {natural} (total, this probe's: "
        + ", ".join(
            f"{c}={v.get('total')}/{v.get('probe')}" for c, v in sorted(logs.items())
        )
        + ")",
    )
    inj = boot.get("injection") or {}
    rec.add(
        "boot",
        "injected_payload_published",
        _is_int(inj.get("offset")),
        f"offset {inj.get('offset')} on {APPLIED_TOPIC}, correlation {inj.get('correlation_id')}",
    )
    rec.add(
        "boot",
        "seam_raised_a_validation_error_after_the_publish",
        _is_int(inj.get("validation_errors_after"))
        and inj["validation_errors_after"] > 0,
        f"stall-alert validation errors after the publish: {inj.get('validation_errors_after')} "
        "(the counter that must read 0 above is shown able to read non-zero)",
    )
    rec.add(
        "boot",
        "seam_dead_lettered_the_injected_correlation",
        _is_int(inj.get("boundary_lines")) and inj["boundary_lines"] > 0,
        f"boundary_swallow_prevented dlq_routed=true lines naming the injected "
        f"correlation id: {inj.get('boundary_lines')}",
    )
    rec.add(
        "boot",
        "injected_marker_durably_on_the_dlq",
        _is_int(inj.get("dlq_copies")) and inj["dlq_copies"] > 0,
        f"{GENERIC_DLQ} messages carrying the marker: {inj.get('dlq_copies')}",
    )


def grade(obs: dict[str, Any]) -> Record:
    rec = Record()
    grade_kinds(rec, obs.get("kinds", {}).get("samples") or [])
    grade_negative(rec, obs.get("negative") or {})
    grade_cursor(rec, obs.get("cursor") or {})
    grade_boot(rec, obs.get("boot") or {})
    return rec


# =============================================================================
# Collection -- the lane, through this host's docker socket and gateway alias.
# =============================================================================


def _run(
    args: Sequence[str], *, stdin: str | None = None, timeout: float = 120.0
) -> str:
    try:
        proc = subprocess.run(
            list(args),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeInputError(
            f"{args[0]} {args[1:3]} could not run: {type(exc).__name__}"
        ) from exc
    if proc.returncode != 0:
        raise ProbeInputError(
            f"{' '.join(args[:4])} exited {proc.returncode}: {proc.stderr.strip()[:400]}"
        )
    return proc.stdout


@dataclass
class Lane:
    docker: str = "docker"
    base_url: str = DEFAULT_BASE_URL
    http_timeout: float = 30.0

    # ---- containers -------------------------------------------------------
    def identity(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in BOOT_CONTAINERS:
            raw = _run([self.docker, "inspect", name], timeout=30)
            try:
                info = json.loads(raw)[0]
            except (ValueError, IndexError) as exc:
                raise ProbeInputError(f"docker inspect {name}: unreadable") from exc
            state = info.get("State") or {}
            out[name] = {
                "id": str(info.get("Id", ""))[:12],
                "started_at": state.get("StartedAt"),
                "status": state.get("Status"),
                "health": (state.get("Health") or {}).get("Status"),
                "image": (info.get("Config") or {}).get("Image"),
            }
        return out

    def wait_settled(
        self, settle_seconds: float, poll: float = 15.0
    ) -> dict[str, dict[str, Any]]:
        deadline = time.monotonic() + settle_seconds
        while True:
            ident = self.identity()
            unsettled = {
                n: (v["status"], v["health"])
                for n, v in ident.items()
                if v["status"] != "running" or v["health"] not in (None, "healthy")
            }
            if not unsettled:
                return ident
            if time.monotonic() >= deadline:
                raise ProbeInputError(
                    f"lane containers not running and healthy after {settle_seconds:.0f}s: {unsettled}"
                )
            time.sleep(poll)

    def logs(self, container: str, since: str | None = None) -> list[str]:
        args = [self.docker, "logs"]
        if since:
            args += ["--since", since]
        args.append(container)
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=180, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProbeInputError(
                f"docker logs {container}: {type(exc).__name__}"
            ) from exc
        if proc.returncode != 0:
            raise ProbeInputError(f"docker logs {container} exited {proc.returncode}")
        # The runtime logs to stderr and stdout both.
        return (proc.stdout + proc.stderr).splitlines()

    # ---- broker (credential stays inside the broker container) -----------
    def rpk(self, *args: str, stdin: str | None = None, timeout: float = 60.0) -> str:
        script = (
            'rpk "$@" -X user="$DEV_KAFKA_SASL_USERNAME" '
            '-X pass="$DEV_KAFKA_SASL_PASSWORD" -X sasl.mechanism=SCRAM-SHA-256'
        )
        return _run(
            [
                self.docker,
                "exec",
                "-i",
                BROKER_CONTAINER,
                "sh",
                "-c",
                script,
                "rpk",
                *args,
            ],
            stdin=stdin,
            timeout=timeout,
        )

    def high_watermark(self, topic: str) -> int:
        return parse_high_watermark(self.rpk("topic", "describe", "-p", topic))

    # ---- database ---------------------------------------------------------
    def live_groups(self) -> list[str]:
        sql = LIVE_WINDOW_SQL
        script = f'psql -U "$POSTGRES_USER" -d {ANALYTICS_DB} -AtX -v ON_ERROR_STOP=1 -c "{sql}"'
        out = _run([self.docker, "exec", PG_CONTAINER, "sh", "-c", script], timeout=120)
        return sorted({line.strip() for line in out.splitlines() if line.strip()})

    # ---- exposure ---------------------------------------------------------
    def page(self, query: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/projection/{EXPOSURE_TOPIC}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        try:
            with urllib.request.urlopen(url, timeout=self.http_timeout) as resp:  # noqa: S310
                body = json.load(resp)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ProbeInputError(f"GET {url}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(body, dict) or not isinstance(body.get("rows"), list):
            raise ProbeInputError(f"GET {url}: no rows list in the response")
        return body


def parse_high_watermark(describe_out: str) -> int:
    """Sum of HIGH-WATERMARK over the partitions table of ``rpk topic describe -p``."""
    col: int | None = None
    total = 0
    seen = False
    for line in describe_out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if "HIGH-WATERMARK" in parts:
            col = parts.index("HIGH-WATERMARK")
            continue
        if col is not None and parts[0].isdigit() and len(parts) > col:
            total += int(parts[col])
            seen = True
    if not seen:
        raise ProbeInputError(
            f"no partition rows in rpk describe output: {describe_out[:200]!r}"
        )
    return total


def walk(lane: Lane, max_pages: int = 200) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    first_rows: list[Any] = []
    second_rows: list[Any] = []
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
    repo: Path, pytest_cmd: Sequence[str], scratch: Path
) -> dict[str, Any]:
    def run(tag: str) -> dict[str, Any]:
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
            proc = subprocess.run(
                cmd, cwd=repo, capture_output=True, text=True, timeout=900, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProbeInputError(
                f"pytest ({tag}) could not run: {type(exc).__name__}"
            ) from exc
        if not xml.exists():
            raise ProbeInputError(
                f"pytest ({tag}) wrote no junit xml, exit {proc.returncode}: {proc.stdout[-400:]}"
            )
        return {
            "returncode": proc.returncode,
            "outcomes": junit_outcomes(xml.read_text("utf-8")),
        }

    target = repo / WIRING_MODULE
    original = target.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    result: dict[str, Any] = {
        "clean": run("clean"),
        "mutations": {},
        "module_sha256": digest,
    }
    for branch, factory in BRANCHES.items():
        mutated = _mutated_source(original.decode("utf-8"), factory)
        entry: dict[str, Any] = {"factory": factory, "applied": mutated is not None}
        if mutated is not None:
            try:
                target.write_text(mutated, encoding="utf-8")
                entry.update(run(branch))
            finally:
                target.write_bytes(original)
        entry["restored"] = hashlib.sha256(target.read_bytes()).hexdigest() == digest
        result["mutations"][branch] = entry
    return result


def probe_correlation_id() -> str:
    return str(uuid.UUID(hex="c28c28c2c28c" + uuid.uuid4().hex[12:]))


def inject(lane: Lane) -> dict[str, Any]:
    """Publish one malformed trigger, envelope copied from the newest real one."""
    newest = lane.rpk("topic", "consume", APPLIED_TOPIC, "-o", "-1", "-n", "1")
    try:
        message = json.loads(newest)
        envelope = json.loads(message["value"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ProbeInputError(
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
    lane: Lane, inj: dict[str, Any], dlq_before: int, wait_seconds: float
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
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
        if (errors and boundaries and copies) or time.monotonic() >= deadline:
            return {
                **inj,
                "validation_errors_after": errors,
                "boundary_lines": boundaries,
                "dlq_copies": copies,
                "generic_dlq_before": dlq_before,
                "generic_dlq_after": dlq_after,
            }
        time.sleep(10)


def observe_lane(
    lane: Lane,
    *,
    samples: int,
    interval: float,
    settle_seconds: float,
    injection_wait: float,
) -> dict[str, Any]:
    ident_before = lane.wait_settled(settle_seconds)

    natural: dict[str, dict[str, Any]] = {}
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
    t0 = time.monotonic()
    kind_samples: list[list[dict[str, Any]]] = []
    kind_view: list[dict[str, Any]] = []
    for i in range(samples):
        if i:
            time.sleep(interval)
        rows = walk(lane)["rows"]
        kind_samples.append(rows)
        kind_view.append({k: kind_rows(rows, p, t) for k, p, t, _c in KINDS})
    # At least a minute on the applied topic, as the readback took it.
    remaining = 60.0 - (time.monotonic() - t0)
    if remaining > 0:
        time.sleep(remaining)
    hwm_after = lane.high_watermark(APPLIED_TOPIC)
    applied_window = round(time.monotonic() - t0)

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
        raise BootChangedError(f"lane containers replaced during the run: {changed}")

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


# =============================================================================
# Record and summary
# =============================================================================


def _observed_summary(obs: dict[str, Any]) -> dict[str, Any]:
    cur = obs.get("cursor") or {}
    live, walked = (
        set(cur.get("live_groups") or []),
        set(cur.get("walked_groups") or []),
    )
    return {
        "boot_identity": obs.get("boot_identity"),
        "kinds": (obs.get("kinds") or {}).get("view"),
        "negative": {
            "module_sha256": (obs.get("negative") or {}).get("module_sha256"),
            "clean": (obs.get("negative") or {}).get("clean"),
            "mutations": (obs.get("negative") or {}).get("mutations"),
        },
        "cursor": {
            "pages": cur.get("pages"),
            "rows": cur.get("rows"),
            "walked_groups": len(walked),
            "walked_pairs": cur.get("walked_pairs"),
            "live_groups": len(live),
            "unreachable": sorted(live - walked),
            "walked_not_live": len(walked - live),
            "undeclared_cursor_control": cur.get("undeclared_cursor_control"),
        },
        "boot": obs.get("boot"),
    }


def _render_summary(rec: Record, *, as_of: str) -> str:
    lines = [
        "## C28 -- consumer-flow observability on the dev lane (OMN-19812)",
        "",
        f"**Verdict: `{rec.verdict}`** at `{as_of}`.",
        "",
        rec.detail,
        "",
        "| Clause | Check | Pass |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {c.clause} | {c.name} | {'yes' if c.ok else '**no**'} |" for c in rec.checks
    ]
    if rec.failures:
        lines += ["", "Failures:", ""] + [f"- `{f}`" for f in rec.failures]
    lines.append("")
    return "\n".join(lines)


def _write_record(path: str, payload: dict[str, Any]) -> None:
    if path:
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", "utf-8"
        )


def main(
    argv: list[str] | None = None, *, lane_factory: Callable[..., Lane] = Lane
) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--sample-interval", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=300.0)
    parser.add_argument("--injection-wait", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--pytest-cmd",
        default="uv run --frozen pytest",
        help="How to invoke pytest in this checkout.",
    )
    parser.add_argument("--scratch", default=".")
    parser.add_argument(
        "--replay", default="", help="Grade a recorded observation instead of the lane."
    )
    parser.add_argument("--record", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)

    as_of = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "version": RECORD_VERSION,
        "criterion": "C28",
        "ticket": "OMN-19812",
        "as_of": as_of,
    }
    try:
        if args.replay:
            try:
                obs = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            except ValueError as exc:
                raise ProbeInputError(
                    f"replay file {args.replay!r} is not JSON"
                ) from exc
            if not isinstance(obs, dict):
                raise ProbeInputError("replay file is not a JSON object")
        else:
            lane = lane_factory(docker=args.docker_bin, base_url=args.base_url)
            last: ProbeInputError | None = None
            obs = {}
            for attempt in range(1, max(1, args.attempts) + 1):
                try:
                    obs = observe_lane(
                        lane,
                        samples=args.samples,
                        interval=args.sample_interval,
                        settle_seconds=args.settle_seconds,
                        injection_wait=args.injection_wait,
                    )
                    last = None
                    break
                except BootChangedError as exc:
                    print(f"::warning::attempt {attempt}: {exc}", file=sys.stderr)
                    last = exc
            if last is not None:
                raise last
            obs["negative"] = run_negative(
                REPO_ROOT, shlex.split(args.pytest_cmd), Path(args.scratch)
            )
    except (ProbeInputError, OSError) as exc:
        print(f"::error::C28 probe could not look: {exc}", file=sys.stderr)
        _write_record(
            args.record, {**base, "verdict": "could_not_look", "detail": str(exc)}
        )
        return EXIT_INPUT

    rec = grade(obs)
    _write_record(
        args.record,
        {
            **base,
            "verdict": rec.verdict,
            "detail": rec.detail,
            "checks": [c.to_dict() for c in rec.checks],
            "failures": rec.failures,
            "observed": _observed_summary(obs),
        },
    )
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(_render_summary(rec, as_of=as_of))
    for c in rec.checks:
        print(f"{c.clause:<9} {c.name:<58} pass={c.ok}")
    if rec.failures:
        for f in rec.failures:
            print(f"::error::{f}", file=sys.stderr)
        print(f"::error::C28 RED -- {rec.detail}", file=sys.stderr)
    else:
        print(f"C28 GREEN -- {rec.detail}")
    return rec.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
