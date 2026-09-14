# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Measure the closer's held-unbound residual (OMN-18329).

Point-in-time diagnostic. Answers two questions the 2026-09-13 mechanical
ticket closeout plan (`knowledge-base-internal` `beta/plans/2026-09-13-
mechanical-ticket-closeout-plan.md`, section 13 step 1) needs a real number
for before any commitment is made about the legacy held corpus:

1. How many DISTINCT Linear tickets are currently held by the evidence
   autoclose sweep (`node_evidence_autoclose_sweep_effect`,
   `.github/workflows/evidence-autoclose-sweep.yml`) on an unbound
   acceptance criterion (``gap_ac_unbound``)? The only number that existed
   before this ticket was 213 *hold events* across overlapping 30-minute
   sweep runs — not a ticket count.
2. How many change-control contracts (`onex_change_control` ``contracts/
   OMN-*.yaml``) declare a criterion binding (``binds_ac``) at all, and how
   many of those additionally carry an ACCEPTED binding record (``ac_bindings``
   entry with both ``accepted_by`` and ``accepted_at`` set)?

Design note — why "currently held" is not "last decision == gap_ac_unbound"
----------------------------------------------------------------------------
A ticket that stays held is not re-posted every 30-minute tick: on the run
that *first* reaches ``gap_ac_unbound`` the sweep posts a comment; every
subsequent run that reaches the identical verdict fingerprint records
``skipped_duplicate_comment`` instead ("this exact verdict is already posted
... not repeating it", `handler_evidence_autoclose_sweep.py`
``_emit_gap_comment``) so it does not spam the ticket. So a ticket's
chronologically LAST outcome row over any window that is long enough to see
a repeat tick is ``skipped_duplicate_comment``, not ``gap_ac_unbound`` —
reading the bare ``decision`` field of the last row would misclassify every
persistently-held ticket as an administrative skip and silently undercount
the very population this ticket exists to measure.

The fix costs nothing extra to read: the sweep's own duplicate-comment
marker embeds the class it is a duplicate of
(``_sweep_comment_marker`` — ``<!-- onex-autoclose-verdict vN
class=<decision> fingerprint=... -->``), and that marker is echoed verbatim
into the ``skipped_duplicate_comment`` row's own ``reason`` string
("... already posted on the ticket ({marker}) ..."). `resolve_decision_class`
recovers the underlying class from that string. This is read from the
receipt's own free-text field, not inferred — an unreadable marker resolves
to ``unknown_duplicate_class`` rather than being silently dropped from the
count (the safe direction: never let an unparseable row disappear from
either bucket).

Data source
-----------
The sweep's receipt (one ``ModelSkillResult``-wrapped
``ModelEvidenceAutocloseSweepResult``, keyed by the marker string
``"skill_name":"node_evidence_autoclose_sweep_effect"``) is printed to
stdout as ONE compact JSON line per run (`receipt_mode.py`::
``click.echo(receipt.model_dump_json())``) — never uploaded as a workflow
artifact, so it is recovered from the archived job log rather than from a
downloadable artifact. ``gh api repos/<repo>/actions/runs/<id>/logs``
returns that archive as a ZIP of per-job ``.txt`` files, each line
``<timestamp> <text>``; the receipt line is recovered by stripping the
leading ISO-8601 timestamp from each line and keeping the one that carries
the marker. **Measured on this ticket:** ``gh run view <id> --log`` — the
more obvious CLI path — returned a silent, exit-0, EMPTY transcript for
every run more than a few hours old in the 2026-08-30..2026-09-13 window,
while the same runs' ``.../logs`` API call returned real data; `fetch_run_log`
below uses the API endpoint for exactly this reason (see its docstring).
This mirrors the method the 2026-09-13
`reports/2026-09-13-closeout-process-infrastructure-report.md` closeout
report used to recover 696 of 699 runs' receipts in its own window.

Everything in this module that reads a `gh` process or the filesystem is
isolated in the ``main`` staging functions below; the counting logic is pure
and takes already-parsed data, so it is unit-testable with fixtures and
carries no network or subprocess dependency.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

# --- Constants ---------------------------------------------------------------

#: The class this measurement is counting AC1 against (OMN-18329 AC1).
GAP_AC_UNBOUND: str = "gap_ac_unbound"

#: The sweep's own skipped-repeat decision value (see module docstring).
SKIPPED_DUPLICATE_COMMENT: str = "skipped_duplicate_comment"

#: Resolved class recorded when a `skipped_duplicate_comment` row's `reason`
#: does not contain a readable marker. Never silently dropped.
_UNKNOWN_DUPLICATE_CLASS: str = "unknown_duplicate_class"

#: Pattern that identifies the sweep's own receipt line among every other
#: line in a `gh run view --log` transcript, which interleaves many
#: unrelated steps (dod_verify diagnostics, checkout, etc). Whitespace around
#: the colon is tolerated: the live receipt is pydantic-compact
#: (`model_dump_json()`, no space), but nothing about this module should
#: depend on that formatting detail holding forever.
_SWEEP_RECEIPT_MARKER_RE = re.compile(
    r'"skill_name"\s*:\s*"node_evidence_autoclose_sweep_effect"'
)

#: `gh run view --log` timestamps each line: `2026-09-13T18:21:46.8512124Z `.
_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s*")

#: Recovers the underlying decision class from a `skipped_duplicate_comment`
#: row's `reason` text, which embeds the sweep's own comment marker verbatim
#: (`_sweep_comment_marker` in `handler_evidence_autoclose_sweep.py`):
#: `<!-- onex-autoclose-verdict v1 class=gap_ac_unbound fingerprint=... -->`.
_DUPLICATE_CLASS_RE = re.compile(r"class=([a-z_]+)\s+fingerprint=")

#: OCC contract glob, relative to the onex_change_control checkout root.
CONTRACT_GLOB: str = "contracts/OMN-*.yaml"


# --- Part 1: distinct held-unbound tickets from sweep receipts --------------


@dataclass(frozen=True)
class TicketDecisionEvent:
    """One (run, ticket) decision row, with the duplicate-comment class resolved."""

    run_created_at: str
    ticket_id: str
    decision: str
    resolved_class: str
    reason: str


def extract_receipt_json_lines(log_text: str) -> list[str]:
    """Return every sweep-receipt JSON string found in a `gh run view --log` transcript.

    `gh run view --log` emits one line per log line across every job and
    step in the run, tab-separated as `<job>\\t<step>\\t<timestamp
    message>`. This takes the last tab-separated field (so it also tolerates
    a plain (untabbed) log source), strips the leading GH Actions timestamp,
    and keeps only lines that carry the sweep's own receipt marker. Ordinarily
    this returns exactly one line per run; zero means the run produced no
    parseable receipt (killed mid-run, kill-switch short-circuit before the
    print, or an expired/unavailable log — the "3 of 699" case the closeout
    report measured).
    """
    matches: list[str] = []
    for raw_line in log_text.splitlines():
        message = raw_line.rsplit("\t", 1)[-1] if "\t" in raw_line else raw_line
        message = _TIMESTAMP_PREFIX_RE.sub("", message, count=1)
        if _SWEEP_RECEIPT_MARKER_RE.search(message):
            matches.append(message)
    return matches


def parse_sweep_receipt(raw_json: str) -> Mapping[str, object]:
    """Parse one receipt JSON line. Raises `json.JSONDecodeError` on malformed input.

    Deliberately does not swallow a parse failure — a caller iterating many
    runs must be able to tell a parseable-but-empty receipt from an
    unparseable one instead of the two collapsing into the same silent zero.
    """
    parsed = json.loads(raw_json)
    if not isinstance(parsed, Mapping):
        raise ValueError(
            f"receipt JSON did not decode to an object: {raw_json[:200]!r}"
        )
    return parsed


def resolve_decision_class(decision: str, reason: str) -> str:
    """Resolve one outcome row's governing class.

    Every decision except `skipped_duplicate_comment` already IS its own
    class. A `skipped_duplicate_comment` row means "the sweep reached this
    same verdict before and already said so" (see module docstring); the
    class it duplicates is embedded in its own `reason` text via the sweep's
    comment marker. An unreadable marker resolves to
    `_UNKNOWN_DUPLICATE_CLASS` rather than being dropped, which is the safe
    direction — it can never inflate the held count, only ever appear as its
    own visible bucket an auditor can inspect.
    """
    if decision != SKIPPED_DUPLICATE_COMMENT:
        return decision
    match = _DUPLICATE_CLASS_RE.search(reason)
    if match is None:
        return _UNKNOWN_DUPLICATE_CLASS
    return match.group(1)


def iter_ticket_decision_events(
    run_created_at: str, receipt: Mapping[str, object]
) -> Iterator[TicketDecisionEvent]:
    """Yield one `TicketDecisionEvent` per (companion, ticket) row in one run's receipt.

    Silently skips a receipt with no readable `result.outcomes` list (an
    error-shaped receipt, e.g. a kill-switch short-circuit) rather than
    raising — a run that made no ticket decisions contributes no events,
    which is the correct contribution of zero.
    """
    result = receipt.get("result")
    if not isinstance(result, Mapping):
        return
    outcomes = result.get("outcomes")
    if not isinstance(outcomes, list):
        return
    for row in outcomes:
        if not isinstance(row, Mapping):
            continue
        ticket_id = str(row.get("ticket_id") or "").strip()
        if not ticket_id:
            continue
        decision = str(row.get("decision") or "")
        reason = str(row.get("reason") or "")
        yield TicketDecisionEvent(
            run_created_at=run_created_at,
            ticket_id=ticket_id,
            decision=decision,
            resolved_class=resolve_decision_class(decision, reason),
            reason=reason,
        )


def distinct_tickets_currently_held(
    events: Iterable[TicketDecisionEvent], target_class: str = GAP_AC_UNBOUND
) -> tuple[str, ...]:
    """Distinct ticket ids whose chronologically LAST decision resolves to `target_class`.

    "Last" is by `run_created_at` (ISO-8601, lexicographically ordered), the
    run's own timestamp rather than any ordering the log happened to be
    fetched in — so events may be handed in from runs processed in any
    order. Returns a sorted tuple (stable output, no dependency on dict
    iteration order) rather than a bare count so a positive control can
    assert a specific ticket id is present, not just that the count moved.
    """
    latest_by_ticket: dict[str, tuple[str, str]] = {}
    for event in events:
        prior = latest_by_ticket.get(event.ticket_id)
        if prior is None or event.run_created_at >= prior[0]:
            latest_by_ticket[event.ticket_id] = (
                event.run_created_at,
                event.resolved_class,
            )
    return tuple(
        sorted(
            ticket_id
            for ticket_id, (_created_at, resolved_class) in latest_by_ticket.items()
            if resolved_class == target_class
        )
    )


# --- Part 2: OCC contract binding / acceptance counts ------------------------


@dataclass(frozen=True)
class ContractBindingCounts:
    """Corpus-wide binds_ac / acceptance counts (OMN-18329 AC2)."""

    total_contracts: int
    contracts_with_evidence: int
    contracts_with_binding: int
    contracts_with_accepted_binding: int


def _dod_evidence_items(
    contract: Mapping[str, object],
) -> Sequence[Mapping[str, object]]:
    items = contract.get("dod_evidence")
    if not isinstance(items, list):
        return ()
    return tuple(item for item in items if isinstance(item, Mapping))


def _item_declares_binding(item: Mapping[str, object]) -> bool:
    """True if this dod_evidence item's `binds_ac` names at least one criterion."""
    raw = item.get("binds_ac")
    return isinstance(raw, list) and len(raw) > 0


def _item_has_accepted_binding(item: Mapping[str, object]) -> bool:
    """True if this item's `ac_bindings` carries a record with BOTH fields set.

    `ac_bindings` is the OCC-local per-criterion binding record (OMN-18236 —
    see `omnimarket` `node_dod_verify/services/evidence_collector.py`
    `_draft_binding_labels`): a record with no `accepted_by` is a draft a
    machine may propose; a person accepting it is what makes it discharge a
    criterion. OMN-18329 AC2 additionally requires `accepted_at` — both
    fields non-empty is "an acceptance", one alone is not.
    """
    raw = item.get("ac_bindings")
    if not isinstance(raw, list):
        return False
    for record in raw:
        if not isinstance(record, Mapping):
            continue
        accepted_by = str(record.get("accepted_by") or "").strip()
        accepted_at = str(record.get("accepted_at") or "").strip()
        if accepted_by and accepted_at:
            return True
    return False


def count_contract_bindings(
    contracts: Iterable[Mapping[str, object]],
) -> ContractBindingCounts:
    """Tally the corpus-wide binding/acceptance counts over already-parsed contracts.

    `contracts_with_evidence` is the AC3 positive control for this half of
    the measurement: a corpus that returns zero for `contracts_with_binding`
    or `contracts_with_accepted_binding` while `contracts_with_evidence` is
    also zero means the input was empty or unreadable, not that the corpus
    genuinely has no bindings.
    """
    total = 0
    with_evidence = 0
    with_binding = 0
    with_accepted = 0
    for contract in contracts:
        total += 1
        items = _dod_evidence_items(contract)
        if items:
            with_evidence += 1
        if any(_item_declares_binding(item) for item in items):
            with_binding += 1
        if any(_item_has_accepted_binding(item) for item in items):
            with_accepted += 1
    return ContractBindingCounts(
        total_contracts=total,
        contracts_with_evidence=with_evidence,
        contracts_with_binding=with_binding,
        contracts_with_accepted_binding=with_accepted,
    )


def load_contracts_from_directory(
    contracts_dir: Path,
) -> Iterator[Mapping[str, object]]:
    """Yield every parsed contract YAML under `contracts_dir` (non-recursive glob).

    Raises `yaml.YAMLError` on a malformed contract rather than skipping it —
    an unreadable contract silently excluded from the corpus is exactly the
    suppressed-failure shape CLAUDE.md rule 16 forbids for a verification
    sweep.
    """
    for path in sorted(contracts_dir.glob("OMN-*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            yield loaded


# --- Live data staging (gh CLI + filesystem; not unit-tested directly) ------


def fetch_sweep_run_ids(
    *, repo: str, workflow: str, limit: int, since_iso: str, until_iso: str = ""
) -> list[tuple[int, str]]:
    """Return `(run_id, created_at)` pairs for completed runs at/after `since_iso`.

    Uses `gh run list --created` with GitHub's own search-filter date syntax
    (`>=since_iso`, or `since_iso..until_iso` when `until_iso` is given)
    rather than fetching the most recent `limit` runs and filtering
    client-side — a bare `--limit` fetch with no date filter silently
    truncates before reaching the window boundary once the window holds more
    runs than `limit`, which reads as an undercount with no error. `limit` is
    still passed as the (generous) page-size ceiling `gh` itself enforces.

    Never suppresses stderr (CLAUDE.md rule 16): a `gh` failure propagates as
    a `CalledProcessError` rather than reading as a silent empty result.
    """
    created_filter = f"{since_iso}..{until_iso}" if until_iso else f">={since_iso}"
    proc = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            workflow,
            "--created",
            created_filter,
            "--limit",
            str(limit),
            "--json",
            "databaseId,createdAt,status,conclusion",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(proc.stdout)
    return [
        (int(row["databaseId"]), str(row["createdAt"]))
        for row in rows
        if row.get("status") == "completed"
    ]


def fetch_run_log(*, repo: str, run_id: int) -> str:
    """Return the concatenated archived job log text for one run.

    Uses `gh api repos/<repo>/actions/runs/<id>/logs` (the archived-log ZIP
    endpoint), NOT `gh run view --log`. Measured live on this ticket: `gh run
    view --log` returned a silent, exit-0, empty transcript for every run
    older than a few hours in the 2026-08-30..2026-09-13 window, while the
    same runs' `.../logs` API call returned real archived log data — `gh run
    view --log`'s own log-fetch path does not reliably reach the archive
    `gh api` reaches directly. Every `.txt` entry in the zip is concatenated
    (a run can have more than one job/step file); each line inside an entry
    is `<timestamp> <text>` with no job/step prefix, which
    `extract_receipt_json_lines` already handles (it only strips a tab
    prefix when one is present).
    """
    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}/logs"],
        check=True,
        capture_output=True,
    )
    with zipfile.ZipFile(io.BytesIO(proc.stdout)) as archive:
        return "\n".join(
            archive.read(name).decode("utf-8", errors="replace")
            for name in archive.namelist()
            if name.endswith(".txt")
        )


def collect_held_unbound_tickets(
    *,
    repo: str,
    workflow: str,
    since_iso: str,
    limit: int,
) -> tuple[tuple[str, ...], int, int]:
    """Live collection: returns (held tickets, runs scanned, runs unparseable)."""
    run_ids = fetch_sweep_run_ids(
        repo=repo, workflow=workflow, limit=limit, since_iso=since_iso
    )
    events: list[TicketDecisionEvent] = []
    unparseable = 0
    for run_id, created_at in run_ids:
        log_text = fetch_run_log(repo=repo, run_id=run_id)
        lines = extract_receipt_json_lines(log_text)
        if not lines:
            unparseable += 1
            continue
        for line in lines:
            try:
                receipt = parse_sweep_receipt(line)
            except (json.JSONDecodeError, ValueError):
                unparseable += 1
                continue
            events.extend(iter_ticket_decision_events(created_at, receipt))
    held = distinct_tickets_currently_held(events, target_class=GAP_AC_UNBOUND)
    return held, len(run_ids), unparseable


# --- CLI ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OMN-18329: measure the closer's held-unbound residual and the OCC "
            "binding/acceptance corpus counts, each with a positive control."
        )
    )
    parser.add_argument("--repo", default="OmniNode-ai/omnibase_infra")
    parser.add_argument("--workflow", default="evidence-autoclose-sweep.yml")
    parser.add_argument(
        "--since",
        required=True,
        help="ISO-8601 window start, e.g. 2026-09-11T00:00:00Z",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=800,
        help="Max runs to list before the --since filter is applied.",
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        required=True,
        help="Path to a onex_change_control checkout's contracts/ directory.",
    )
    parser.add_argument(
        "--positive-control-ticket",
        default="",
        help="A ticket id expected to be in the held set, asserted as a live positive control.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    held, runs_scanned, runs_unparseable = collect_held_unbound_tickets(
        repo=args.repo, workflow=args.workflow, since_iso=args.since, limit=args.limit
    )
    print(
        f"AC1 window: since={args.since} runs_scanned={runs_scanned} runs_unparseable={runs_unparseable}"
    )
    print(f"AC1 distinct tickets currently held on {GAP_AC_UNBOUND}: {len(held)}")
    print(f"AC1 held ticket ids: {', '.join(held) if held else '(none)'}")
    if args.positive_control_ticket:
        present = args.positive_control_ticket in held
        print(
            f"AC1 positive control ({args.positive_control_ticket} expected held): "
            f"{'PASS' if present else 'FAIL'}"
        )
        if not present:
            return 1

    contracts = list(load_contracts_from_directory(args.contracts_dir))
    counts = count_contract_bindings(contracts)
    print(
        "AC2 contract corpus: "
        f"total={counts.total_contracts} "
        f"with_evidence={counts.contracts_with_evidence} "
        f"with_binding={counts.contracts_with_binding} "
        f"with_accepted_binding={counts.contracts_with_accepted_binding}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
