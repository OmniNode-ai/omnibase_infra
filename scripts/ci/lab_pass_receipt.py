#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17530 — the lab-pass receipt, and the fail-closed gate that reads it.

``omni_home`` ``CLAUDE.md`` Operating Rule 24 ("The lab is the first place a
change runs; staging is promotion", landed ``98b9fb764``) states two mechanics,
and the code below is both of them:

  (a) every runtime-affecting merge to ``dev`` produces an automatic lab pass,
      *each emitting a durable lab-pass receipt keyed by the merged sha*;
  (b) delivery to staging FAILS CLOSED unless a passing lab receipt exists for
      the delivered sha.

Rule 24 shipped with (b) marked *being wired*. This module is the wiring.

Why a receipt at all, when ``dispatch-to-staging`` already ``needs:`` the boot
gate
--------------------------------------------------------------------------
A job dependency is a fact about ONE workflow run. It cannot be read by the
next run, by the deploy that follows in another repository, by a person asking
"was this sha ever exercised on the lab", or by a manual dispatch that skipped
the push path entirely. Rule 24 asks for a *durable artifact keyed by sha*
precisely because the premise has to outlive the run that established it. The
2026-09-08 window is the argument: the tenant-form reader/writer split and the
quality-gate ``NOT NULL`` timestamp were both found on ``onex-dev`` after twelve
failed staging runs, and afterwards nobody could point at an artifact saying
which shas had, and had not, been exercised on a lab lane.

The surface, and why this one
-----------------------------
**A GitHub Actions artifact in this repository, named**
``lab-pass-receipt-<lane>-<sha>``, **carrying one ``receipt.json``.**

Weighed against the alternatives that were on the table:

``OCC dod receipts``
    Durable and git-backed, but a receipt per dev merge means a commit (in
    practice a PR) per dev merge into ``onex_change_control``. That repo is the
    org's evidence-companion surface and is already the deepest CI queue we
    have; adding an automatic per-merge writer to it buys durability at the cost
    of making the delivery path depend on a second repository's merge latency.

``A bus topic projected by the runtime``
    The plan of record's shape (``beta/plans/2026-09-02-staging-mirror-lab-lane-plan.md``
    §6: the ``lane_mirror_refresh_receipt`` projection and its ``golden-gate``
    route, terminal state ``GATED_OK``). It is the right long-run home and this
    module's field names are chosen to port onto it. It is also Phase 4 of that
    plan, sized at nine engineer-days, and every one of its consumers needs a
    broker credential or cluster access. The gate below must run on
    GitHub-hosted compute, in a job that has neither.

``S3``
    Durable, and this repository already assumes an AWS OIDC role in the very
    workflow the gate lives in. But the *emitters* are the constraint, not the
    reader: the ``.201`` lab has no AWS identity, and minting one so a lab lane
    can write a receipt is a new credential surface for a problem a built-in
    already solves.

``A GitHub Actions artifact`` (chosen)
    Exact-sha keyed **by construction**, because the sha is in the artifact
    NAME and the REST API supports an exact-name query
    (``GET /repos/{repo}/actions/artifacts?name=...``). No new credential, no
    new bucket, no new table, no new service, and no cluster access: a
    ``actions: read`` token is enough, which every job in this repository
    already has. Retention is set explicitly to the maximum this repository
    allows at the upload site.

    Its honest limit, stated rather than discovered later: artifact retention is
    finite, so a receipt ages out. That bounds how far back a *historical*
    question can be answered. It does not weaken the gate, which only ever asks
    about a sha whose delivery run is minutes old.

One surface, not two. An "authoritative artifact plus a human-visible commit
status" pair was considered and refused: two surfaces that can disagree is a
worse property than one surface with a stated retention bound, and the gate
would then have to declare which one wins.

What the receipt is NOT
-----------------------
A lab pass is not a staging proof and does not replace the business-proof gate
(rule 24's own stated limit). It also does not carry parity claims: the
``k8s/onex-lab`` overlay's ``parity_exclusions.yaml`` says what a lab lane
cannot prove, and it travels with the boot gate's diagnostics artifact.

Subcommands
-----------
``probe-lane``
    Runs the read-only integration probes an emitting job can prove from the
    runner, and prints the resulting checks. Separated from ``emit`` so the
    probes are unit-testable without a live lane.

``emit``
    Builds and validates a receipt, and writes it to a path the job then
    uploads. Emission is deliberately just "write a validated JSON file": the
    publishing mechanism is ``actions/upload-artifact``, which needs no
    credential this repository does not already grant.

``gate``
    Reads the surface for one sha, prints every receipt it found (sha, lane,
    result, checks), and exits non-zero unless a ``PASS`` receipt exists for the
    EXACT sha. There is no ``--force`` and no skip input, by design: a missing,
    unreadable, malformed or ``FAIL`` receipt is a FAILURE naming the sha, never
    a skip.

Exit codes: ``0`` a PASS receipt for the exact sha exists; ``1`` it does not, or
could not be proven to.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess  # fixed argv, no shell, trusted gh binary
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

#: Bump when a field is added or a meaning changes. A receipt carrying an
#: unknown version is REFUSED by the gate rather than best-effort parsed: a
#: reader that guesses at an unfamiliar receipt is how a gate goes green on a
#: record it did not understand.
RECEIPT_VERSION = "lab_pass_receipt.v1"

DEFAULT_REPO = "OmniNode-ai/omnibase_infra"

#: Artifact retention is capped per-repository; 90 is the GitHub default
#: maximum for public plans and what the upload sites request.
ARTIFACT_RETENTION_DAYS = 90

#: A full 40-character lowercase commit sha. Rule 24(b) says "the exact
#: candidate sha" and the plan of record grants no descendant window, so an
#: abbreviated sha is refused: two different receipts could share a prefix, and
#: a gate that resolves a prefix is a gate that can match the wrong commit.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class EnumLabLane(StrEnum):
    """The lab surfaces rule 24(a) names as receipt emitters.

    ``COMPOSE_DEV`` is the ``.201`` compose dev lane (compose project
    ``omnibase-infra``, ports 8085/8086). ``ONEX_LAB`` is the ``k8s/onex-lab``
    overlay applied from the same head.

    No other value is admissible, and in particular no governed lane
    (``prod``, ``stability-test``, ``judge``, or a collaborator lane) can name
    itself in a receipt. A lab pass is a statement about a lab.
    """

    COMPOSE_DEV = "compose-dev"
    ONEX_LAB = "onex-lab"


class EnumLabPassResult(StrEnum):
    """Terminal verdict. There is no third value.

    An in-flight or indeterminate lab pass emits NO receipt at all rather than a
    ``PENDING`` one, so the gate's "absent" branch and its "not yet passing"
    branch are the same branch, and both fail closed.
    """

    PASS = "PASS"
    FAIL = "FAIL"


class ModelLabPassCheck(BaseModel):
    """One named integration check and the evidence for its verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    ok: bool
    #: What was actually read. Required and non-empty on BOTH verdicts: an
    #: ``ok: true`` with no evidence is indistinguishable from a check that was
    #: never run, which is the shape rule 16 (never suppress stderr; prove a
    #: zero with a positive control) exists to refuse.
    evidence: str = Field(min_length=1)


class ModelLabPassReceipt(BaseModel):
    """A durable statement that one sha was exercised on one lab lane.

    Keyed by ``(sha, lane)``. Two lanes may each emit a receipt for the same
    sha; the gate is satisfied by any one of them passing (rule 24(b) asks for
    "a passing lab receipt", not for a specific lane's).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt_version: str = RECEIPT_VERSION
    #: The exact commit the lab lane exercised.
    sha: str
    lane: EnumLabLane
    started_at: datetime
    finished_at: datetime
    result: EnumLabPassResult
    checks: tuple[ModelLabPassCheck, ...]
    #: The deploy agent's correlation id for the rebuild this receipt attests
    #: to. Required with no default (rule 8: fail fast rather than guess), and
    #: explicitly nullable, because an emitter that cannot resolve it must say
    #: so rather than invent one. The ``onex-lab`` apply has no agent command
    #: at all and always carries ``null``.
    agent_command_id: str | None = Field(...)

    @model_validator(mode="after")
    def validate_sha_is_exact(self) -> ModelLabPassReceipt:
        if not _SHA_RE.match(self.sha):
            msg = (
                f"sha={self.sha!r} is not a 40-character lowercase commit sha. "
                "Rule 24(b) gates on the exact delivered sha; an abbreviated or "
                "uppercase value cannot be matched safely."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_version(self) -> ModelLabPassReceipt:
        if self.receipt_version != RECEIPT_VERSION:
            msg = (
                f"receipt_version={self.receipt_version!r} is not "
                f"{RECEIPT_VERSION!r}. Refusing to interpret a receipt written "
                "against a different contract."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_checks_present(self) -> ModelLabPassReceipt:
        if not self.checks:
            msg = (
                "checks is empty. A receipt with no checks asserts that nothing "
                "was verified, which is not a lab pass."
            )
            raise ValueError(msg)
        names = [c.name for c in self.checks]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            msg = (
                f"duplicate check names {duplicates}. Two rows with one name "
                "means one of them is unreadable, and a reader cannot tell which."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_result_matches_checks(self) -> ModelLabPassReceipt:
        """``PASS`` iff every check passed.

        Without this, a ``PASS`` receipt carrying a failed check would satisfy
        the gate — the exact "green while doing nothing" shape rule 15 records.
        The verdict is therefore not an independent field the emitter may set
        freely; it is a claim the record itself has to support.
        """
        all_ok = all(c.ok for c in self.checks)
        if self.result == EnumLabPassResult.PASS and not all_ok:
            failed = sorted(c.name for c in self.checks if not c.ok)
            msg = (
                f"result=PASS but these checks failed: {failed}. A PASS receipt "
                "must be supported by every check it carries."
            )
            raise ValueError(msg)
        if self.result == EnumLabPassResult.FAIL and all_ok:
            msg = (
                "result=FAIL but every check passed. A verdict that contradicts "
                "its own evidence is refused in both directions, so a FAIL "
                "cannot be used to hide a check the emitter forgot to record."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_window(self) -> ModelLabPassReceipt:
        if self.finished_at < self.started_at:
            msg = (
                f"finished_at={self.finished_at.isoformat()} precedes "
                f"started_at={self.started_at.isoformat()}."
            )
            raise ValueError(msg)
        return self


def artifact_name(lane: EnumLabLane, sha: str) -> str:
    """The exact-name key the gate queries.

    The sha lives in the NAME, not only in the payload, so the lookup cannot
    return a receipt for a different commit: there is no path in which the gate
    reads a receipt it then has to check the sha of. (It checks anyway — see
    ``gate`` — because a name and a payload that disagree is itself a finding.)
    """
    return f"lab-pass-receipt-{lane.value}-{sha}"


# ---------------------------------------------------------------------------
# probe-lane
# ---------------------------------------------------------------------------
#: Checks a ``compose-dev`` emitter proves from the runner, and the reason each
#: is here. The set is deliberately the subset that is provable from an HTTP
#: surface plus the job's own staleness guard; see ``PROBES_NOT_YET_WIRED``.
COMPOSE_DEV_HTTP_CHECKS = ("ready_main", "ready_effects", "health_dimensions")

#: Named here rather than silently absent, so a reader can see what a
#: ``compose-dev`` receipt does NOT cover. Each needs database or broker access
#: the emitting job does not have today; widening the set is a follow-up that
#: adds the probe AND the check name in one change.
PROBES_NOT_YET_WIRED = (
    "migrations_applied",
    "consumer_group_lag",
    "delegation_golden_chain",
)


def _http_get(url: str, timeout_seconds: float) -> tuple[int, str]:
    """GET a lane endpoint. Never raises; the failure IS the evidence."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - any transport failure is a FAIL
        return 0, f"{type(exc).__name__}: {exc}"
    return status, body


def _truncate(text: str, limit: int = 240) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def check_ready(name: str, url: str, timeout_seconds: float) -> ModelLabPassCheck:
    """A lane readiness endpoint must answer 200. Anything else is a FAIL."""
    status, body = _http_get(url, timeout_seconds)
    return ModelLabPassCheck(
        name=name,
        ok=status == 200,
        evidence=f"GET {url} -> {status} {_truncate(body)}",
    )


def check_health_dimensions(url: str, timeout_seconds: float) -> ModelLabPassCheck:
    """Every dimension the health payload reports must be healthy.

    Fails closed on an unparseable body and on a payload carrying no
    dimensions at all: "we could not find an unhealthy dimension" is not the
    same statement as "every dimension is healthy", and only the second one is
    a check.
    """
    status, body = _http_get(url, timeout_seconds)
    if status != 200:
        return ModelLabPassCheck(
            name="health_dimensions",
            ok=False,
            evidence=f"GET {url} -> {status} {_truncate(body)}",
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return ModelLabPassCheck(
            name="health_dimensions",
            ok=False,
            evidence=f"GET {url} -> 200 but body is not JSON: {exc}",
        )
    dimensions = payload.get("dimensions") if isinstance(payload, dict) else None
    if not isinstance(dimensions, dict) or not dimensions:
        return ModelLabPassCheck(
            name="health_dimensions",
            ok=False,
            evidence=(
                f"GET {url} -> 200 but carries no 'dimensions' object; "
                "an absent dimension set is not a healthy dimension set"
            ),
        )
    unhealthy = sorted(
        key
        for key, value in dimensions.items()
        if str(_dimension_status(value)).lower() not in {"healthy", "ok", "pass", "up"}
    )
    return ModelLabPassCheck(
        name="health_dimensions",
        ok=not unhealthy,
        evidence=(
            f"GET {url} -> 200, {len(dimensions)} dimensions, "
            + ("all healthy" if not unhealthy else f"unhealthy: {unhealthy}")
        ),
    )


def _dimension_status(value: Any) -> Any:
    """A dimension is either a status string or an object carrying one."""
    if isinstance(value, dict):
        return value.get("status", value.get("state", "unknown"))
    return value


def probe_compose_dev(
    main_url: str, effects_url: str, timeout_seconds: float
) -> list[ModelLabPassCheck]:
    """The read-only probes the ``.201`` dev lane emitter runs."""
    return [
        check_ready("ready_main", f"{main_url.rstrip('/')}/ready", timeout_seconds),
        check_ready(
            "ready_effects", f"{effects_url.rstrip('/')}/ready", timeout_seconds
        ),
        check_health_dimensions(f"{main_url.rstrip('/')}/health", timeout_seconds),
    ]


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------
def parse_check_argument(raw: str) -> ModelLabPassCheck:
    """Parse ``name:ok|fail:evidence`` from the command line.

    Split on the first two colons only, so evidence may contain colons (URLs
    and timestamps both do).
    """
    parts = raw.split(":", 2)
    if len(parts) != 3:
        msg = (
            f"--check {raw!r} is not 'name:ok|fail:evidence'. All three fields "
            "are required; a check with no evidence is not a check."
        )
        raise ValueError(msg)
    name, verdict, evidence = parts
    verdict_normalised = verdict.strip().lower()
    if verdict_normalised not in {"ok", "fail"}:
        msg = f"--check {raw!r}: verdict must be 'ok' or 'fail', got {verdict!r}."
        raise ValueError(msg)
    return ModelLabPassCheck(
        name=name.strip(), ok=verdict_normalised == "ok", evidence=evidence.strip()
    )


def load_checks_json(path: Path | None) -> list[ModelLabPassCheck]:
    """Load checks written by ``probe-lane``.

    A missing or unparseable file RAISES. An emitter that pointed at a probe
    output and got nothing must not quietly emit a receipt with fewer checks
    than it believes it carries — that is a PASS narrower than it looks.
    """
    if path is None:
        return []
    body = path.read_text(encoding="utf-8")
    payload = json.loads(body)
    if not isinstance(payload, list) or not payload:
        msg = (
            f"{path} does not carry a non-empty JSON array of checks. "
            "Refusing to emit a receipt whose probe output is empty."
        )
        raise ValueError(msg)
    return [ModelLabPassCheck.model_validate(entry) for entry in payload]


def build_receipt(
    sha: str,
    lane: EnumLabLane,
    started_at: datetime,
    finished_at: datetime,
    checks: Sequence[ModelLabPassCheck],
    agent_command_id: str | None,
) -> ModelLabPassReceipt:
    """Build a receipt whose verdict is DERIVED from its checks.

    The verdict is never an input. An emitter that could pass ``result`` would
    eventually pass the wrong one.
    """
    result = (
        EnumLabPassResult.PASS
        if checks and all(c.ok for c in checks)
        else EnumLabPassResult.FAIL
    )
    return ModelLabPassReceipt(
        sha=sha,
        lane=lane,
        started_at=started_at,
        finished_at=finished_at,
        result=result,
        checks=tuple(checks),
        agent_command_id=agent_command_id,
    )


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
class ReceiptLookupError(RuntimeError):
    """The surface could not be read. Always a gate FAILURE, never a skip."""


def _gh_api(path: str) -> bytes:
    """One read against the GitHub REST API, as raw bytes.

    Bytes rather than text because the artifact-zip endpoint is binary, and a
    non-zero exit RAISES rather than returning empty output — rule 16: a sweep
    whose error is discarded returns zero rows and reads like a clean bill of
    health.
    """
    argv = ["gh", "api", path]
    completed = subprocess.run(argv, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        msg = f"`gh api {path}` exited {completed.returncode}: {stderr}"
        raise ReceiptLookupError(msg)
    return completed.stdout


def list_artifacts(repo: str, name: str) -> list[dict[str, Any]]:
    """Exact-name artifact query. A transport failure raises; it never returns []."""
    raw = _gh_api(f"repos/{repo}/actions/artifacts?name={name}&per_page=100")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"artifact listing for {name!r} is not JSON: {exc}"
        raise ReceiptLookupError(msg) from exc
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        msg = f"artifact listing for {name!r} carries no 'artifacts' list."
        raise ReceiptLookupError(msg)
    return [a for a in artifacts if isinstance(a, dict) and not a.get("expired", False)]


def download_receipt(repo: str, artifact_id: int) -> ModelLabPassReceipt:
    """Download one artifact zip and parse the single ``receipt.json`` in it."""
    blob = _gh_api(f"repos/{repo}/actions/artifacts/{artifact_id}/zip")
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            members = [n for n in archive.namelist() if n.endswith("receipt.json")]
            if len(members) != 1:
                msg = (
                    f"artifact {artifact_id} carries {len(members)} receipt.json "
                    "entries; exactly one is required."
                )
                raise ReceiptLookupError(msg)
            body = archive.read(members[0]).decode("utf-8")
    except zipfile.BadZipFile as exc:
        msg = f"artifact {artifact_id} is not a readable zip: {exc}"
        raise ReceiptLookupError(msg) from exc
    return parse_receipt(body)


def parse_receipt(body: str) -> ModelLabPassReceipt:
    """Parse and VALIDATE. A malformed receipt raises rather than degrading."""
    try:
        return ModelLabPassReceipt.model_validate_json(body)
    except ValidationError as exc:
        msg = f"receipt is malformed and cannot be trusted: {exc}"
        raise ReceiptLookupError(msg) from exc


def render_receipt(receipt: ModelLabPassReceipt) -> str:
    """The gate PRINTS what it read. A verdict with no readback is folklore."""
    lines = [
        f"  sha        : {receipt.sha}",
        f"  lane       : {receipt.lane.value}",
        f"  result     : {receipt.result.value}",
        f"  window     : {receipt.started_at.isoformat()} -> "
        f"{receipt.finished_at.isoformat()}",
        f"  agent cmd  : {receipt.agent_command_id or '(none — not an agent deploy)'}",
        "  checks     :",
    ]
    lines.extend(
        f"    [{'ok  ' if c.ok else 'FAIL'}] {c.name}: {c.evidence}"
        for c in receipt.checks
    )
    return "\n".join(lines)


def evaluate_gate(repo: str, sha: str, lanes: Sequence[EnumLabLane], out: Any) -> int:
    """Fail closed unless a PASS receipt exists for the EXACT sha.

    Every terminal branch prints the sha. "The gate failed" with no commit named
    is unactionable at 3am, and the whole point of a sha-keyed receipt is that
    the answer is about one commit.
    """
    if not _SHA_RE.match(sha):
        print(
            f"::error::lab-pass gate: {sha!r} is not a 40-character lowercase "
            "commit sha. Refusing to resolve an abbreviated ref.",
            file=out,
        )
        return 1

    found: list[ModelLabPassReceipt] = []
    problems: list[str] = []

    for lane in lanes:
        name = artifact_name(lane, sha)
        try:
            artifacts = list_artifacts(repo, name)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad. Rule 16: a verification sweep that errors and
            # is not caught reads as an absence of findings; here it would read
            # as a traceback with no sha in it. Any failure to READ the surface
            # is reported as "unreadable" against the named lane and fails the
            # gate, which is the same verdict as a missing receipt.
            problems.append(f"{lane.value}: {exc}")
            continue
        if not artifacts:
            problems.append(f"{lane.value}: no receipt artifact named {name}")
            continue
        # Newest first: a re-run of the emitting job supersedes an earlier
        # attempt for the same sha and lane.
        artifacts.sort(key=lambda a: str(a.get("created_at", "")), reverse=True)
        try:
            receipt = download_receipt(repo, int(artifacts[0]["id"]))
        except Exception as exc:  # noqa: BLE001 - same reasoning as above
            problems.append(f"{lane.value}: {exc}")
            continue
        if receipt.sha != sha:
            problems.append(
                f"{lane.value}: artifact {name} carries sha {receipt.sha}; "
                "the name and the payload disagree"
            )
            continue
        if receipt.lane != lane:
            problems.append(
                f"{lane.value}: artifact {name} carries lane {receipt.lane.value}; "
                "the name and the payload disagree"
            )
            continue
        found.append(receipt)

    print(f"lab-pass gate (rule 24(b), OMN-17530) for sha {sha}", file=out)
    print(f"  repository : {repo}", file=out)
    print(f"  lanes read : {', '.join(lane.value for lane in lanes)}", file=out)
    for receipt in found:
        print("", file=out)
        print(render_receipt(receipt), file=out)
    for problem in problems:
        print(f"  unreadable : {problem}", file=out)

    passing = [r for r in found if r.result == EnumLabPassResult.PASS]
    if passing:
        lanes_passing = ", ".join(r.lane.value for r in passing)
        print("", file=out)
        print(
            f"lab-pass gate PASSED for {sha} on lane(s): {lanes_passing}.",
            file=out,
        )
        return 0

    print("", file=out)
    print(
        f"::error::lab-pass gate FAILED for {sha}: no PASS lab-pass receipt "
        f"exists for this exact sha on any of {', '.join(lane.value for lane in lanes)}. "
        "Rule 24(b) — the lab is the first place a change runs; staging is "
        "promotion — so this candidate is not deliverable. This is not a skip: a "
        "missing, unreadable, malformed or FAIL receipt all fail here, and there "
        "is no override flag. Exercise the sha on a lab lane and let its emitter "
        "publish the receipt.",
        file=out,
    )
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser(
        "probe-lane", help="run the read-only lab probes and print checks"
    )
    probe.add_argument("--lane", required=True, choices=[e.value for e in EnumLabLane])
    probe.add_argument("--main-url", required=True)
    probe.add_argument("--effects-url", required=True)
    probe.add_argument("--timeout-seconds", type=float, default=15.0)

    emit = sub.add_parser("emit", help="build, validate and write a receipt")
    emit.add_argument("--sha", required=True)
    emit.add_argument("--lane", required=True, choices=[e.value for e in EnumLabLane])
    emit.add_argument("--started-at", required=True)
    emit.add_argument("--finished-at", required=True)
    emit.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="NAME:ok|fail:EVIDENCE",
        help="repeatable; at least one is required",
    )
    emit.add_argument(
        "--checks-json",
        type=Path,
        default=None,
        help=(
            "a JSON array of {name, ok, evidence} objects, as written by "
            "probe-lane; merged with any --check arguments"
        ),
    )
    emit.add_argument(
        "--agent-command-id",
        default=None,
        help=(
            "the deploy agent's correlation id; omit only when the emitter "
            "genuinely has none (the onex-lab apply never does)"
        ),
    )
    emit.add_argument("--out", required=True, type=Path)

    gate = sub.add_parser("gate", help="fail closed unless a PASS receipt exists")
    gate.add_argument("--sha", required=True)
    gate.add_argument("--repo", default=DEFAULT_REPO)
    gate.add_argument(
        "--lane",
        action="append",
        default=[],
        choices=[e.value for e in EnumLabLane],
        help="repeatable; defaults to every lab lane",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "probe-lane":
        checks = probe_compose_dev(
            args.main_url, args.effects_url, args.timeout_seconds
        )
        print(json.dumps([c.model_dump() for c in checks], indent=2))
        return 0

    if args.command == "emit":
        if not args.check and args.checks_json is None:
            print(
                "::error::emit requires at least one --check. A receipt with no "
                "checks asserts that nothing was verified.",
                file=sys.stderr,
            )
            return 1
        try:
            checks = [parse_check_argument(raw) for raw in args.check]
            checks.extend(load_checks_json(args.checks_json))
            receipt = build_receipt(
                sha=args.sha,
                lane=EnumLabLane(args.lane),
                started_at=_parse_ts(args.started_at),
                finished_at=_parse_ts(args.finished_at),
                checks=checks,
                agent_command_id=args.agent_command_id,
            )
        except (ValueError, ValidationError) as exc:
            print(
                f"::error::refusing to emit an invalid receipt: {exc}", file=sys.stderr
            )
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        print(f"wrote {artifact_name(receipt.lane, receipt.sha)} -> {args.out}")
        print(render_receipt(receipt))
        # A FAIL receipt is still EMITTED — the record of a failed lab pass is
        # exactly as valuable as the record of a passing one — but the emitting
        # step goes red so the failure is visible on the run that produced it.
        return 0 if receipt.result == EnumLabPassResult.PASS else 1

    if args.command == "gate":
        lanes = (
            [EnumLabLane(value) for value in args.lane]
            if args.lane
            else list(EnumLabLane)
        )
        return evaluate_gate(args.repo, args.sha, lanes, sys.stdout)

    raise AssertionError(f"unreachable subcommand {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
