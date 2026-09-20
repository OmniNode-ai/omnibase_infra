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

STDLIB ONLY, and that is a requirement rather than a preference
------------------------------------------------------------
Both call sites run on a bare runner with no project environment. The boot gate
checks this repository out into a SUBDIRECTORY (``path: omnibase_infra``), so
``$GITHUB_WORKSPACE/.github/actions/setup-python-uv`` does not exist there and a
local composite action cannot be referenced at all -- which is why every script
that job already calls (``boot_gate.py``, ``render_ci_secrets.py``) is stdlib
plus PyYAML. This module joins them.

Measured, not assumed: the first live run of the emitter (run 34235502322,
2026-09-08T14:14:36Z) died with ``ModuleNotFoundError: No module named
'pydantic'`` at import, after all four of its checks had already passed, and
took the whole delivery with it. The validation below is therefore hand-written
on frozen dataclasses. It is the SAME set of invariants a Pydantic model would
carry -- they are load-bearing, so they are asserted in ``__post_init__`` rather
than dropped -- and ``ValueError`` is the single failure type.

Exit codes: ``0`` a PASS receipt for the exact sha exists; ``1`` it does not, or
could not be proven to.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess  # fixed argv, no shell, trusted gh binary
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol

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

#: OMN-18708. A contract content hash as
#: ``omnibase_infra/runtime/util_contract_content_hash.py`` declares it: 64
#: lowercase hex, no prefix. Refused rather than normalised -- two contracts
#: can share a prefix, and a reader that normalises can match the wrong one.
_CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

#: The runtime's own introspection surface, served by the health server on the
#: SAME port as ``/ready`` and ``/health`` (``ServiceHealth`` route table). It
#: is the lane's answer about itself, which is why it is the evidence for the
#: inventory check: a restatement of what the build stamped would prove only
#: that the emitter can echo its own input.
INTROSPECTION_MANIFEST_PATH: Final[str] = "/v1/introspection/manifest"


@dataclass(frozen=True)
class ModelNodeInventoryTriple:
    """One node the LANE reports running, as (name, version, contract hash).

    OMN-18708. The same triple the image label carries, read from the other
    end: the label says what an image SHIPS, this says what a lane WIRED. They
    are not the same set -- a lane wires the profile-filtered subset of the
    image's contracts -- and conflating them would be a claim nobody made. What
    they do share is the third field: both compute it with the one canonical
    hasher over the same contract file, so a contract whose body drifted is
    visible from either end.

    Defined here rather than imported from ``omnibase_infra.runtime`` for the
    reason ``test_the_module_imports_nothing_outside_the_stdlib`` records: run
    34235502322 died at import on a bare runner and took a whole delivery with
    it. This module is stdlib-only, repo-local imports included.
    """

    name: str
    node_version: str
    contract_content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            msg = "node inventory triple: name is required and must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.node_version, str) or not self.node_version:
            msg = (
                f"node inventory triple {self.name!r}: node_version is required "
                "and must be non-empty"
            )
            raise ValueError(msg)
        if not isinstance(
            self.contract_content_hash, str
        ) or not _CONTENT_HASH_RE.match(self.contract_content_hash):
            msg = (
                f"node inventory triple {self.name!r}: contract_content_hash="
                f"{self.contract_content_hash!r} is not 64 lowercase hex "
                "characters. A truncated or absent hash cannot detect the drift "
                "class the field exists for, so it is refused rather than kept."
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "node_version": self.node_version,
            "contract_content_hash": self.contract_content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ModelNodeInventoryTriple:
        if not isinstance(payload, dict):
            msg = (
                "a node inventory triple must be an object, got "
                f"{type(payload).__name__}"
            )
            raise ValueError(msg)
        unknown = sorted(
            set(payload) - {"name", "node_version", "contract_content_hash"}
        )
        if unknown:
            msg = f"unknown node inventory field(s) {unknown}"
            raise ValueError(msg)
        try:
            return cls(
                name=payload["name"],
                node_version=payload["node_version"],
                contract_content_hash=payload["contract_content_hash"],
            )
        except KeyError as exc:
            msg = f"node inventory triple is missing required field {exc.args[0]!r}"
            raise ValueError(msg) from exc


def parse_node_inventory(payload: Any) -> tuple[ModelNodeInventoryTriple, ...]:
    """Parse a triple list, refusing duplicates and anything unrecognised."""
    if not isinstance(payload, list):
        msg = f"node inventory must be a list, got {type(payload).__name__}"
        raise ValueError(msg)
    triples = tuple(ModelNodeInventoryTriple.from_dict(row) for row in payload)
    names = [t.name for t in triples]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        msg = (
            f"duplicate node name(s) {duplicates} in the node inventory. A "
            "reader resolving a name would have to pick one, and whichever it "
            "picked would look authoritative."
        )
        raise ValueError(msg)
    return tuple(sorted(triples, key=lambda t: (t.name, t.node_version)))


# Declared by omnimarket's projection API
# (src/omnimarket/projection/api_server.py::list_projections) and pinned by
# tests/test_projection_api_server.py::test_projections_entry_has_required_fields.
PROJECTION_TOPIC_REQUIRED_FIELDS = frozenset(
    {
        "topic",
        "table",
        "status",
        "columns",
        "limit",
        "source_contract",
        "bus_backed",
        "backing",
    }
)


class EnumLabLane(StrEnum):
    """The lab surfaces rule 24(a) names as receipt emitters.

    ``COMPOSE_DEV`` is the ``.201`` compose dev lane (compose project
    ``omnibase-infra``, ports 8085/8086). ``ONEX_LAB`` is the ``k8s/onex-lab``
    overlay applied from the same head.

    ``ONEX_LAB_K3S`` (OMN-18200) is the PERSISTENT lab cluster -- the same
    overlay, applied to the k3s node on the lab host by
    ``k8s/onex-lab/apply_lab_lane.sh`` rather than to a per-candidate ``kind``
    cluster. It is a separate value rather than a second emitter on ``ONEX_LAB``
    for two reasons, both load-bearing:

    *``evaluate_gate`` assumes one emitter per name.* It sorts an exact-name
    artifact query newest-first and reads only the newest, on the premise that a
    later artifact for a name is a re-run of the same job. Two unrelated
    emitters on one name would therefore let whichever finished last silently
    supersede the other's verdict, with nothing recording that a verdict had
    been discarded.

    *They are not the same claim.* ``ONEX_LAB`` proves the candidate BOOTS
    against the real manifests on a throwaway node with one side-loaded image and
    an inert credential store. ``ONEX_LAB_K3S`` proves the persistent lane a
    chain runner actually grades against is RUNNING the merged sha, with the lab
    host's own secret store bound and the lane's tenant minted. A receipt that
    conflated them would answer a question nobody asked.

    No other value is admissible, and in particular no governed lane
    (``prod``, ``stability-test``, ``judge``, or a collaborator lane) can name
    itself in a receipt. A lab pass is a statement about a lab.
    """

    COMPOSE_DEV = "compose-dev"
    ONEX_LAB = "onex-lab"
    ONEX_LAB_K3S = "onex-lab-k3s"


class EnumLabPassResult(StrEnum):
    """Terminal verdict. There is no third value.

    An in-flight or indeterminate lab pass emits NO receipt at all rather than a
    ``PENDING`` one, so the gate's "absent" branch and its "not yet passing"
    branch are the same branch, and both fail closed.
    """

    PASS = "PASS"
    FAIL = "FAIL"


class EnumLabPassCheckOutcome(StrEnum):
    """What one check concluded. THREE values, unlike the receipt's verdict.

    The receipt's own ``result`` stays two-valued (see
    :class:`EnumLabPassResult`): a lab pass either passed or it did not, and an
    in-flight one emits no receipt. A CHECK is a different question, and
    OMN-18573 measured the cost of collapsing its third answer into ``FAIL``.

    ``deployed_revision`` reports whether the lane converged onto the merge
    sha. It can only mean that once the lane has been GRANTED a budget to
    converge in, which starts when the deploy agent accepts the rebuild
    command. When that acceptance cannot be established -- no correlation id,
    an unreachable agent, a command the effect never handed over -- the check
    has learnt nothing about the lane. Reporting that as ``FAIL`` asserts the
    lane misbehaved, and rule 24(b) then refuses a sha on a claim nobody made.

    ``INDETERMINATE`` still leaves the receipt non-PASS, because ``ok`` is
    false and the verdict rule is "PASS iff every check passed". Nothing opens;
    what changes is what the receipt SAYS, and therefore what the gate's
    refusal says.
    """

    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ModelLabPassCheck:
    """One named integration check and the evidence for its verdict."""

    name: str
    ok: bool
    #: What was actually read. Required and non-empty on BOTH verdicts: an
    #: ``ok: true`` with no evidence is indistinguishable from a check that was
    #: never run, which is the shape rule 16 (never suppress stderr; prove a
    #: zero with a positive control) exists to refuse.
    evidence: str
    #: OMN-18573. Set only when the check could not be ESTABLISHED, as opposed
    #: to being established and negative. ``ok`` is false either way -- this
    #: never opens anything -- so the pair has exactly one illegal combination
    #: (``ok`` and ``indeterminate`` both true) and it is refused below.
    indeterminate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            msg = "check name is required and must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(self.ok, bool):
            msg = (
                f"check {self.name!r}: ok must be a bool, got {type(self.ok).__name__}"
            )
            raise ValueError(msg)
        if not isinstance(self.indeterminate, bool):
            msg = (
                f"check {self.name!r}: indeterminate must be a bool, got "
                f"{type(self.indeterminate).__name__}"
            )
            raise ValueError(msg)
        if self.ok and self.indeterminate:
            msg = (
                f"check {self.name!r}: a check cannot be both passing and "
                "indeterminate. Indeterminate means the check could not be "
                "established, which is never a pass."
            )
            raise ValueError(msg)
        if not isinstance(self.evidence, str) or not self.evidence:
            msg = (
                f"check {self.name!r}: evidence is required and must be non-empty. "
                "A check with no evidence is indistinguishable from one that "
                "was never run."
            )
            raise ValueError(msg)

    @classmethod
    def indeterminate_check(cls, name: str, evidence: str) -> ModelLabPassCheck:
        """The named constructor for the third outcome.

        A classmethod rather than a keyword at every call site, so a reader of
        an emitter can see which of the three a check is without reading two
        booleans and combining them.
        """
        return cls(name=name, ok=False, evidence=evidence, indeterminate=True)

    @property
    def outcome(self) -> EnumLabPassCheckOutcome:
        """The tri-state, for readers that must tell the three apart."""
        if self.ok:
            return EnumLabPassCheckOutcome.PASS
        if self.indeterminate:
            return EnumLabPassCheckOutcome.INDETERMINATE
        return EnumLabPassCheckOutcome.FAIL

    def to_dict(self) -> dict[str, Any]:
        """The wire form, which carries ``outcome`` ONLY when it adds something.

        A PASS and a FAIL are fully described by ``ok``, so they serialise
        byte-identically to every receipt written before OMN-18573. The extra
        key appears only on an INDETERMINATE check -- so a reader that predates
        this change refuses exactly the receipts it could not have interpreted,
        and parses unchanged every receipt it could. That refusal is the gate
        failing CLOSED on a receipt it does not understand, which is this
        module's stated behaviour for an unparseable receipt everywhere else.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "ok": self.ok,
            "evidence": self.evidence,
        }
        if self.indeterminate:
            payload["outcome"] = EnumLabPassCheckOutcome.INDETERMINATE.value
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ModelLabPassCheck:
        if not isinstance(payload, dict):
            msg = f"a check must be an object, got {type(payload).__name__}"
            raise ValueError(msg)
        unknown = sorted(set(payload) - {"name", "ok", "evidence", "outcome"})
        if unknown:
            # extra="forbid", by hand: an unrecognised field means the writer
            # and the reader disagree about the contract.
            msg = f"unknown check field(s) {unknown}"
            raise ValueError(msg)
        indeterminate = False
        if "outcome" in payload:
            try:
                outcome = EnumLabPassCheckOutcome(payload["outcome"])
            except ValueError as exc:
                msg = (
                    f"check outcome {payload['outcome']!r} is not one of "
                    f"{[m.value for m in EnumLabPassCheckOutcome]}"
                )
                raise ValueError(msg) from exc
            indeterminate = outcome is EnumLabPassCheckOutcome.INDETERMINATE
            if not indeterminate:
                # The two-valued outcomes are carried by ``ok`` alone, so a
                # writer spelling one here is a writer this reader does not
                # understand. Refuse rather than pick a winner.
                msg = (
                    f"check {payload.get('name')!r} spells outcome "
                    f"{outcome.value!r} explicitly; only "
                    f"{EnumLabPassCheckOutcome.INDETERMINATE.value!r} is written "
                    "on the wire, because ok carries the other two."
                )
                raise ValueError(msg)
        try:
            return cls(
                name=payload["name"],
                ok=payload["ok"],
                evidence=payload["evidence"],
                indeterminate=indeterminate,
            )
        except KeyError as exc:
            msg = f"check is missing required field {exc.args[0]!r}"
            raise ValueError(msg) from exc


@dataclass(frozen=True)
class ModelLabPassReceipt:
    """A durable statement that one sha was exercised on one lab lane.

    Keyed by ``(sha, lane)``. Two lanes may each emit a receipt for the same
    sha; the gate is satisfied by any one of them passing (rule 24(b) asks for
    "a passing lab receipt", not for a specific lane's).

    Every invariant below is asserted in ``__post_init__`` so a receipt cannot
    exist in an inconsistent state -- construction is the only gate, and there
    is no path that builds one and validates it later.
    """

    sha: str
    lane: EnumLabLane
    started_at: datetime
    finished_at: datetime
    result: EnumLabPassResult
    checks: tuple[ModelLabPassCheck, ...]
    #: The deploy agent's correlation id for the rebuild this receipt attests
    #: to. Required with no default (rule 8: fail fast rather than guess), and
    #: explicitly nullable, because an emitter that cannot resolve it must say
    #: so rather than invent one. The ``onex-lab`` boot gate has no agent command
    #: at all and always carries ``null``; ``onex-lab-k3s`` DOES carry one, since
    #: its apply is performed by the agent as part of one rebuild correlation
    #: (OMN-18200), which is another respect in which the two are not one lane.
    agent_command_id: str | None
    #: OMN-18708. The (name, node_version, contract_content_hash) triples the
    #: LANE reported for itself, read from its introspection manifest by the
    #: ``node_inventory`` check. Empty when that check did not run, which is
    #: the pre-OMN-18708 shape and is why the field defaults rather than being
    #: required: a receipt from an emitter that never probed the manifest is
    #: still a valid receipt about the checks it did run.
    #:
    #: It is written to the wire ONLY when non-empty, so every receipt written
    #: before this change still parses byte-identically and a reader predating
    #: it refuses exactly the receipts it could not have interpreted -- the
    #: same rule ``ModelLabPassCheck.to_dict`` applies to ``outcome``. That is
    #: why ``RECEIPT_VERSION`` is unchanged: no existing field moved, and a
    #: bump would have made the gate refuse every receipt already in flight.
    node_inventory: tuple[ModelNodeInventoryTriple, ...] = ()
    receipt_version: str = RECEIPT_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        self._validate_sha_is_exact()
        self._validate_checks_present()
        self._validate_result_matches_checks()
        self._validate_window()
        self._validate_node_inventory()

    def _validate_version(self) -> None:
        if self.receipt_version != RECEIPT_VERSION:
            msg = (
                f"receipt_version={self.receipt_version!r} is not "
                f"{RECEIPT_VERSION!r}. Refusing to interpret a receipt written "
                "against a different contract."
            )
            raise ValueError(msg)

    def _validate_sha_is_exact(self) -> None:
        if not isinstance(self.sha, str) or not _SHA_RE.match(self.sha):
            msg = (
                f"sha={self.sha!r} is not a 40-character lowercase commit sha. "
                "Rule 24(b) gates on the exact delivered sha; an abbreviated or "
                "uppercase value cannot be matched safely."
            )
            raise ValueError(msg)

    def _validate_checks_present(self) -> None:
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

    def _validate_result_matches_checks(self) -> None:
        """``PASS`` iff every check passed.

        Without this, a ``PASS`` receipt carrying a failed check would satisfy
        the gate -- the exact "green while doing nothing" shape rule 15 records.
        The verdict is therefore not an independent field the emitter may set
        freely; it is a claim the record itself has to support.
        """
        all_ok = all(c.ok for c in self.checks)
        if self.result == EnumLabPassResult.PASS and not all_ok:
            failed = sorted(
                c.name for c in self.checks if c.outcome is EnumLabPassCheckOutcome.FAIL
            )
            unestablished = sorted(
                c.name
                for c in self.checks
                if c.outcome is EnumLabPassCheckOutcome.INDETERMINATE
            )
            msg = (
                f"result=PASS but these checks failed: {failed} and these could "
                f"not be established: {unestablished}. A PASS receipt must be "
                "supported by every check it carries, and an indeterminate "
                "check supports nothing."
            )
            raise ValueError(msg)
        if self.result == EnumLabPassResult.FAIL and all_ok:
            msg = (
                "result=FAIL but every check passed. A verdict that contradicts "
                "its own evidence is refused in both directions, so a FAIL "
                "cannot be used to hide a check the emitter forgot to record."
            )
            raise ValueError(msg)

    def _validate_node_inventory(self) -> None:
        """A passing inventory check must be backed by the triples it read.

        The check's verdict says "I read the lane's manifest and it named its
        nodes". If the receipt then carries no triples, the check asserts
        something the record does not contain -- the "green while doing
        nothing" shape this module refuses everywhere else. Duplicates are
        refused for the same reason two checks with one name are.
        """
        names = [t.name for t in self.node_inventory]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            msg = (
                f"duplicate node name(s) {duplicates} in node_inventory. A "
                "reader resolving a name would have to pick one."
            )
            raise ValueError(msg)
        inventory_check = next(
            (c for c in self.checks if c.name == NODE_INVENTORY_CHECK), None
        )
        if (
            inventory_check is not None
            and inventory_check.ok
            and not self.node_inventory
        ):
            msg = (
                f"the {NODE_INVENTORY_CHECK!r} check passed but the receipt "
                "carries no node_inventory. The check asserts the lane named "
                "its nodes; a receipt that then carries none does not support "
                "its own check."
            )
            raise ValueError(msg)

    def _validate_window(self) -> None:
        if self.finished_at < self.started_at:
            msg = (
                f"finished_at={self.finished_at.isoformat()} precedes "
                f"started_at={self.started_at.isoformat()}."
            )
            raise ValueError(msg)

    # -- serialisation ------------------------------------------------------
    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            {
                "receipt_version": self.receipt_version,
                "sha": self.sha,
                "lane": self.lane.value,
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat(),
                "result": self.result.value,
                "checks": [c.to_dict() for c in self.checks],
                "agent_command_id": self.agent_command_id,
                # Omitted entirely when empty, so a receipt from an emitter
                # that did not probe the manifest is byte-identical to one
                # written before OMN-18708.
                **(
                    {"node_inventory": [t.to_dict() for t in self.node_inventory]}
                    if self.node_inventory
                    else {}
                ),
            },
            indent=indent,
        )

    @classmethod
    def from_json(cls, body: str) -> ModelLabPassReceipt:
        """Parse a receipt, refusing anything it does not fully understand.

        Unknown fields, missing fields, a lane or result outside its enum, and
        an unparseable timestamp are all refusals -- a reader that guesses at an
        unfamiliar receipt is how a gate goes green on a record it did not
        understand.
        """
        payload = json.loads(body)
        if not isinstance(payload, dict):
            msg = f"a receipt must be a JSON object, got {type(payload).__name__}"
            raise ValueError(msg)
        known = {
            "receipt_version",
            "sha",
            "lane",
            "started_at",
            "finished_at",
            "result",
            "checks",
            "agent_command_id",
        }
        # OMN-18708: present only on receipts whose emitter probed the lane's
        # introspection manifest, so it is known-but-optional rather than
        # required. Absent means "not probed", which is a real answer.
        optional = {"node_inventory"}
        unknown = sorted(set(payload) - known - optional)
        if unknown:
            msg = f"unknown receipt field(s) {unknown}"
            raise ValueError(msg)
        missing = sorted(known - set(payload))
        if missing:
            msg = f"receipt is missing required field(s) {missing}"
            raise ValueError(msg)
        raw_checks = payload["checks"]
        if not isinstance(raw_checks, list):
            msg = "receipt 'checks' must be a list"
            raise ValueError(msg)
        agent_command_id = payload["agent_command_id"]
        if agent_command_id is not None and not isinstance(agent_command_id, str):
            msg = "agent_command_id must be a string or null"
            raise ValueError(msg)
        return cls(
            receipt_version=str(payload["receipt_version"]),
            sha=payload["sha"],
            lane=EnumLabLane(payload["lane"]),
            started_at=_parse_ts(str(payload["started_at"])),
            finished_at=_parse_ts(str(payload["finished_at"])),
            result=EnumLabPassResult(payload["result"]),
            checks=tuple(ModelLabPassCheck.from_dict(c) for c in raw_checks),
            agent_command_id=agent_command_id,
            node_inventory=(
                parse_node_inventory(payload["node_inventory"])
                if "node_inventory" in payload
                else ()
            ),
        )


def artifact_name(lane: EnumLabLane, sha: str) -> str:
    """The exact-name key the gate queries.

    The sha lives in the NAME, not only in the payload, so the lookup cannot
    return a receipt for a different commit: there is no path in which the gate
    reads a receipt it then has to check the sha of. (It checks anyway — see
    ``gate`` — because a name and a payload that disagree is itself a finding.)
    """
    return f"lab-pass-receipt-{lane.value}-{sha}"


# ---------------------------------------------------------------------------
# lane generation and settle budget -- receipt vocabulary, stdlib only
# ---------------------------------------------------------------------------
# THESE LIVE HERE, not in a sibling module, for the reason
# ``test_the_module_imports_nothing_outside_the_stdlib`` records: run
# 34235502322 died at import on a bare runner and took a whole dev-candidate
# delivery with it. That gate bans EVERY non-stdlib import root, a repo-local
# one included, because the boot gate checks this repository out into a
# subdirectory where a package path is not guaranteed to resolve. So the models
# a receipt is written in terms of are defined in the module that writes it, and
# the sibling that READS a declaration (and needs PyYAML to do it) imports THEM
# rather than the other way round.

#: Stamped from the git sha at build time by ``docker/Dockerfile.runtime``; the
#: same label ``check_dev_lane_staleness.py`` resolves convergence from.
REVISION_LABEL: Final[str] = "org.opencontainers.image.revision"


@dataclass(frozen=True)
class ModelLaneGeneration:
    """One running container, identified so two reads can be compared.

    ``container_id`` is the whole 64-character id as docker reports it; it is
    abbreviated only for display. A generation comparison on an abbreviation is
    a comparison that can collide, and the collision would read as agreement.
    """

    container: str
    container_id: str
    image: str
    revision: str

    def to_dict(self) -> dict[str, str]:
        return {
            "container": self.container,
            "container_id": self.container_id,
            "image": self.image,
            "revision": self.revision,
        }

    def to_json(self) -> str:
        """One line, because this crosses a ``GITHUB_OUTPUT`` boundary."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @property
    def short(self) -> str:
        return (
            f"{self.container} id={self.container_id[:12] or '?'} "
            f"image={self.image[:24] or '?'} revision={self.revision[:12] or '?'}"
        )

    def same_generation_as(self, other: ModelLaneGeneration) -> bool:
        """Identity is the container and its image, never the revision alone.

        Two recreates from one build carry the same revision label and are two
        different generations; a probe that read one and a convergence guard
        that read the other have not agreed about anything.
        """
        return (
            self.container == other.container
            and self.container_id == other.container_id
            and self.image == other.image
        )


def parse_generation(payload: Any) -> ModelLaneGeneration:
    """Project a ``docker inspect`` object, or a re-read JSON record, onto the model.

    Split out from :func:`read_lane_generation` for the same reason
    ``check_dev_lane_staleness.parse_docker_inspect`` is: a test that rebuilds
    the shape by hand proves the comparison logic and nothing about whether the
    reader can read what docker actually returns.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            msg = f"generation record is not JSON: {exc}"
            raise ValueError(msg) from exc
    if isinstance(payload, list):  # some docker versions wrap in a list
        if not payload:
            msg = "docker inspect returned an empty list; no container was read"
            raise ValueError(msg)
        payload = payload[0]
    if not isinstance(payload, dict):
        msg = f"a generation record must be an object, got {type(payload).__name__}"
        raise ValueError(msg)

    # The already-projected form, as written to GITHUB_OUTPUT by the
    # convergence step and read back by the probe.
    if "container_id" in payload:
        missing = sorted(
            {"container", "container_id", "image", "revision"} - set(payload)
        )
        if missing:
            msg = f"generation record is missing required field(s) {missing}"
            raise ValueError(msg)
        return ModelLaneGeneration(
            container=str(payload["container"]),
            container_id=str(payload["container_id"]),
            image=str(payload["image"]),
            revision=str(payload["revision"]),
        )

    # The raw `docker inspect` form.
    labels = (payload.get("Config") or {}).get("Labels") or {}
    name = str(payload.get("Name", "")).lstrip("/")
    container_id = str(payload.get("Id", ""))
    if not container_id:
        msg = (
            "docker inspect returned no Id. A generation with no identity cannot "
            "be compared, and an empty string comparing equal to another empty "
            "string would read as agreement."
        )
        raise ValueError(msg)
    return ModelLaneGeneration(
        container=name,
        container_id=container_id,
        image=str(payload.get("Image", "")),
        revision=str(labels.get(REVISION_LABEL, "")),
    )


def read_lane_generation(container: str) -> ModelLaneGeneration:
    """Read one running container's identity. Read-only; never mutates."""
    result = subprocess.run(
        ["docker", "inspect", container, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"docker inspect {container} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
        raise ValueError(msg)
    return parse_generation(json.loads(result.stdout))


@dataclass(frozen=True)
class ModelSettleBudget:
    """What the probe was granted, what it was owed, and whether they agree.

    Constructed by ``scripts/ci/lane_settle_budget.py`` from the per-lane
    declaration in ``config/lab_pass_settle_budget.yaml`` and handed to
    ``probe-lane`` as one line of JSON, so this module needs no YAML reader.

    ``granted_seconds`` is ``min(declared, affordable)`` so the probe never
    overruns the job that has to WRITE the receipt -- an unwritten receipt is
    the one outcome worse than a failing one. ``sufficient`` is the fact that
    makes a short grant diagnosable instead of anonymous: before OMN-18436 a
    job that could afford only 180s of a 463-668s boot reported the shortfall
    as ``ready_effects: fail``, indistinguishable from an unhealthy lane.
    """

    lane: str
    declared_seconds: int
    affordable_seconds: int
    job_ceiling_seconds: int
    elapsed_seconds: int
    reserved_tail_seconds: int
    source: str

    @property
    def granted_seconds(self) -> int:
        return max(0, min(self.declared_seconds, self.affordable_seconds))

    @property
    def sufficient(self) -> bool:
        return self.affordable_seconds >= self.declared_seconds

    @property
    def shortfall_seconds(self) -> int:
        """How far short of the declaration this job's affordance fell."""
        return max(0, self.declared_seconds - self.affordable_seconds)

    @property
    def head(self) -> str:
        """Both numbers and the arithmetic behind them, on every verdict.

        Separate from :attr:`evidence` so a caller that reaches a different
        verdict about the SAME numbers still reports them identically.
        """
        return (
            f"declared {self.declared_seconds}s for lane {self.lane} "
            f"({self.source}); this job could afford {self.affordable_seconds}s "
            f"(ceiling {self.job_ceiling_seconds}s - elapsed "
            f"{self.elapsed_seconds}s - reserved tail "
            f"{self.reserved_tail_seconds}s); granted {self.granted_seconds}s"
        )

    @property
    def evidence(self) -> str:
        """The one line the ``settle_budget_sufficient`` check carries.

        Names BOTH numbers on both verdicts. A short grant reported only as
        "the lane was not ready" is indistinguishable from an unhealthy lane,
        which is the confusion this check exists to remove.
        """
        if self.sufficient:
            return f"{self.head} -- the job could afford the declared budget"
        return (
            f"{self.head} -- the job could NOT afford the declared budget, short by "
            f"{self.shortfall_seconds}s, so a lane that "
            "does not answer inside the grant has run out of CLOCK and has not "
            "been shown to be unhealthy"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "declared_seconds": self.declared_seconds,
            "affordable_seconds": self.affordable_seconds,
            "job_ceiling_seconds": self.job_ceiling_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "reserved_tail_seconds": self.reserved_tail_seconds,
            "source": self.source,
        }

    def to_json(self) -> str:
        """One line, because this crosses a shell boundary."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, body: str) -> ModelSettleBudget:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            msg = f"a settle budget must be an object, got {type(payload).__name__}"
            raise ValueError(msg)
        fields = {
            "lane",
            "declared_seconds",
            "affordable_seconds",
            "job_ceiling_seconds",
            "elapsed_seconds",
            "reserved_tail_seconds",
            "source",
        }
        missing = sorted(fields - set(payload))
        if missing:
            msg = f"settle budget is missing required field(s) {missing}"
            raise ValueError(msg)
        unknown = sorted(set(payload) - fields)
        if unknown:
            msg = f"unknown settle-budget field(s) {unknown}"
            raise ValueError(msg)
        return cls(
            lane=str(payload["lane"]),
            declared_seconds=int(payload["declared_seconds"]),
            affordable_seconds=int(payload["affordable_seconds"]),
            job_ceiling_seconds=int(payload["job_ceiling_seconds"]),
            elapsed_seconds=int(payload["elapsed_seconds"]),
            reserved_tail_seconds=int(payload["reserved_tail_seconds"]),
            source=str(payload["source"]),
        )


# ---------------------------------------------------------------------------
# probe-lane
# ---------------------------------------------------------------------------
#: Checks a ``compose-dev`` emitter proves from the runner, and the reason each
#: is here. The set is deliberately the subset that is provable from an HTTP
#: surface plus the job's own staleness guard; see ``PROBES_NOT_YET_WIRED``.
#:
#: ``projection_ready`` (OMN-18387): the ``omnimarket-projection-api``
#: container was a second gap in the same rule 24(a) automatic lab pass --
#: neither the runtime-rebuild classifier nor the deploy agent's up-target set
#: reached it, so a five-day-stale container passed every other check here.
#: Reachability from THIS runner is not assumed: confirmed live 2026-09-15
#: from inside the ``omninode-deploy-runner`` container on the same host that
#: already answers ``ready_main``/``ready_effects`` --
#: ``curl http://host.docker.internal:3002/projections`` -> ``HTTP_200``,
#: identically to ``http://host.docker.internal:8085/ready``. Same host, same
#: published-port mechanism, no additional isolation layer between the two.
#: ``node_inventory`` (OMN-18708): the lane's own answer to "which nodes, at
#: which contract bodies, are you running". It is on this list rather than in
#: PROBES_NOT_YET_WIRED because it needs nothing the other four do not -- the
#: introspection manifest is served by the same health server, on the same
#: port, as ``/ready`` and ``/health``.
MIGRATIONS_APPLIED_CHECK: Final[str] = "migrations_applied"
CONSUMER_GROUP_LAG_CHECK: Final[str] = "consumer_group_lag"
DELEGATION_GOLDEN_CHAIN_CHECK: Final[str] = "delegation_golden_chain"

COMPOSE_DEV_HTTP_CHECKS = (
    "ready_main",
    "ready_effects",
    "health_dimensions",
    "projection_ready",
    "node_inventory",
)

#: The three integration checks OMN-18866 wired, named here beside the HTTP set
#: because they are emitted by a different mechanism -- the docker socket and
#: the chain canary's receipt, not an HTTP GET -- and a reader counting the
#: checks on a receipt should be able to see where each came from.
#:
#: Each is emitted ONLY when the caller supplies its subject, on the same rule
#: the settle budget and the generation binding already follow: supplying the
#: input IS the claim, so an ad hoc read makes no claim about migrations, lag
#: or delegation instead of making an empty one.
COMPOSE_DEV_INTEGRATION_CHECKS = (
    MIGRATIONS_APPLIED_CHECK,
    CONSUMER_GROUP_LAG_CHECK,
    DELEGATION_GOLDEN_CHAIN_CHECK,
)

#: Named here rather than silently absent, so a reader can see what a
#: ``compose-dev`` receipt does NOT cover. Widening the set means adding the
#: probe AND the check name in ONE change -- never the name alone (rule 24).
#:
#: EMPTIED by OMN-18866. Its three entries were `migrations_applied`,
#: `consumer_group_lag` and `delegation_golden_chain`, all three of which are
#: now wired above. The tuple is kept rather than deleted because the NEXT
#: unwired probe needs somewhere honest to be declared, and an absent list is
#: how a gap stops being visible. It is empty because nothing is currently
#: known to be missing, which is a different statement from nothing being
#: missing.
PROBES_NOT_YET_WIRED: tuple[str, ...] = ()


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


def check_projections_ready(url: str, timeout_seconds: float) -> ModelLabPassCheck:
    """The projection API's ``/projections`` endpoint must answer 200 with a
    non-empty ``topics`` list whose entries match the declared omnimarket
    metadata shape (OMN-18387).

    The top-level key is ``topics``, not ``projections``. That is declared by
    omnimarket's projection API route, not inferred from one live response.
    What this asserts is exactly what ``omnimarket-projection-api`` staying
    stale would have failed: the container the deploy agent now reaches is
    actually serving projection metadata.
    """
    status, body = _http_get(url, timeout_seconds)
    if status != 200:
        return ModelLabPassCheck(
            name="projection_ready",
            ok=False,
            evidence=f"GET {url} -> {status} {_truncate(body)}",
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return ModelLabPassCheck(
            name="projection_ready",
            ok=False,
            evidence=f"GET {url} -> 200 but body is not JSON: {exc}",
        )
    topics = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(topics, list):
        return ModelLabPassCheck(
            name="projection_ready",
            ok=False,
            evidence=(
                f"GET {url} -> 200 but carries no 'topics' list; an "
                "absent list is not a serving projection API"
            ),
        )
    if not topics:
        return ModelLabPassCheck(
            name="projection_ready",
            ok=False,
            evidence=(
                f"GET {url} -> 200 but reports zero topics; an empty projection "
                "catalogue is not a serving projection API"
            ),
        )
    for index, entry in enumerate(topics):
        if not isinstance(entry, dict):
            return ModelLabPassCheck(
                name="projection_ready",
                ok=False,
                evidence=(f"GET {url} -> 200 but topics[{index}] is not an object"),
            )
        missing = sorted(PROJECTION_TOPIC_REQUIRED_FIELDS - set(entry))
        if missing:
            return ModelLabPassCheck(
                name="projection_ready",
                ok=False,
                evidence=(
                    f"GET {url} -> 200 but topics[{index}] is missing "
                    f"required field(s): {missing}"
                ),
            )
    return ModelLabPassCheck(
        name="projection_ready",
        ok=True,
        evidence=f"GET {url} -> 200, {len(topics)} topic(s) reported",
    )


#: Where the runtime actually mounts its dimension block, and the shape it
#: mounts. NOT a guess: ``omnibase_infra.runtime.health.runtime_health_block``
#: builds ``{"status", "observed_at", ..., "dimensions": [{name, status,
#: detail}, ...]}`` and ``omnibase_infra.services.health_checker`` assigns it to
#: ``details[RUNTIME_HEALTH_DETAIL_KEY]``, where that key is ``"runtime_health"``.
#:
#: The first revision of this check read a top-level ``dimensions`` OBJECT keyed
#: by dimension name. Nothing in this repository has ever produced that shape.
#: It went unnoticed because the probe never reached a lane -- the URL it used
#: resolved inside the runner container, so every call ended in ``Errno 111``
#: and the parser was never entered.
HEALTH_DETAILS_KEY = "details"
HEALTH_RUNTIME_BLOCK_KEY = "runtime_health"
HEALTHY_DIMENSION_WORDS = frozenset({"healthy", "ok", "pass", "up"})


def check_health_dimensions(url: str, timeout_seconds: float) -> ModelLabPassCheck:
    """Every dimension the health payload reports must be healthy.

    Fails closed on an unparseable body, on a payload carrying no dimension
    block, and on a dimension carrying no status: "we could not find an
    unhealthy dimension" is not the same statement as "every dimension is
    healthy", and only the second one is a check.
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

    def _absent(reason: str) -> ModelLabPassCheck:
        return ModelLabPassCheck(
            name="health_dimensions",
            ok=False,
            evidence=(
                f"GET {url} -> 200 but {reason}; an absent dimension set is not "
                "a healthy dimension set"
            ),
        )

    details = payload.get(HEALTH_DETAILS_KEY) if isinstance(payload, dict) else None
    if not isinstance(details, dict):
        return _absent(f"carries no {HEALTH_DETAILS_KEY!r} object")
    block = details.get(HEALTH_RUNTIME_BLOCK_KEY)
    if not isinstance(block, dict):
        return _absent(
            f"{HEALTH_DETAILS_KEY}.{HEALTH_RUNTIME_BLOCK_KEY} is absent or not "
            "an object"
        )
    dimensions = block.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return _absent(
            f"{HEALTH_DETAILS_KEY}.{HEALTH_RUNTIME_BLOCK_KEY}.dimensions is "
            "absent, empty, or not a list"
        )

    unhealthy = sorted(_unhealthy_dimension_names(dimensions))
    return ModelLabPassCheck(
        name="health_dimensions",
        ok=not unhealthy,
        evidence=(
            f"GET {url} -> 200, {len(dimensions)} dimensions, "
            + ("all healthy" if not unhealthy else f"unhealthy: {unhealthy}")
        ),
    )


#: OMN-18708. Named as a constant because the receipt model validates against
#: it: a passing check under this name must be backed by triples on the record.
NODE_INVENTORY_CHECK: Final[str] = "node_inventory"


@dataclass(frozen=True)
class ModelNodeInventoryProbe:
    """One read of the lane's introspection manifest: its verdict AND its data.

    The check and the triples travel together, from ONE read, deliberately.
    Two separate reads could disagree -- a lane recreated between them would
    make the receipt assert a check about one generation and carry the
    inventory of another -- and nothing on the record would show it.
    """

    check: ModelLabPassCheck
    triples: tuple[ModelNodeInventoryTriple, ...]


def check_node_inventory(url: str, timeout_seconds: float) -> ModelNodeInventoryProbe:
    """Read the lane's own node inventory off ``/v1/introspection/manifest``.

    OMN-18708 (AC3). The evidence for this check is what the LANE said about
    itself, never a restatement of what the build stamped: an emitter echoing
    its own input proves only that it can echo. What the lane serves is the
    auto-wiring manifest, whose ``contracts`` each carry ``name``,
    ``node_version`` and -- since this change -- ``contract_content_hash``,
    computed by the discovery pass over the contract file it parsed.

    The set here is the profile-filtered subset the lane WIRED, which is a
    subset of what the image ships and is deliberately not compared to the
    image label for equality. The third field is what ties the two together:
    both ends compute it with the one canonical hasher over the same file, so
    a contract whose body drifted is detectable from either.

    A contract served with no content hash FAILS the check rather than being
    dropped from the triples: a silently shorter inventory is a PASS narrower
    than it looks, which is the shape this module refuses everywhere else.
    """
    status, body = _http_get(url, timeout_seconds)
    if status != 200:
        return ModelNodeInventoryProbe(
            check=ModelLabPassCheck(
                name=NODE_INVENTORY_CHECK,
                ok=False,
                evidence=f"GET {url} -> {status} {_truncate(body)}",
            ),
            triples=(),
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return ModelNodeInventoryProbe(
            check=ModelLabPassCheck(
                name=NODE_INVENTORY_CHECK,
                ok=False,
                evidence=f"GET {url} -> 200 but body is not JSON: {exc}",
            ),
            triples=(),
        )
    if not isinstance(payload, dict):
        return ModelNodeInventoryProbe(
            check=ModelLabPassCheck(
                name=NODE_INVENTORY_CHECK,
                ok=False,
                evidence=(
                    f"GET {url} -> 200 but the manifest is a "
                    f"{type(payload).__name__}, not an object"
                ),
            ),
            triples=(),
        )
    contracts = payload.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        return ModelNodeInventoryProbe(
            check=ModelLabPassCheck(
                name=NODE_INVENTORY_CHECK,
                ok=False,
                evidence=(
                    f"GET {url} -> 200 but 'contracts' is absent or empty "
                    f"({_truncate(body)}). A lane that wired no contracts has "
                    "no inventory to report."
                ),
            ),
            triples=(),
        )

    rows: list[dict[str, str]] = []
    missing_hash: list[str] = []
    for contract in contracts:
        if not isinstance(contract, dict):
            missing_hash.append(f"<{type(contract).__name__}>")
            continue
        name = contract.get("name")
        content_hash = contract.get("contract_content_hash")
        if not isinstance(name, str) or not name:
            missing_hash.append("<unnamed>")
            continue
        if not isinstance(content_hash, str) or not content_hash:
            missing_hash.append(name)
            continue
        rows.append(
            {
                "name": name,
                "node_version": str(contract.get("node_version", "")),
                "contract_content_hash": content_hash,
            }
        )

    if missing_hash:
        return ModelNodeInventoryProbe(
            check=ModelLabPassCheck(
                name=NODE_INVENTORY_CHECK,
                ok=False,
                evidence=(
                    f"GET {url} -> 200, {len(contracts)} contract(s), but "
                    f"{len(missing_hash)} carry no contract_content_hash: "
                    f"{_truncate(', '.join(sorted(missing_hash)[:20]))}. A lane "
                    "that cannot name a contract's body cannot be checked for "
                    "drift in it."
                ),
            ),
            triples=(),
        )

    try:
        triples = parse_node_inventory(rows)
    except ValueError as exc:
        return ModelNodeInventoryProbe(
            check=ModelLabPassCheck(
                name=NODE_INVENTORY_CHECK,
                ok=False,
                evidence=f"GET {url} -> 200 but the inventory is unusable: {exc}",
            ),
            triples=(),
        )

    sample = ", ".join(f"{t.name}@{t.node_version}" for t in triples[:5])
    return ModelNodeInventoryProbe(
        check=ModelLabPassCheck(
            name=NODE_INVENTORY_CHECK,
            ok=True,
            evidence=(
                f"GET {url} -> 200; the lane reports {len(triples)} wired "
                f"contract(s), each with a content hash (e.g. {sample}"
                f"{', ...' if len(triples) > 5 else ''})"
            ),
        ),
        triples=triples,
    )


# ---------------------------------------------------------------------------
# OMN-18866 -- the three probes that were named in PROBES_NOT_YET_WIRED
# ---------------------------------------------------------------------------
#
# WHY THEY ARE WIRED NOW, and what changed. The list they came off carried the
# reason "each needs database or broker access the emitting job does not have
# today". That premise was true of the `omnibase-deploy` runner the emitting
# job used to run on. It is FALSE of the runner it runs on now: OMN-18602 moved
# `verify-lane-converged` to `[self-hosted, omnibase-verify, host-201]`, and
# every runner in `docker/docker-compose.runners.yml` bind-mounts the host's
# `/var/run/docker.sock` and resolves `host.docker.internal` to the docker
# bridge gateway where the lane publishes its ports.
#
# Re-measured 2026-09-20 from INSIDE `omninode-verify-runner-1`, with the
# positive/negative control pair rule 16 requires, because "cannot connect" and
# "connected, spoke no HTTP" are different facts and a probe that cannot tell
# them apart reports a lane outage for a protocol mismatch:
#
#     host.docker.internal:5436  (lane postgres)  -> curl exit 52  CONNECTED
#     host.docker.internal:19092 (lane broker)    -> curl exit 52  CONNECTED
#     host.docker.internal:8085  (runtime, DOWN)  -> curl exit 7   REFUSED
#
# So the reachability claim is not assumed; exit 7 is what unreachable actually
# looks like from this runner, and neither dependency produced it.
#
# HOW EACH ONE READS ITS SUBJECT, and why that choice:
#
# `migrations_applied` goes through the docker socket (`docker exec <pg> psql`)
#     rather than over TCP. Inside the container the connection is trusted, so
#     the probe needs NO network credential at all -- the smallest possible
#     authority for the question, and the same mechanism `read_lane_generation`
#     already uses for `docker inspect`.
#
# `consumer_group_lag` goes through `docker exec <broker> rpk`, because the
#     lane broker requires SASL and `rpk` inside the container is where the
#     existing stability-lane gate already runs it
#     (`scripts/runtime_build/verify_stability_refresh.py:check_cluster_health`).
#     The credential is the SAME org secret this workflow already hands its
#     sibling `trigger-rebuild` job, so no new credential surface is created.
#     The flag path is proven by its own negative control, recorded here
#     because it is the evidence that the probe will authenticate rather than
#     merely that it is spelled plausibly: with no credential the broker
#     answers "SASL required but not provided"; with a deliberately WRONG
#     credential it answers "SASL authentication failed". The second error is
#     the flags being read.
#
# `delegation_golden_chain` does NOT fire its own delegation. `chain-canary.yml`
#     already submits one through the deployed lane ingress and reads the
#     terminal back off the broker for its own correlation, FROM THIS SAME
#     RUNNER CLASS. A second dispatcher would be two delegation probes free to
#     disagree about what a delegation is, so this reads the canary's receipt
#     and grades it. The canary owns the dispatch; this owns the verdict.
#
# All three are READ-ONLY. None starts, stops, recreates or writes to anything.

#: The forward-migration ledger the lane keeps, and the column holding the id.
#: Measured on the live dev lane 2026-09-20: 90 rows, each shaped
#: ``docker/<NNN_name.sql>``, exactly equalling the 90 flat ``*.sql`` files the
#: delivered tree declares. A migration listed in ``skip-manifest.yaml`` still
#: gets a row (with checksum ``skip-manifest``), so a skip is recorded rather
#: than absent -- which is why presence, not execution, is the right question.
MIGRATION_LEDGER_QUERY: Final[str] = "select migration_id from public.schema_migrations"

#: How the delivered tree declares its forward migrations. FLAT and ``*.sql``
#: only, matching ``scripts/check_schema_fingerprint.py``'s own glob
#: ("The glob is intentionally flat (non-recursive)"). The two ``*.sh`` files
#: beside them are bootstrap helpers the runner does not ledger, and the
#: subdirectories are node-owned streams with their own ledgers.
MIGRATION_ID_PREFIX: Final[str] = "docker/"


class CommandRunner(Protocol):
    """A ``subprocess.run``-shaped callable, so the probes are testable.

    Declared rather than passing ``subprocess.run`` implicitly because each
    probe below must be exercised by a NEGATIVE control -- a test that drives
    it against a known-bad reading and asserts it fails. A probe no test has
    ever made fail is a probe never proven capable of failing.
    """

    def __call__(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]: ...


def _run_read_only(
    argv: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a fixed-argv, no-shell, read-only command."""
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


@dataclass(frozen=True)
class ModelMigrationLedger:
    """Where the lane keeps its applied-migration ledger."""

    container: str
    database: str
    psql_user: str = "postgres"

    def __post_init__(self) -> None:
        for field_name in ("container", "database", "psql_user"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                msg = f"migration ledger {field_name} is required and must be non-empty"
                raise ValueError(msg)


def declared_forward_migrations(migrations_dir: Path) -> tuple[str, ...]:
    """The migration ids the DELIVERED TREE declares, as the ledger spells them.

    Raises rather than returning an empty tuple when the directory is absent:
    an enumerator that answers "nothing is declared" for a missing directory
    hands the caller a vacuous comparison that passes against any lane at all.
    """
    if not migrations_dir.is_dir():
        msg = f"forward-migration directory not found: {migrations_dir}"
        raise ValueError(msg)
    names = sorted(path.name for path in migrations_dir.glob("*.sql"))
    if not names:
        msg = (
            f"no *.sql forward migrations under {migrations_dir}; refusing to "
            "compare a lane against an empty declaration"
        )
        raise ValueError(msg)
    return tuple(f"{MIGRATION_ID_PREFIX}{name}" for name in names)


def read_applied_migrations(
    ledger: ModelMigrationLedger,
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[str, ...]:
    """Read the lane's applied-migration ids through the docker socket."""
    run = runner or _run_read_only
    result = run(
        [
            "docker",
            "exec",
            ledger.container,
            "psql",
            "-U",
            ledger.psql_user,
            "-d",
            ledger.database,
            "-tAc",
            MIGRATION_LEDGER_QUERY,
        ],
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        msg = (
            f"reading {ledger.database}.public.schema_migrations via "
            f"{ledger.container} failed (exit {result.returncode}): "
            f"{_truncate((result.stderr or '').strip())}"
        )
        raise ValueError(msg)
    rows = [line.strip() for line in (result.stdout or "").splitlines()]
    return tuple(row for row in rows if row)


def check_migrations_applied(
    declared: Sequence[str],
    ledger: ModelMigrationLedger,
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 30.0,
) -> ModelLabPassCheck:
    """Every migration the delivered tree declares has a row on the lane.

    A lane can legitimately be AHEAD of the declaration -- OMN-18388 already
    accepts a lane converged onto a DESCENDANT of the merge sha, and a
    descendant may carry migrations this tree does not. So an extra row is
    reported and does not fail. A MISSING row is the failure: it is a lane
    serving an image whose schema was never applied, which every readiness
    endpoint answers 200 straight through.
    """
    if not declared:
        return ModelLabPassCheck.indeterminate_check(
            MIGRATIONS_APPLIED_CHECK,
            "the delivered tree declared no forward migrations, so there is "
            "nothing to compare; refusing to report a pass on a vacuous set",
        )
    try:
        applied = read_applied_migrations(
            ledger, runner=runner, timeout_seconds=timeout_seconds
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return ModelLabPassCheck.indeterminate_check(
            MIGRATIONS_APPLIED_CHECK,
            f"could not read the lane's migration ledger: {type(exc).__name__}: {exc}",
        )
    if not applied:
        return ModelLabPassCheck.indeterminate_check(
            MIGRATIONS_APPLIED_CHECK,
            f"{ledger.database}.public.schema_migrations returned zero rows via "
            f"{ledger.container}; an empty ledger is a reading this probe cannot "
            "tell apart from a table it failed to query, so it is not a pass",
        )
    declared_set = set(declared)
    applied_set = set(applied)
    missing = sorted(declared_set - applied_set)
    extra = sorted(applied_set - declared_set)
    evidence = (
        f"{len(declared_set)} declared, {len(applied_set)} recorded applied on "
        f"{ledger.container}:{ledger.database}"
    )
    if extra:
        evidence += (
            f"; {len(extra)} recorded but not declared here "
            f"({', '.join(extra[:3])}{', ...' if len(extra) > 3 else ''}) "
            "-- a lane ahead of this tree, which does not fail this check"
        )
    if missing:
        return ModelLabPassCheck(
            name=MIGRATIONS_APPLIED_CHECK,
            ok=False,
            evidence=(
                f"{evidence}; {len(missing)} DECLARED BUT NOT APPLIED: "
                f"{', '.join(missing[:5])}{', ...' if len(missing) > 5 else ''}"
            ),
        )
    return ModelLabPassCheck(
        name=MIGRATIONS_APPLIED_CHECK, ok=True, evidence=f"{evidence}; none missing"
    )


@dataclass(frozen=True)
class ModelBrokerAccess:
    """How to reach the lane broker from inside its own container.

    ``sasl_username`` / ``sasl_password`` come from the job environment, never
    from argv, and are never rendered into evidence. What evidence records is
    the VARIABLE NAME and whether authentication succeeded, which is the part a
    reader can act on.
    """

    container: str
    brokers: str
    sasl_mechanism: str = ""
    sasl_username: str = ""
    sasl_password: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.container, str) or not self.container:
            msg = "broker container is required and must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.brokers, str) or not self.brokers:
            msg = "broker address is required and must be non-empty"
            raise ValueError(msg)
        supplied = [
            bool(self.sasl_mechanism),
            bool(self.sasl_username),
            bool(self.sasl_password),
        ]
        if any(supplied) and not all(supplied):
            msg = (
                "SASL is all-or-nothing: mechanism, username and password must "
                "be supplied together. A partial credential would be silently "
                "dropped by rpk and read as an unauthenticated probe."
            )
            raise ValueError(msg)

    @property
    def authenticated(self) -> bool:
        return bool(self.sasl_mechanism)

    def rpk_flags(self) -> list[str]:
        flags = ["-X", f"brokers={self.brokers}"]
        if self.authenticated:
            flags += [
                "-X",
                f"user={self.sasl_username}",
                "-X",
                f"pass={self.sasl_password}",
                "-X",
                f"sasl.mechanism={self.sasl_mechanism}",
            ]
        return flags


def read_group_total_lag(
    access: ModelBrokerAccess,
    group: str,
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 60.0,
) -> int:
    """Read one consumer group's TOTAL-LAG off ``rpk group describe``.

    The parse mirrors ``scripts/runtime_build/declared_consumer_groups.py``'s
    ``parse_group_describe`` -- a ``TOTAL-LAG <n>`` line, matched on the label
    rather than a column offset, because rpk pads that table differently
    between versions. One parse rule for one output format; two would be two
    declarations free to disagree.
    """
    run = runner or _run_read_only
    result = run(
        ["docker", "exec", access.container, "rpk", "group", "describe", group]
        + access.rpk_flags(),
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        msg = (
            f"rpk group describe {group} failed (exit {result.returncode}): "
            f"{_truncate((result.stderr or '').strip())}"
        )
        raise ValueError(msg)
    for line in (result.stdout or "").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "TOTAL-LAG":
            try:
                return int(fields[1])
            except ValueError as exc:
                msg = f"TOTAL-LAG for {group} is not an integer: {fields[1]!r}"
                raise ValueError(msg) from exc
    msg = f"rpk group describe {group} printed no TOTAL-LAG line"
    raise ValueError(msg)


def check_consumer_group_lag(
    access: ModelBrokerAccess,
    groups: Sequence[str],
    *,
    max_lag: int,
    first_sample: Mapping[str, int] | None = None,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 60.0,
) -> ModelLabPassCheck:
    """Declared groups are under their lag bound AND not growing.

    TWO conditions, and the second is the one that matters. A bound alone
    cannot see the failure this probe exists for: the savings writer sat at lag
    498 for NINE DAYS (OMN-18851), a number a generous bound admits and a tight
    bound would have flagged on every healthy busy group as well. Growth across
    two samples separates a backlog being worked from a consumer that has
    stopped, and it is the only one of the two that is scale-free.

    ``first_sample`` is the earlier reading, taken by the caller before the
    settle wait so the two samples straddle real time rather than being two
    reads a millisecond apart. Absent, the probe reports the bound only and
    says so in its evidence rather than implying it checked growth.
    """
    if not groups:
        return ModelLabPassCheck.indeterminate_check(
            CONSUMER_GROUP_LAG_CHECK,
            "no consumer groups were declared for this lane, so there is "
            "nothing to measure; an empty declaration is not a healthy lane",
        )
    second: dict[str, int] = {}
    unreadable: list[str] = []
    for group in groups:
        try:
            second[group] = read_group_total_lag(
                access, group, runner=runner, timeout_seconds=timeout_seconds
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            unreadable.append(
                f"{group} ({type(exc).__name__}: {_truncate(str(exc), 90)})"
            )
    if unreadable:
        return ModelLabPassCheck.indeterminate_check(
            CONSUMER_GROUP_LAG_CHECK,
            f"{len(unreadable)} of {len(groups)} declared group(s) could not be "
            f"read via {access.container} "
            f"(authenticated={access.authenticated}): {'; '.join(unreadable[:3])}"
            f"{', ...' if len(unreadable) > 3 else ''}",
        )
    over_bound = sorted(g for g, lag in second.items() if lag > max_lag)
    growing: list[str] = []
    if first_sample is not None:
        growing = sorted(
            g
            for g, lag in second.items()
            if g in first_sample and lag > first_sample[g]
        )
    worst = max(second.values())
    evidence = (
        f"{len(second)} declared group(s) read via {access.container}; "
        f"max TOTAL-LAG {worst} against bound {max_lag}; "
        + (
            f"growth measured against an earlier sample of {len(first_sample)} group(s)"
            if first_sample is not None
            else "NO earlier sample supplied, so growth was NOT measured and "
            "this check covers the bound only"
        )
    )
    problems: list[str] = []
    if over_bound:
        problems.append(
            f"over bound: {', '.join(f'{g}={second[g]}' for g in over_bound[:5])}"
        )
    if growing:
        problems.append(
            "GROWING across two samples: "
            + ", ".join(
                f"{g} {first_sample[g]}->{second[g]}"  # type: ignore[index]
                for g in growing[:5]
            )
        )
    if problems:
        return ModelLabPassCheck(
            name=CONSUMER_GROUP_LAG_CHECK,
            ok=False,
            evidence=f"{evidence}; {'; '.join(problems)}",
        )
    return ModelLabPassCheck(
        name=CONSUMER_GROUP_LAG_CHECK,
        ok=True,
        evidence=f"{evidence}; none over bound"
        + ("" if first_sample is None else ", none growing"),
    )


def parse_group_list_argument(raw: str) -> tuple[str, ...]:
    """Split a comma- or newline-separated group list, dropping blanks."""
    if not raw:
        return ()
    parts = [chunk.strip() for chunk in raw.replace("\n", ",").split(",")]
    return tuple(part for part in parts if part)


def sample_group_lag(
    access: ModelBrokerAccess,
    groups: Sequence[str],
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, int]:
    """One reading of every declared group's lag, for the EARLIER sample.

    A group that cannot be read is OMITTED rather than recorded as zero. The
    growth comparison then simply has no baseline for it, which is honest; a
    zero would manufacture a baseline that makes any later reading look like
    growth.
    """
    sample: dict[str, int] = {}
    for group in groups:
        try:
            sample[group] = read_group_total_lag(
                access, group, runner=runner, timeout_seconds=timeout_seconds
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return sample


def load_lag_sample(path: Path | None) -> dict[str, int] | None:
    """Read an earlier lag sample, or None when the caller supplied none.

    An unreadable or malformed file returns None rather than raising: the
    growth arm is then not measured and ``check_consumer_group_lag`` SAYS it
    was not measured in its own evidence. Silently treating a broken baseline
    as an empty one would let the check claim it checked growth against
    nothing.
    """
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        str(key): int(value)
        for key, value in payload.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def check_delegation_golden_chain(receipt_path: Path) -> ModelLabPassCheck:
    """Grade the chain canary's receipt for a delegation fired at THIS lane.

    WHAT THE BAR IS, and what it deliberately is not. The canary reports two
    different claims: whether the delegation it submitted reached a terminal on
    the bus for its own correlation (``success``), and whether all five declared
    chain links were PROVEN (``chain_proof_complete``). This check is bound to
    the first.

    The second is not the bar because link 5 has no leg in any probe today --
    OMN-16964 is the open ticket that says so in its own title -- so requiring
    it would make this check permanently red and, within a week, permanently
    ignored. The link counts ARE carried into the evidence, so a reader can see
    the weaker claim rather than having to assume the stronger one. That is the
    same distinction the canary's own summary step draws, and for the same
    reason: a green probe verdict is not a five-link chain proof.
    """
    if not receipt_path.exists():
        return ModelLabPassCheck.indeterminate_check(
            DELEGATION_GOLDEN_CHAIN_CHECK,
            f"no chain-canary receipt at {receipt_path}: the dispatch did not "
            "run, or died before it could report. That is a fact about this "
            "probe's own execution path, not about the lane, so it is "
            "indeterminate rather than a lane failure",
        )
    raw = receipt_path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return ModelLabPassCheck.indeterminate_check(
            DELEGATION_GOLDEN_CHAIN_CHECK,
            f"the chain-canary receipt at {receipt_path} is empty",
        )
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ModelLabPassCheck.indeterminate_check(
            DELEGATION_GOLDEN_CHAIN_CHECK,
            f"the chain-canary receipt at {receipt_path} is not valid JSON: {exc}",
        )
    if not isinstance(envelope, dict):
        return ModelLabPassCheck.indeterminate_check(
            DELEGATION_GOLDEN_CHAIN_CHECK,
            "the chain-canary receipt is not a JSON object",
        )
    result = envelope.get("result")
    if not isinstance(result, dict):
        return ModelLabPassCheck.indeterminate_check(
            DELEGATION_GOLDEN_CHAIN_CHECK,
            "the chain-canary receipt carries no 'result' object, so its "
            "verdict cannot be read",
        )
    success = result.get("success")
    if not isinstance(success, bool):
        return ModelLabPassCheck.indeterminate_check(
            DELEGATION_GOLDEN_CHAIN_CHECK,
            "the chain-canary receipt's 'success' is missing or not a boolean; "
            "a verdict that has to be guessed at is not a verdict",
        )
    verdict = str(result.get("verdict") or "<unnamed>")
    detail = _truncate(str(result.get("detail") or ""), 160)
    proven = result.get("links_proven")
    total = result.get("links_total")
    links = (
        f"; chain links proven {proven} of {total} "
        "(NOT the bar for this check -- see OMN-16964)"
        if isinstance(proven, int) and isinstance(total, int)
        else "; the receipt reported no link counts"
    )
    evidence = (
        f"one live delegation through the deployed lane ingress: "
        f"verdict={verdict}{links}"
        f"{'; ' + detail if detail else ''}"
    )
    return ModelLabPassCheck(
        name=DELEGATION_GOLDEN_CHAIN_CHECK, ok=success, evidence=evidence
    )


def _unhealthy_dimension_names(dimensions: list[Any]) -> list[str]:
    """Name every dimension that is not affirmatively healthy.

    A malformed entry counts as unhealthy rather than being skipped: an entry
    the reader cannot judge is exactly the case a silent ``continue`` would
    turn into a green run.
    """
    names: list[str] = []
    for index, entry in enumerate(dimensions):
        if not isinstance(entry, dict):
            names.append(f"<dimension[{index}] is not an object>")
            continue
        name = str(entry.get("name") or f"<dimension[{index}] has no name>")
        status = entry.get("status")
        if not isinstance(status, str) or status.lower() not in (
            HEALTHY_DIMENSION_WORDS
        ):
            names.append(name)
    return names


#: How long to sleep between readiness polls while the lane finishes coming up.
SETTLE_POLL_SECONDS = 10.0


#: The name of the check that says the job could afford the DECLARED settle
#: budget. Named rather than folded into the readiness evidence because the two
#: facts have different remedies: a short grant is a job-ceiling problem and an
#: unready lane is a lane problem, and before OMN-18436 both arrived as
#: ``ready_effects: fail``.
SETTLE_BUDGET_CHECK = "settle_budget_sufficient"

#: The name of the check that says the lane did not answer inside the grant.
#: Distinct from the per-endpoint checks, which say WHAT did not answer; this
#: one says the wait ran out, and carries how long it actually waited.
SETTLE_TIMEOUT_CHECK = "timed_out_before_ready"

#: The name of the check that binds these HTTP reads to the container
#: generation the convergence guard observed.
GENERATION_CHECK = "probe_generation_bound"


@dataclass(frozen=True)
class ModelSettleOutcome:
    """What the settle wait actually did, rather than only what it concluded.

    ``waited_seconds`` is the MEASURED boot when ``ready`` is true, which is the
    number that sizes the declaration in
    ``config/lab_pass_settle_budget.yaml``. Recording it on the failing path too
    is the point: "still not ready after 180s" and "still not ready after 900s"
    are different findings and the old string made them one.
    """

    ready: bool
    waited_seconds: float
    granted_seconds: float
    pending: tuple[str, ...]

    @property
    def phrase(self) -> str:
        if self.granted_seconds <= 0:
            return "no settle budget remained, so this is a first-look read"
        if self.ready:
            return (
                f"lane ready after {self.waited_seconds:.0f}s of a "
                f"{self.granted_seconds:.0f}s settle budget"
            )
        return (
            f"still not ready after the full {self.granted_seconds:.0f}s "
            f"settle budget: {sorted(self.pending)}"
        )


def wait_for_lane_ready(
    ready_urls: Sequence[str], timeout_seconds: float, settle_timeout_seconds: float
) -> ModelSettleOutcome:
    """Poll one readiness endpoint until it answers 200, or the budget expires.

    Returns what happened, structured. Every check below appends
    :attr:`ModelSettleOutcome.phrase` to its own evidence, and the probe turns
    the same outcome into the ``timed_out_before_ready`` check. It is
    deliberately a WAIT and never a verdict: the checks still run afterwards and
    still fail on a lane that never came up.

    WHY THIS EXISTS. ``check_dev_lane_staleness.py`` returns the moment the
    RUNNING container carries the expected ``org.opencontainers.image.revision``
    label, and that label is set when the container is created, not when the
    runtime has bound its port. So the instant convergence succeeds is the
    instant the lane is LEAST able to answer. Measured on run 34478680748:
    convergence at ``13:12:28.5442420Z``, probe at ``13:12:28.7181102Z``, first
    ``Errno 111`` at ``13:12:28.8260884Z`` -- 174 milliseconds, straight into a
    compose recreate.

    The consequence was perverse and is the reason this is not cosmetic: the
    HTTP checks passed only when ``deployed_revision`` FAILED (a stale lane
    serving happily) and failed exactly when it SUCCEEDED, so a ``compose-dev``
    receipt could not be a PASS by construction.

    WHERE THE BUDGET COMES FROM, corrected by OMN-18436. It is still supplied by
    the caller, but the caller no longer computes it as a job-ceiling remainder.
    It is DECLARED per lane in ``config/lab_pass_settle_budget.yaml``, bounded
    below by the worst boot this lane has been observed to take and above by the
    lane's own compose ``start_period``, and the emitting job's ceiling is
    derived from it. The old remainder made the lane's boot budget a function of
    how long the deploy agent queue happened to be: a convergence slower than
    about twenty minutes left 180s against a 463-668s boot, and the receipt
    FAILED on timing alone. The earlier note here -- that the compose
    ``start_period`` is 10s and measures something else -- was reading the
    ``x-healthcheck-defaults`` anchor; the runtime services override it to
    ``1800s`` each, which is what the upper bound is taken from.
    """
    started = time.monotonic()
    if settle_timeout_seconds <= 0:
        return ModelSettleOutcome(
            ready=False,
            waited_seconds=0.0,
            granted_seconds=0.0,
            pending=tuple(ready_urls),
        )
    deadline = started + settle_timeout_seconds
    while True:
        pending = [
            url for url in ready_urls if _http_get(url, timeout_seconds)[0] != 200
        ]
        waited = time.monotonic() - started
        if not pending:
            return ModelSettleOutcome(
                ready=True,
                waited_seconds=waited,
                granted_seconds=settle_timeout_seconds,
                pending=(),
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ModelSettleOutcome(
                ready=False,
                waited_seconds=waited,
                granted_seconds=settle_timeout_seconds,
                pending=tuple(pending),
            )
        time.sleep(min(SETTLE_POLL_SECONDS, remaining))


def settle_budget_check(
    budget: ModelSettleBudget, outcome: ModelSettleOutcome | None = None
) -> ModelLabPassCheck:
    """Record whether the job could afford the lane's declared settle budget.

    Emitted on BOTH verdicts. A check that appeared only when the budget was
    short would be indistinguishable from one that was never run, which is the
    shape this module's receipt contract refuses everywhere else.

    A SHORTFALL THAT NEVER BIT IS NOT A FAILURE (OMN-18436). The reason a short
    grant fails is stated in the evidence itself: a lane that does not answer
    inside it has run out of CLOCK and has not been shown to be unhealthy. That
    reason is conditional, and until this change the verdict was not. Receipt
    ``63cea2aa`` FAILed on a grant 2 seconds short of the declaration while
    every readiness endpoint had answered after 101 of the 898 granted seconds
    -- a receipt failing on timing alone, on a lane it had just proven healthy,
    which is the exact shape this ticket exists to remove, and rule 24(b) then
    refuses that sha for staging.

    So when ``outcome`` says the lane answered inside the grant, the shortfall
    could not have changed the answer and the check passes, still naming both
    numbers so the short ceiling stays diagnosable from the artifact alone.
    ``outcome=None`` is a caller that knows nothing about the wait and gets the
    bare affordability verdict: an absent outcome is not evidence that the lane
    answered.
    """
    if budget.sufficient or outcome is None or not outcome.ready:
        return ModelLabPassCheck(
            name=SETTLE_BUDGET_CHECK, ok=budget.sufficient, evidence=budget.evidence
        )
    return ModelLabPassCheck(
        name=SETTLE_BUDGET_CHECK,
        ok=True,
        evidence=(
            f"{budget.head} -- the job could NOT afford the declared budget, "
            f"short by {budget.shortfall_seconds}s, but the lane answered every "
            f"readiness endpoint after {outcome.waited_seconds:.0f}s, inside the "
            "grant, so the shortfall could not have changed the answer and says "
            "nothing about the lane. The ceiling is still short and this line is "
            "the durable record of it."
        ),
    )


def settle_timeout_check(outcome: ModelSettleOutcome) -> ModelLabPassCheck:
    """Record a boot that ran out of budget, with the measured wait.

    Separate from the per-endpoint checks on purpose. Those say WHICH endpoint
    did not answer; this says the wait itself expired, and carries the number
    that sizes the declaration. A reader of a FAIL receipt could previously not
    tell "the lane is unhealthy" from "the job ran out of clock eight seconds
    before the lane bound its port".
    """
    if outcome.ready:
        return ModelLabPassCheck(
            name=SETTLE_TIMEOUT_CHECK,
            ok=True,
            evidence=(
                f"lane answered every readiness endpoint after "
                f"{outcome.waited_seconds:.0f}s of a "
                f"{outcome.granted_seconds:.0f}s grant"
            ),
        )
    return ModelLabPassCheck(
        name=SETTLE_TIMEOUT_CHECK,
        ok=False,
        evidence=(
            f"timed out before ready: waited {outcome.waited_seconds:.0f}s of a "
            f"{outcome.granted_seconds:.0f}s grant with "
            f"{sorted(outcome.pending)} still not answering 200"
        ),
    )


def generation_check(
    expected: ModelLaneGeneration | None,
    observed: ModelLaneGeneration | None,
    read_error: str = "",
) -> ModelLabPassCheck:
    """Bind these HTTP reads to the container generation convergence observed.

    Four indeterminate cases, four evidence strings, one verdict. A probe that
    cannot prove which container answered has not proven anything about the sha
    this receipt is keyed by, and the greens it recorded are about some other
    lane (receipt ``4853e0e1``: ``deployed_revision`` FAIL with ``ready_effects``
    TRUE -- two true statements about two different containers).
    """
    if expected is None:
        return ModelLabPassCheck(
            name=GENERATION_CHECK,
            ok=False,
            evidence=(
                "the convergence step published no container generation, so "
                "these reads are not bound to any container and cannot be "
                "evidence for this sha"
            ),
        )
    if observed is None:
        return ModelLabPassCheck(
            name=GENERATION_CHECK,
            ok=False,
            evidence=(
                f"expected generation {expected.short}, but the container could "
                f"not be re-read at probe time ({read_error or 'no reason given'})"
            ),
        )
    if not expected.same_generation_as(observed):
        return ModelLabPassCheck(
            name=GENERATION_CHECK,
            ok=False,
            evidence=(
                f"the lane was recreated between convergence and this probe: "
                f"convergence read {expected.short}, the probe read "
                f"{observed.short}. These HTTP reads are about a different "
                "container generation than the one convergence verified"
            ),
        )
    return ModelLabPassCheck(
        name=GENERATION_CHECK,
        ok=True,
        evidence=(f"probe bound to the generation convergence read: {observed.short}"),
    )


def probe_compose_dev(
    main_url: str,
    effects_url: str,
    timeout_seconds: float,
    settle_timeout_seconds: float,
    projection_url: str,
    *,
    budget: ModelSettleBudget | None = None,
    expected_generation: ModelLaneGeneration | None = None,
    generation_container: str | None = None,
    node_inventory_probe: ModelNodeInventoryProbe | None = None,
    declared_migrations: Sequence[str] | None = None,
    migration_ledger: ModelMigrationLedger | None = None,
    broker_access: ModelBrokerAccess | None = None,
    declared_consumer_groups: Sequence[str] | None = None,
    max_consumer_lag: int = 0,
    first_lag_sample: Mapping[str, int] | None = None,
    chain_canary_receipt: Path | None = None,
    runner: CommandRunner | None = None,
) -> list[ModelLabPassCheck]:
    """The read-only probes the ``.201`` dev lane emitter runs.

    A convergence success hands us a lane that has just been recreated, so the
    probes wait for it to come up before reading it. See
    :func:`wait_for_lane_ready` for the measurement that made this necessary.

    ``budget`` and ``generation_container`` are supplied by the emitting job and
    omitted by an ad hoc read. Their PRESENCE is the caller asserting a claim --
    "this probe had a declared budget", "this probe is bound to a generation" --
    so an ad hoc read makes neither claim and emits neither check, while the job
    makes both and is held to both. ``generation_container`` with
    ``expected_generation=None`` is the job saying convergence produced no
    record, which is a failure and not an absence.

    ``node_inventory_probe`` follows the same rule (OMN-18708): it is passed in
    already read, rather than read here, so the check and the triples the
    receipt carries come from ONE read of the manifest. A caller that did not
    probe the manifest passes nothing and makes no claim about the lane's
    inventory -- it does not emit an empty one.
    """
    # BOTH readiness endpoints, not just main. Measured on the .201 lane
    # 2026-09-10: at 12:36:29Z omninode-runtime read "Up 3 minutes (health:
    # starting)" while omninode-runtime-effects was still "Created", and at
    # 12:37:29Z main was healthy while effects had been up 44 seconds. Waiting
    # on main alone would clear the race for ready_main and leave it for
    # ready_effects. projection-api joins the same wait for the same reason
    # (OMN-18387): it is its own container with its own recreate timing, and
    # it carries its own ``/ready`` endpoint (confirmed live 2026-09-15).
    outcome = wait_for_lane_ready(
        [
            f"{main_url.rstrip('/')}/ready",
            f"{effects_url.rstrip('/')}/ready",
            f"{projection_url.rstrip('/')}/ready",
        ],
        timeout_seconds,
        settle_timeout_seconds,
    )
    settle = outcome.phrase
    checks = [
        check_ready("ready_main", f"{main_url.rstrip('/')}/ready", timeout_seconds),
        check_ready(
            "ready_effects", f"{effects_url.rstrip('/')}/ready", timeout_seconds
        ),
        check_health_dimensions(f"{main_url.rstrip('/')}/health", timeout_seconds),
        check_projections_ready(
            f"{projection_url.rstrip('/')}/projections", timeout_seconds
        ),
    ]
    # Every check carries what the probe waited for, passing ones included: a
    # green read taken with no settle budget is a different fact from a green
    # read taken after the lane reported itself up, and the receipt should not
    # make them look identical.
    annotated = [
        ModelLabPassCheck(
            name=check.name, ok=check.ok, evidence=f"{check.evidence} [{settle}]"
        )
        for check in checks
    ]
    if node_inventory_probe is not None:
        # Annotated with the same settle phrase as the four HTTP checks above:
        # an inventory read before the lane reported itself up is a different
        # fact from one read after, and the receipt should not hide that.
        annotated.append(
            ModelLabPassCheck(
                name=node_inventory_probe.check.name,
                ok=node_inventory_probe.check.ok,
                evidence=f"{node_inventory_probe.check.evidence} [{settle}]",
            )
        )
    if budget is not None:
        annotated.append(settle_budget_check(budget, outcome=outcome))
    if settle_timeout_seconds > 0:
        annotated.append(settle_timeout_check(outcome))
    if generation_container is not None:
        observed: ModelLaneGeneration | None = None
        read_error = ""
        try:
            observed = read_lane_generation(generation_container)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            read_error = f"{type(exc).__name__}: {exc}"
        annotated.append(generation_check(expected_generation, observed, read_error))

    # OMN-18866. The three integration checks. Each is appended only when its
    # SUBJECT was supplied, on the same rule the budget and generation checks
    # above already follow: the input IS the claim. An ad hoc `probe-lane` that
    # passes none of them emits none of them, and so asserts nothing about
    # migrations, lag or delegation rather than asserting an empty result.
    #
    # These are NOT annotated with the settle phrase. The four HTTP checks read
    # a lane that may still be coming up, so when they were read is part of
    # what they mean. A migration ledger, a consumer group's lag and a
    # delegation's terminal are not properties of the lane's boot, and stamping
    # a settle phrase onto them would imply a relationship that is not there.
    if migration_ledger is not None:
        annotated.append(
            check_migrations_applied(
                declared_migrations or (), migration_ledger, runner=runner
            )
        )
    if broker_access is not None:
        annotated.append(
            check_consumer_group_lag(
                broker_access,
                declared_consumer_groups or (),
                max_lag=max_consumer_lag,
                first_sample=first_lag_sample,
                runner=runner,
            )
        )
    if chain_canary_receipt is not None:
        annotated.append(check_delegation_golden_chain(chain_canary_receipt))
    return annotated


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------
#: The three spellings ``--check`` accepts, and the check each builds.
#: ``indeterminate`` (OMN-18573) is a check that could not be ESTABLISHED. It is
#: not a pass: ``ok`` is false, so the receipt is still non-PASS and the
#: delivery gate still refuses the sha. What it buys is that the refusal says
#: the budget could not be resolved rather than asserting the lane failed.
_CHECK_VERDICTS: Final[frozenset[str]] = frozenset({"ok", "fail", "indeterminate"})


#: A deploy-agent correlation id is a uuid. The publishing job writes an EMPTY
#: value when it published no command at all, which a workflow expression
#: delivers as an empty string rather than as an absent flag -- so the two
#: cases the field must tell apart arrive through the same argv.
_CORRELATION_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def parse_agent_command_id(raw: str | None) -> str | None:
    """Normalise ``--agent-command-id`` into the field's two legal states.

    OMN-18573. The field existed and every compose-dev receipt carried ``null``
    in it, because no emitter passed the flag. The correlation id was knowable
    the whole time -- the publishing job had it -- and the readback that found
    this had the id legible in the ``deployed_revision`` EVIDENCE STRING while
    the typed field a machine reads sat empty. A fact present in prose and
    absent from the contract is a fact no consumer can use.

    Empty or whitespace-only means the emitter genuinely has none: the run
    published no command (the ``onex-lab`` boot gate never does), and ``None``
    is the honest record of that. It is NOT an error, because refusing here
    would lose the whole receipt, and an unwritten receipt is the one outcome
    worse than a failing one.

    Anything else must be a uuid. A non-uuid is REFUSED rather than stored: a
    field carrying a value no agent job can be looked up by is worse than the
    null it replaced, since a reader would believe it.
    """
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    if not _CORRELATION_ID_RE.match(candidate):
        msg = (
            f"agent_command_id {raw!r} is not a uuid. A correlation id no agent "
            "job can be resolved by is worse than an absent one, because a "
            "reader would believe it."
        )
        raise ValueError(msg)
    return candidate


def parse_check_argument(raw: str) -> ModelLabPassCheck:
    """Parse ``name:ok|fail|indeterminate:evidence`` from the command line.

    Split on the first two colons only, so evidence may contain colons (URLs
    and timestamps both do).
    """
    parts = raw.split(":", 2)
    if len(parts) != 3:
        msg = (
            f"--check {raw!r} is not 'name:ok|fail|indeterminate:evidence'. All "
            "three fields are required; a check with no evidence is not a check."
        )
        raise ValueError(msg)
    name, verdict, evidence = parts
    verdict_normalised = verdict.strip().lower()
    if verdict_normalised not in _CHECK_VERDICTS:
        msg = (
            f"--check {raw!r}: verdict must be one of "
            f"{sorted(_CHECK_VERDICTS)}, got {verdict!r}."
        )
        raise ValueError(msg)
    if verdict_normalised == "indeterminate":
        return ModelLabPassCheck.indeterminate_check(
            name=name.strip(), evidence=evidence.strip()
        )
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
    return [ModelLabPassCheck.from_dict(entry) for entry in payload]


def load_node_inventory_json(
    path: Path | None,
) -> tuple[ModelNodeInventoryTriple, ...]:
    """Load the triples ``probe-lane --node-inventory-out`` wrote.

    A missing or unparseable file RAISES, for the same reason
    :func:`load_checks_json` does: an emitter that pointed at a probe output
    and got nothing must not quietly emit a receipt carrying less than it
    believes it carries.
    """
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_node_inventory(payload)


def build_receipt(
    sha: str,
    lane: EnumLabLane,
    started_at: datetime,
    finished_at: datetime,
    checks: Sequence[ModelLabPassCheck],
    agent_command_id: str | None,
    node_inventory: Sequence[ModelNodeInventoryTriple] = (),
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
        node_inventory=tuple(node_inventory),
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
        return ModelLabPassReceipt.from_json(body)
    except (ValueError, TypeError, KeyError) as exc:
        msg = f"receipt is malformed and cannot be trusted: {exc}"
        raise ReceiptLookupError(msg) from exc


#: How each outcome is badged in a rendered receipt. INDETERMINATE is spelled
#: in full rather than abbreviated: a reader scanning a failing gate's output
#: must not have to know a four-letter code to tell "the lane did not converge"
#: from "we could not establish whether it had a chance to".
_CHECK_BADGE: Final[dict[EnumLabPassCheckOutcome, str]] = {
    EnumLabPassCheckOutcome.PASS: "ok  ",
    EnumLabPassCheckOutcome.FAIL: "FAIL",
    EnumLabPassCheckOutcome.INDETERMINATE: "INDETERMINATE",
}


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
        f"    [{_CHECK_BADGE[c.outcome]}] {c.name}: {c.evidence}"
        for c in receipt.checks
    )
    if receipt.node_inventory:
        # Printed as a count plus a sample, never in full: a lane reports
        # ~150 contracts and a wall of them buries the verdict above it. The
        # full set is on the receipt for a reader that wants it.
        sample = ", ".join(
            f"{t.name}@{t.node_version}" for t in receipt.node_inventory[:5]
        )
        ellipsis = ", ..." if len(receipt.node_inventory) > 5 else ""
        lines.append(
            f"  inventory  : {len(receipt.node_inventory)} node(s) ({sample}{ellipsis})"
        )
    return "\n".join(lines)


def verify_emitted(path: Path, sha: str, lane: EnumLabLane, out: Any) -> int:
    """Refuse a receipt file that is not the one this job just emitted (OMN-18420).

    The presence assertion this sits beside (``assert_evidence_artifact.py``)
    answers "is there a non-empty file here". On a self-hosted host that reuses
    its filesystem across jobs, a STALE file answers that question identically
    to a fresh one -- which is how omnimarket run ``35035178406`` published an
    artifact named for an omnimarket commit carrying an ``omnibase_infra``
    commit's receipt.

    ``evaluate_gate`` already cross-checks the name against the payload, so the
    consuming side was never fooled. But it is in a DIFFERENT REPOSITORY and
    runs an hour later, so the emitting job reported green and the defect
    surfaced as an unexplained delivery refusal. This moves the same assertion
    to the side that can see what happened, and makes it about the file on
    disk rather than about the artifact that will be built from it.

    Deliberately not folded into ``emit``: ``emit`` writes the file, so it can
    only ever agree with itself. The value of this check is that it runs as a
    SEPARATE step over whatever is actually on disk at upload time.
    """
    try:
        receipt = ModelLabPassReceipt.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(
            f"::error::lab-pass receipt at {path} is unreadable or malformed "
            f"({exc}). Refusing to upload it as evidence for {sha}.",
            file=out,
        )
        return 1

    if receipt.sha != sha:
        print(
            f"::error::lab-pass receipt at {path} carries sha {receipt.sha}, but "
            f"this job is emitting for {sha}. This is a receipt left behind by "
            "another job on this host (OMN-18420) -- refusing to upload it under "
            "this job's name.",
            file=out,
        )
        return 1

    if receipt.lane != lane:
        print(
            f"::error::lab-pass receipt at {path} carries lane "
            f"{receipt.lane.value}, but this job is emitting for {lane.value}. "
            "Refusing to upload it under this job's name.",
            file=out,
        )
        return 1

    print(
        f"verified {path} is this job's own receipt: sha {receipt.sha}, lane "
        f"{receipt.lane.value}, result {receipt.result.value}",
        file=out,
    )
    return 0


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

    # OMN-18573. An INDETERMINATE check is named on its own line, with the sha
    # and the check's own evidence, BEFORE the generic refusal. The two are
    # different findings and a reader acts on them differently: a FAIL is a
    # question for the lane, an INDETERMINATE is a question for the hop that
    # was supposed to establish the fact. Collapsing them into one sentence is
    # what made the 2026-09-16/17 receipts read as lane failures.
    unestablished = [
        (receipt, check)
        for receipt in found
        for check in receipt.checks
        if check.outcome is EnumLabPassCheckOutcome.INDETERMINATE
    ]
    print("", file=out)
    for receipt, check in unestablished:
        print(
            f"::error::lab-pass gate: for sha {sha} on lane "
            f"{receipt.lane.value}, check {check.name!r} is INDETERMINATE and "
            f"asserts nothing about the lab lane: {check.evidence}. The sha is "
            "refused because an unestablished check is not a pass, NOT because "
            "the lane was shown to misbehave.",
            file=out,
        )
    print(
        f"::error::lab-pass gate FAILED for {sha}: no PASS lab-pass receipt "
        f"exists for this exact sha on any of {', '.join(lane.value for lane in lanes)}. "
        "Rule 24(b) — the lab is the first place a change runs; staging is "
        "promotion — so this candidate is not deliverable. This is not a skip: a "
        "missing, unreadable, malformed, INDETERMINATE or FAIL receipt all fail "
        "here, and there is no override flag. Exercise the sha on a lab lane and "
        "let its emitter publish the receipt.",
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
    probe.add_argument("--projection-url", required=True)
    probe.add_argument("--timeout-seconds", type=float, default=15.0)
    probe.add_argument(
        "--manifest-url",
        default="",
        help=(
            "the lane's introspection base URL (OMN-18708). Supplying it emits "
            "the node_inventory check, whose evidence is the lane's own "
            f"{INTROSPECTION_MANIFEST_PATH}. Omitting it makes no claim about "
            "the lane's inventory rather than claiming an empty one. Normally "
            "the same value as --main-url: the manifest is served by the same "
            "health server, on the same port, as /ready and /health."
        ),
    )
    probe.add_argument(
        "--node-inventory-out",
        type=Path,
        default=None,
        help=(
            "write the triples read from the manifest to this path, for "
            "`emit --node-inventory-json` to carry onto the receipt. Written "
            "only when the inventory check passed: a file here always contains "
            "triples the lane actually reported."
        ),
    )
    probe.add_argument(
        "--settle-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "seconds to wait for the lane to report ready before reading it. "
            "A convergence success hands the probe a lane that has JUST been "
            "recreated, so without a budget here the probes race the compose "
            "recreate (measured: 174ms after convergence on run 34478680748). "
            "For an ad hoc read only; the default of 0 preserves the first-look "
            "behaviour. An emitting job passes --job-ceiling-seconds instead, "
            "which DERIVES the budget from the lane's declaration rather than "
            "from a remainder (OMN-18436)."
        ),
    )
    probe.add_argument(
        "--settle-budget-json",
        default="",
        help=(
            "the lane's DECLARED settle budget and this job's affordance for "
            "it, as one line of JSON from scripts/ci/lane_settle_budget.py. "
            "Supplying it replaces --settle-timeout-seconds: the declaration is "
            "the budget, and the job's ceiling only decides whether it could be "
            "AFFORDED, which is recorded as its own check instead of arriving "
            "as a lane-health failure (OMN-18436)."
        ),
    )
    probe.add_argument(
        "--generation-container",
        default=None,
        help=(
            "the container whose identity binds these reads to the generation "
            "the convergence guard observed. Supplying it asserts the binding "
            "and emits the probe_generation_bound check; omitting it makes no "
            "claim, which is the ad hoc case."
        ),
    )
    probe.add_argument(
        "--expect-generation",
        default="",
        help=(
            "the convergence step's `generation` output, as one line of JSON. "
            "An EMPTY value with --generation-container supplied means "
            "convergence published none, which fails the binding check rather "
            "than skipping it."
        ),
    )
    # --- OMN-18866: the three integration subjects -------------------------
    # Every one is opt-in and defaults to unsupplied, so `probe-lane` keeps
    # exactly its previous behaviour for an ad hoc caller. There is no flag
    # here that SKIPS a check whose subject was supplied: a subject named is a
    # check owed, and the only way to not owe it is not to name it.
    probe.add_argument(
        "--migration-container",
        default="",
        help=(
            "the lane's postgres container. Supplying it emits the "
            f"{MIGRATIONS_APPLIED_CHECK} check, read through the docker socket "
            "with no network credential. Omitting it makes no claim about the "
            "lane's schema."
        ),
    )
    probe.add_argument(
        "--migration-database",
        default="omnibase_infra",
        help="the database holding public.schema_migrations on that container.",
    )
    probe.add_argument(
        "--forward-migrations-dir",
        type=Path,
        default=Path("docker/migrations/forward"),
        help=(
            "the DELIVERED TREE's forward migrations. Flat *.sql only, matching "
            "check_schema_fingerprint.py's own glob. An absent or empty "
            "directory is an ERROR, not an empty declaration: comparing a lane "
            "against nothing passes against every lane."
        ),
    )
    probe.add_argument(
        "--broker-container",
        default="",
        help=(
            "the lane's broker container. Supplying it emits the "
            f"{CONSUMER_GROUP_LAG_CHECK} check via `rpk` inside that container. "
            "Omitting it makes no claim about consumer lag."
        ),
    )
    probe.add_argument(
        "--broker-address",
        default="redpanda:9092",
        help=(
            "the broker's INTERNAL listener, as addressed from inside its own "
            "container -- not the published host port."
        ),
    )
    probe.add_argument(
        "--consumer-groups",
        default="",
        help=(
            "the consumer groups this lane declares, comma- or newline-"
            "separated. An empty list with --broker-container supplied reports "
            "INDETERMINATE rather than passing: a lane declaring no groups is "
            "not a lane with no lag."
        ),
    )
    probe.add_argument(
        "--max-consumer-lag",
        type=int,
        default=0,
        help=(
            "the declared lag bound. A bound alone cannot see a consumer that "
            "has STOPPED at a lag the bound admits, which is why the growth arm "
            "below exists and is the one that matters."
        ),
    )
    probe.add_argument(
        "--first-lag-sample-json",
        type=Path,
        default=None,
        help=(
            "an earlier lag reading from `sample-lag`, taken BEFORE the "
            "convergence wait so the two samples straddle real time. Absent, "
            "growth is not measured and the check says so in its evidence "
            "rather than implying it looked."
        ),
    )
    probe.add_argument(
        "--chain-canary-receipt",
        type=Path,
        default=None,
        help=(
            "the receipt written by `onex skill chain_canary`, which fires ONE "
            "live delegation through the deployed lane ingress. Supplying it "
            f"emits the {DELEGATION_GOLDEN_CHAIN_CHECK} check. This probe does "
            "not dispatch its own delegation: two dispatchers would be two "
            "definitions of a delegation, free to disagree."
        ),
    )
    for spec in (
        ("--sasl-mechanism-env", "KAFKA_SASL_MECHANISM"),
        ("--sasl-username-env", "KAFKA_SASL_USERNAME"),
        ("--sasl-password-env", "KAFKA_SASL_PASSWORD"),
    ):
        probe.add_argument(
            spec[0],
            default=spec[1],
            help=(
                "the NAME of the environment variable carrying this credential "
                "component. The name, never the value: a credential on a "
                "command line is a credential in a process listing and in every "
                "log that echoes the command."
            ),
        )

    sample = sub.add_parser(
        "sample-lag",
        help=(
            "take ONE lag reading per declared group and write it as JSON, for "
            "`probe-lane --first-lag-sample-json` to compare against later"
        ),
    )
    sample.add_argument("--broker-container", required=True)
    sample.add_argument("--broker-address", default="redpanda:9092")
    sample.add_argument("--consumer-groups", required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--sasl-mechanism-env", default="KAFKA_SASL_MECHANISM")
    sample.add_argument("--sasl-username-env", default="KAFKA_SASL_USERNAME")
    sample.add_argument("--sasl-password-env", default="KAFKA_SASL_PASSWORD")

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
        "--node-inventory-json",
        type=Path,
        default=None,
        help=(
            "a JSON array of {name, node_version, contract_content_hash} "
            "objects, as written by `probe-lane --node-inventory-out`. Carried "
            "onto the receipt so a later reader learns which nodes, at which "
            "contract bodies, this lab pass actually exercised (OMN-18708)."
        ),
    )
    emit.add_argument(
        "--agent-command-id",
        default=None,
        help=(
            "the deploy agent's correlation id for the rebuild this receipt "
            "attests to. An EMPTY value records that the emitter genuinely has "
            "none (no command was published); a non-uuid is refused."
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

    verify = sub.add_parser(
        "verify",
        help="refuse a receipt file that is not the one this job emitted",
    )
    verify.add_argument("--path", required=True, type=Path)
    verify.add_argument("--sha", required=True)
    verify.add_argument("--lane", required=True, choices=[e.value for e in EnumLabLane])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "sample-lag":
        # Never fails the job. A baseline that could not be taken means the
        # growth arm has nothing to compare against, and `check_consumer_group_lag`
        # reports exactly that. Failing here would turn a missing baseline into
        # a missing RECEIPT, which is strictly worse: an unwritten receipt is
        # the one outcome worse than a failing one.
        try:
            sample_access = ModelBrokerAccess(
                container=args.broker_container.strip(),
                brokers=args.broker_address.strip(),
                sasl_mechanism=os.environ.get(args.sasl_mechanism_env, ""),
                sasl_username=os.environ.get(args.sasl_username_env, ""),
                sasl_password=os.environ.get(args.sasl_password_env, ""),
            )
        except ValueError as exc:
            print(f"::warning::lag baseline not taken: {exc}", file=sys.stderr)
            return 0
        readings = sample_group_lag(
            sample_access, parse_group_list_argument(args.consumer_groups)
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(readings, indent=2, sort_keys=True), "utf-8")
        print(
            f"lag baseline: {len(readings)} group(s) read, written to {args.out}",
            file=sys.stderr,
        )
        return 0

    if args.command == "probe-lane":
        budget: ModelSettleBudget | None = None
        settle_seconds = args.settle_timeout_seconds
        if args.settle_budget_json.strip():
            try:
                budget = ModelSettleBudget.from_json(args.settle_budget_json)
            except (ValueError, json.JSONDecodeError) as exc:
                # Fail closed and LOUD. There is no fallback to a job-ceiling
                # remainder: the remainder is the defect this replaced, and
                # falling back to it exactly when the declaration could not be
                # read would reintroduce it on the runs nobody is watching.
                print(f"::error::settle budget unreadable: {exc}", file=sys.stderr)
                return 1
            settle_seconds = float(budget.granted_seconds)
            print(f"settle budget: {budget.evidence}", file=sys.stderr)

        expected_generation: ModelLaneGeneration | None = None
        if args.expect_generation.strip():
            try:
                expected_generation = parse_generation(args.expect_generation)
            except ValueError as exc:
                print(
                    f"::warning::convergence generation record unreadable: {exc}",
                    file=sys.stderr,
                )

        inventory_probe: ModelNodeInventoryProbe | None = None
        if args.manifest_url.strip():
            inventory_probe = check_node_inventory(
                f"{args.manifest_url.rstrip('/')}{INTROSPECTION_MANIFEST_PATH}",
                args.timeout_seconds,
            )

        # OMN-18866. Each subject is assembled only when the caller named it,
        # and a caller that names it PARTLY is refused rather than quietly
        # degraded -- a half-supplied subject is the shape that produces a
        # check asserting less than its name says.
        migration_ledger: ModelMigrationLedger | None = None
        declared_migrations: tuple[str, ...] = ()
        if args.migration_container.strip():
            try:
                migration_ledger = ModelMigrationLedger(
                    container=args.migration_container.strip(),
                    database=args.migration_database.strip(),
                )
                declared_migrations = declared_forward_migrations(
                    args.forward_migrations_dir
                )
            except ValueError as exc:
                print(
                    f"::error::migration subject unusable: {exc}",
                    file=sys.stderr,
                )
                return 1

        broker_access: ModelBrokerAccess | None = None
        if args.broker_container.strip():
            try:
                broker_access = ModelBrokerAccess(
                    container=args.broker_container.strip(),
                    brokers=args.broker_address.strip(),
                    sasl_mechanism=os.environ.get(args.sasl_mechanism_env, ""),
                    sasl_username=os.environ.get(args.sasl_username_env, ""),
                    sasl_password=os.environ.get(args.sasl_password_env, ""),
                )
            except ValueError as exc:
                print(f"::error::broker subject unusable: {exc}", file=sys.stderr)
                return 1

        first_lag_sample = load_lag_sample(args.first_lag_sample_json)

        checks = probe_compose_dev(
            args.main_url,
            args.effects_url,
            args.timeout_seconds,
            settle_seconds,
            args.projection_url,
            budget=budget,
            expected_generation=expected_generation,
            generation_container=args.generation_container,
            node_inventory_probe=inventory_probe,
            declared_migrations=declared_migrations,
            migration_ledger=migration_ledger,
            broker_access=broker_access,
            declared_consumer_groups=parse_group_list_argument(args.consumer_groups),
            max_consumer_lag=args.max_consumer_lag,
            first_lag_sample=first_lag_sample,
            chain_canary_receipt=args.chain_canary_receipt,
        )
        if args.node_inventory_out is not None:
            # Written even when the probe found nothing usable, as an empty
            # array: a downstream `emit` that pointed here must get a file, so
            # a missing one stays a real error rather than an empty inventory.
            args.node_inventory_out.parent.mkdir(parents=True, exist_ok=True)
            args.node_inventory_out.write_text(
                json.dumps(
                    [
                        t.to_dict()
                        for t in (inventory_probe.triples if inventory_probe else ())
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(json.dumps([c.to_dict() for c in checks], indent=2))
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
                agent_command_id=parse_agent_command_id(args.agent_command_id),
                node_inventory=load_node_inventory_json(args.node_inventory_json),
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            print(
                f"::error::refusing to emit an invalid receipt: {exc}", file=sys.stderr
            )
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(receipt.to_json(indent=2), encoding="utf-8")
        print(f"wrote {artifact_name(receipt.lane, receipt.sha)} -> {args.out}")
        print(render_receipt(receipt))
        # A FAIL receipt is still EMITTED — the record of a failed lab pass is
        # exactly as valuable as the record of a passing one — but the emitting
        # step goes red so the failure is visible on the run that produced it.
        return 0 if receipt.result == EnumLabPassResult.PASS else 1

    if args.command == "verify":
        return verify_emitted(args.path, args.sha, EnumLabLane(args.lane), sys.stdout)

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
