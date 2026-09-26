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

    ``--lane`` is ANY-OF: one PASS among them satisfies it. ``--require-lane``
    (OMN-19312) is ALL-OF: every named lane must carry its own PASS, and no
    other lane's PASS substitutes. The distinction is load-bearing: the push
    path's unqualified call is satisfied by the onex-lab boot receipt the same
    workflow emits, so a verdict can only bind delivery as a required lane.
    ``--wait-seconds`` polls, bounded, for a required receipt that has not
    landed yet, and expiry refuses. ``--resolve-runtime-ancestor`` asks required
    lanes about the nearest runtime-affecting ancestor, by the release train's
    OMN-18664 rule (loaded from ``release_train.py``, never restated).

``workflow-verdict``
    OMN-18866. For a check that has no sha-keyed receipt of its own (or whose
    lane may not name itself in one, a governed lane), reads the newest
    completed run of one workflow on one branch and exits non-zero unless it
    concluded ``success`` within a freshness bound.

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
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess  # fixed argv, no shell, trusted git/gh binaries
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
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

#: The digest of the exact PR diff bytes the proof covers. The algorithm is
#: part of the value so a later contract can add another digest without a
#: reader silently interpreting it as SHA-256.
_PR_DIFF_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

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

    ``COMPOSE_DEV_CHAIN`` and ``COMPOSE_DEV_CORPUS`` (OMN-19312) are VERDICT
    lanes on the same ``.201`` compose dev lane, each with exactly one emitter:
    the chain canary (``chain-canary.yml``, check C15) and the per-candidate
    delegation corpus (check D11's dev-lane leg). They are separate values for
    the ``ONEX_LAB_K3S`` reasons above -- one emitter per name, and a different
    claim -- and for a third: the ``compose-dev`` receipt is also the release
    train's premise, so folding either verdict into it would stop release cuts,
    a surface the 2026-09-23T17:12:15Z operator ruling did not name (the shape
    OMN-18872 was reverted for). A verdict lane binds delivery only when a
    caller REQUIRES it with ``gate --require-lane``; it is not in
    ``ANY_OF_DEFAULT_LANES``, so its PASS never satisfies the any-of lab-pass
    premise on its own.

    ``COMPOSE_DEV_202`` (OMN-19507) is the SECOND deployed dev lane, ``dev-202``
    on the ``.202`` host (compose project ``omnibase-infra-dev-202``, ports
    61085/61086), run by a second deploy-agent instance (OMN-19506). It is a
    separate value for the ``ONEX_LAB_K3S`` reasons: one emitter per name, and a
    different host proving a different lane. The operator ruled on
    2026-09-25T00:56:45Z that its PASS proves omnimarket changes only, and
    omnibase_infra stays proven on ``.201``. So it is NOT in
    ``ANY_OF_DEFAULT_LANES``: an unqualified gate never reads it, and only the
    reads OMN-19508 names (omnimarket's sibling read and its release-cut premise)
    may admit it.

    ``COMPOSE_DEV_200`` (OMN-19543) is the third deployed dev lane, ``dev-200``
    on the ``.200`` host (compose project ``omnibase-infra-dev-200``, ports
    42085/42086), on exactly ``COMPOSE_DEV_202``'s terms: one emitter, omnimarket
    changes only (operator ruling 2026-09-25T10:22:22Z, one deploy slot per lab
    host), NOT in ``ANY_OF_DEFAULT_LANES``. Which repositories an instance lane
    may prove is read from ``config/deploy_lane_routing.yaml``
    (``scripts/ci/instance_receipt_lanes.py``), not from this enum.

    ``PR_HEAD`` (OMN-19566) is the pre-merge proof lane. Its one emitter is the
    PR-head verifier, which mints a proof for one exact pull-request head and
    profile. It is NOT in ``ANY_OF_DEFAULT_LANES``: a PR-head proof is not a
    post-merge lab pass and must never satisfy the rule 24(b) delivery gate.

    No other value is admissible, and in particular no governed lane
    (``prod``, ``stability-test``, ``judge``, or a collaborator lane) can name
    itself in a receipt. A lab pass is a statement about a lab. A governed
    lane's verdict is read state-keyed instead, with ``workflow-verdict``.
    """

    COMPOSE_DEV = "compose-dev"
    ONEX_LAB = "onex-lab"
    ONEX_LAB_K3S = "onex-lab-k3s"
    COMPOSE_DEV_CHAIN = "compose-dev-chain"
    COMPOSE_DEV_CORPUS = "compose-dev-corpus"
    COMPOSE_DEV_202 = "compose-dev-202"
    COMPOSE_DEV_200 = "compose-dev-200"
    PR_HEAD = "pr-head"


#: The lanes an unqualified ``gate`` reads with ANY-OF semantics: the three lab
#: surfaces rule 24(b) means by "a passing lab receipt". The OMN-19312 verdict
#: lanes are deliberately absent -- a chain canary PASS is not evidence that the
#: candidate booted, and must never be able to stand in for that premise.
#: ``COMPOSE_DEV_202`` is absent too (OMN-19507), and ``COMPOSE_DEV_200``
#: (OMN-19543): they prove omnimarket changes only, so no unqualified read may
#: take either for the any-of premise.
ANY_OF_DEFAULT_LANES: Final[tuple[EnumLabLane, ...]] = (
    EnumLabLane.COMPOSE_DEV,
    EnumLabLane.ONEX_LAB,
    EnumLabLane.ONEX_LAB_K3S,
)


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


class EnumLabProofHandlerKind(StrEnum):
    """The execution shape used to prove one PR head."""

    RUNTIME_IMAGE = "runtime_image"
    FOUNDATION_OVERRIDE = "foundation_override"
    PYPI_SIBLING_OVERRIDE = "pypi_sibling_override"
    PLUGIN_SESSION = "plugin_session"
    WEB_RENDER = "web_render"
    K8S_NAMESPACE = "k8s_namespace"
    SCRIPT_REPLAY = "script_replay"
    EXEMPT = "exempt"


@dataclass(frozen=True)
class ModelLabProofSubject:
    """The exact pull request, profile, and identities a PR-head proof binds."""

    repo: str
    pr_number: int
    base_sha: str
    merge_base_sha: str
    profile_id: str
    profile_version: str
    handler_kind: EnumLabProofHandlerKind
    host: str
    slot: str
    runner_identity: str
    verifier_identity: str
    pr_diff_digest: str
    carried_from: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.repo, str) or not re.fullmatch(
            r"OmniNode-ai/[A-Za-z0-9._-]+", self.repo
        ):
            msg = (
                f"repo={self.repo!r} must name an OmniNode-ai repository as "
                "'OmniNode-ai/<name>'"
            )
            raise ValueError(msg)
        if (
            not isinstance(self.pr_number, int)
            or isinstance(self.pr_number, bool)
            or self.pr_number <= 0
        ):
            msg = f"pr_number={self.pr_number!r} must be an integer greater than zero"
            raise ValueError(msg)
        for field_name, value in (
            ("base_sha", self.base_sha),
            ("merge_base_sha", self.merge_base_sha),
        ):
            if not isinstance(value, str) or not _SHA_RE.match(value):
                msg = f"{field_name}={value!r} must be a 40-character lowercase sha"
                raise ValueError(msg)
        for field_name, value in (
            ("profile_id", self.profile_id),
            ("profile_version", self.profile_version),
            ("host", self.host),
            ("slot", self.slot),
            ("runner_identity", self.runner_identity),
            ("verifier_identity", self.verifier_identity),
        ):
            if not isinstance(value, str) or not value:
                msg = f"{field_name} is required and must be a non-empty string"
                raise ValueError(msg)
        if not isinstance(self.handler_kind, EnumLabProofHandlerKind):
            msg = (
                f"handler_kind={self.handler_kind!r} is not an EnumLabProofHandlerKind"
            )
            raise ValueError(msg)
        if not isinstance(self.pr_diff_digest, str) or not _PR_DIFF_DIGEST_RE.match(
            self.pr_diff_digest
        ):
            msg = (
                f"pr_diff_digest={self.pr_diff_digest!r} must be 'sha256:' "
                "followed by 64 lowercase hex characters"
            )
            raise ValueError(msg)
        if not isinstance(self.carried_from, str) or (
            self.carried_from and not _SHA_RE.match(self.carried_from)
        ):
            msg = (
                f"carried_from={self.carried_from!r} must be empty or a "
                "40-character lowercase sha"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "pr_number": self.pr_number,
            "base_sha": self.base_sha,
            "merge_base_sha": self.merge_base_sha,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "handler_kind": self.handler_kind.value,
            "host": self.host,
            "slot": self.slot,
            "runner_identity": self.runner_identity,
            "verifier_identity": self.verifier_identity,
            "pr_diff_digest": self.pr_diff_digest,
            "carried_from": self.carried_from,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ModelLabProofSubject:
        if not isinstance(payload, dict):
            msg = f"a PR-head subject must be an object, got {type(payload).__name__}"
            raise ValueError(msg)
        required = {
            "repo",
            "pr_number",
            "base_sha",
            "merge_base_sha",
            "profile_id",
            "profile_version",
            "handler_kind",
            "host",
            "slot",
            "runner_identity",
            "verifier_identity",
            "pr_diff_digest",
        }
        optional = {"carried_from"}
        unknown = sorted(set(payload) - required - optional)
        if unknown:
            msg = f"unknown PR-head subject field(s) {unknown}"
            raise ValueError(msg)
        missing = sorted(required - set(payload))
        if missing:
            msg = f"PR-head subject is missing required field(s) {missing}"
            raise ValueError(msg)
        try:
            handler_kind = EnumLabProofHandlerKind(payload["handler_kind"])
        except ValueError as exc:
            msg = (
                f"handler_kind={payload['handler_kind']!r} is not one of "
                f"{[kind.value for kind in EnumLabProofHandlerKind]}"
            )
            raise ValueError(msg) from exc
        return cls(
            repo=payload["repo"],
            pr_number=payload["pr_number"],
            base_sha=payload["base_sha"],
            merge_base_sha=payload["merge_base_sha"],
            profile_id=payload["profile_id"],
            profile_version=payload["profile_version"],
            handler_kind=handler_kind,
            host=payload["host"],
            slot=payload["slot"],
            runner_identity=payload["runner_identity"],
            verifier_identity=payload["verifier_identity"],
            pr_diff_digest=payload["pr_diff_digest"],
            carried_from=payload.get("carried_from", ""),
        )


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
    #: OMN-18976. The sha whose convergence produced this receipt, when that is
    #: NOT ``sha`` itself. Empty on a receipt written by the sha's own run,
    #: which is every receipt written before this change and most written after.
    #:
    #: A merge queued behind another deploy emits no receipt of its own; a
    #: later run that converges on a revision CONTAINING it answers on its
    #: behalf, with probes taken against that very image. Both are honest
    #: evidence and they are not the same evidence, so the difference is
    #: recorded rather than left for a reader to infer from timestamps. It is
    #: written to the wire only when non-empty, on the same terms as
    #: ``node_inventory``, so ``RECEIPT_VERSION`` is unchanged and every
    #: receipt already in flight still parses byte-identically.
    converged_via: str = ""
    receipt_version: str = RECEIPT_VERSION
    #: OMN-19566. Present only for the pre-merge ``pr-head`` lane. Like
    #: ``node_inventory`` and ``converged_via``, absence preserves the exact
    #: wire form of every receipt written before this field existed.
    subject: ModelLabProofSubject | None = None

    def __post_init__(self) -> None:
        self._validate_version()
        self._validate_sha_is_exact()
        self._validate_checks_present()
        self._validate_result_matches_checks()
        self._validate_window()
        self._validate_node_inventory()
        self._validate_subject()

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

    def _validate_subject(self) -> None:
        if self.lane is EnumLabLane.PR_HEAD and self.subject is None:
            msg = "lane='pr-head' requires a PR-head subject"
            raise ValueError(msg)
        if self.lane is not EnumLabLane.PR_HEAD and self.subject is not None:
            msg = f"lane={self.lane.value!r} refuses a PR-head subject"
            raise ValueError(msg)
        if self.subject is not None and self.subject.carried_from == self.sha:
            msg = (
                "subject.carried_from must differ from the receipt sha; carrying "
                "a proof from the same head is not carry-over"
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
                **({"converged_via": self.converged_via} if self.converged_via else {}),
                **({"subject": self.subject.to_dict()} if self.subject else {}),
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
        optional = {"node_inventory", "converged_via", "subject"}
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
            converged_via=str(payload.get("converged_via", "")),
            subject=(
                ModelLabProofSubject.from_dict(payload["subject"])
                if "subject" in payload
                else None
            ),
        )


#: OMN-18769. The topic the receipt's VERDICT is published on, beside the
#: artifact, at the moment the artifact is written.
#:
#: The artifact stays the durable evidence and the gate keeps reading it; this
#: event changes nothing about that. What it adds is renderability: an artifact
#: is reachable only by an exact-name REST query with ``actions: read``, which
#: a dashboard panel cannot make, so a lab-pass verdict has until now been
#: invisible to every surface except the gate that consumes it.
#:
#: It is published for a FAIL exactly as for a PASS, for the same reason the
#: artifact is written under ``always()``: a record that exists only on success
#: cannot distinguish a failed lab pass from one nobody ran, and telling those
#: apart is the whole job.
LAB_PASS_EVENT_TOPIC = "onex.evt.omnibase-infra.lab-pass-receipt.v1"


def build_bus_event(receipt: ModelLabPassReceipt) -> dict[str, Any]:
    """Render the receipt as the bus event a lane-health projection folds.

    It is a PROJECTION of the receipt, not a second source of truth: every
    field is copied from the receipt, the verdict included, and nothing is
    re-derived here. ``ModelLabPassReceipt`` derives ``result`` from the check
    set precisely so an emitter cannot record green over a failed readback, and
    re-deriving it in this function would reintroduce exactly that seam.

    ``failing_checks`` is a convenience the consumer would otherwise have to
    compute, and it is named rather than implied: a check that is INDETERMINATE
    is not passing, so it appears here. A panel that showed only hard failures
    would report an unprobed lane as fully checked.
    """
    return {
        "schema_version": "1.0.0",
        "event_type": "lab-pass-receipt",
        "topic": LAB_PASS_EVENT_TOPIC,
        "receipt_version": receipt.receipt_version,
        "sha": receipt.sha,
        "lane": receipt.lane.value,
        "started_at": receipt.started_at.isoformat(),
        "finished_at": receipt.finished_at.isoformat(),
        "result": receipt.result.value,
        "checks": [check.to_dict() for check in receipt.checks],
        "failing_checks": [check.name for check in receipt.checks if not check.ok],
        "artifact_name": artifact_name(receipt.lane, receipt.sha),
        "agent_command_id": receipt.agent_command_id,
    }


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
)

#: Named here rather than silently absent, so a reader can see what a
#: ``compose-dev`` receipt does NOT cover. Widening the set means adding the
#: probe AND the check name in ONE change -- never the name alone (rule 24).
#:
#: OMN-18866 wired `migrations_applied` and `consumer_group_lag`, which are
#: live-proven against the lane. It ALSO wired `delegation_golden_chain`, and
#: that was a mistake, reverted here within the hour. The reason is recorded
#: rather than quietly dropped, because the next person to reach for it will
#: otherwise make the same one.
#:
#: WHAT WENT WRONG. The check was bound to a chain-canary dispatch fired inline
#: by `verify-lane-converged`. The canary is a SEPARATELY OWNED surface, its
#: scheduled runs were already red on the tenant-bearing ingress (OMN-18872,
#: gateway-key work), and this job sits on the delivery critical path. So the
#: binding made every compose-dev receipt FAIL on a defect in another lane's
#: surface, which blocked the infra release train under rule 24(b) on shas that
#: had nothing to do with it. Measured on run 35488731787: the lane was healthy
#: and converged, seven checks passed, and delivery was refused anyway.
#:
#: A check that cannot pass is exactly as useless as one that cannot fail, and
#: considerably more expensive, because it stops delivery instead of merely
#: failing to report. That is the rule-5 lesson the wiring got backwards.
#:
#: WHAT IT NEEDS INSTEAD, and it is a design rather than a patch: the canary
#: should emit a SHA-KEYED receipt of its own, the way this module's own
#: docstring already argues a durable premise must be keyed, and this gate
#: should READ that receipt rather than dispatch the canary itself. That
#: decouples the two surfaces, keeps the delivery path off another lane's
#: dispatch, and answers "was this sha's delegation exercised" for a sha whose
#: run is minutes old. The probe function, its parser flag and its full test
#: set are DELIBERATELY RETAINED below, unused by the emitting job, so that
#: design is a wiring change and not a rewrite.
#:
#: The tuple is kept rather than deleted for the same reason it always was: an
#: absent list is how a gap stops being visible.
PROBES_NOT_YET_WIRED: tuple[str, ...] = (DELEGATION_GOLDEN_CHAIN_CHECK,)


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


#: OMN-18886. How long to keep asking for the health dimensions after the lane
#: is READY, before calling them never observed. Separate from the readiness
#: budget because it measures a different thing: readiness is the port binding,
#: this is the first completed background observation, and on a fresh lane the
#: second lags the first.
HEALTH_OBSERVE_POLL_SECONDS: Final[float] = 10.0


def check_health_dimensions(url: str, timeout_seconds: float) -> ModelLabPassCheck:
    """Every dimension the health payload reports must be healthy.

    Fails closed on an unparseable body, on a payload carrying no dimension
    block, and on a dimension carrying no status: "we could not find an
    unhealthy dimension" is not the same statement as "every dimension is
    healthy", and only the second one is a check.

    ONE SAMPLE. Callers that can afford to wait should use
    :func:`check_health_dimensions_observed`, which polls; this is the single
    read it is built from, kept separate so the read and the waiting are
    testable apart.
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


#: Distinguishes the two failing outcomes below in the evidence a reader sees.
#: "Never observed" is a statement about the OBSERVER; "still unhealthy" is a
#: statement about the LANE, and conflating them sent two lanes hunting a lane
#: defect that did not exist (OMN-18886).
_HEALTH_NEVER_OBSERVED = "never observed"


def check_health_dimensions_observed(
    url: str,
    timeout_seconds: float,
    observe_budget_seconds: float,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ModelLabPassCheck:
    """Wait for the health dimensions to be OBSERVED, then judge them.

    THE DEFECT THIS FIXES (OMN-18886). ``details.runtime_health`` is populated
    ASYNCHRONOUSLY -- it carries its own ``observed_at`` and ``age_seconds``,
    so it is a background observation and not a synchronous read of the
    runtime. A single sample taken after readiness but before the first
    observation completes finds it absent, and the single-sample check then
    reported "an absent dimension set is not a healthy dimension set". That
    sentence is correct and the conclusion was wrong: the dimensions were not
    absent, they had not happened yet.

    Measured on the omnimarket sibling emitter, runs 35505864089 (10:43Z) and
    35507742784 (11:24Z): FAIL with exactly one failing check, on a lane where
    every convergence check passed, sampled 595 s into a 900 s budget. Read
    live afterwards the block was a present dict. Those receipts BLOCK a
    staging delivery -- ``deliver-dev-candidate-to-staging.yml``'s
    ``lab-pass-gate`` reads the sibling receipt and ``dispatch-to-staging``
    needs that job -- so this was not a cosmetic red.

    THREE OUTCOMES, deliberately distinct, because two of them were previously
    one:

    * never observed inside the budget -> FAIL, evidence says so in those
      words. Still fail-closed: absent is not healthy, and this does not
      weaken that. What changes is that the receipt now says whether the
      OBSERVER ran out of time or the LANE is sick.
    * observed and every dimension healthy -> PASS, evidence records how long
      the observation took to appear, so a lane drifting slower is visible
      before it starts failing.
    * observed and still unhealthy when the budget expires -> FAIL, naming the
      dimensions and how long they stayed that way.

    WHY IT KEEPS POLLING AFTER AN UNHEALTHY OBSERVATION, and why that is not
    papering over a degradation. Several dimensions are rolling-window
    measures (``projection_dlq_saturation`` is "over 10 flow windows"), so the
    first observation on a freshly recreated lane can be unhealthy from the
    boot itself. Polling lets that clear. It cannot manufacture a green,
    because the terminal condition on failure is "STILL unhealthy after the
    full budget", with the offending dimension names in the evidence -- a
    genuine degradation fails exactly as it did before, just later and with a
    duration attached. **It never passes on first sight of an unhealthy set
    and never stops early on one.**

    A real degradation was live on the lane while this was written -- one
    projection routing 100% of consumed events to a dead-letter sink -- and
    this function reports it as a failure. That is the intended behaviour and
    the reason the poll is bounded rather than open.
    """
    deadline = clock() + max(0.0, observe_budget_seconds)
    started = clock()
    last: ModelLabPassCheck | None = None
    observed_after: float | None = None

    while True:
        last = check_health_dimensions(url, timeout_seconds)
        elapsed = clock() - started
        # "Observed" is decided by the evidence the single read produced, not
        # by re-parsing the body here: one parser, one meaning. The absent
        # branches are the only ones that phrase it this way.
        is_absent = "an absent dimension set is not a healthy dimension set" in (
            last.evidence
        )
        if not is_absent and last.ok:
            return ModelLabPassCheck(
                name=last.name,
                ok=True,
                evidence=(
                    f"{last.evidence} [dimensions observed after {elapsed:.0f}s "
                    f"of a {observe_budget_seconds:.0f}s budget]"
                ),
            )
        if not is_absent and observed_after is None:
            observed_after = elapsed
        if clock() >= deadline:
            break
        sleep_fn(min(HEALTH_OBSERVE_POLL_SECONDS, max(0.0, deadline - clock())))

    waited = clock() - started
    assert last is not None
    if observed_after is None:
        return ModelLabPassCheck(
            name=last.name,
            ok=False,
            evidence=(
                f"{last.evidence} [the dimension set was {_HEALTH_NEVER_OBSERVED} "
                f"within {waited:.0f}s; runtime_health is populated "
                "asynchronously, so this says the observation never completed, "
                "NOT that the lane reported an unhealthy dimension]"
            ),
        )
    return ModelLabPassCheck(
        name=last.name,
        ok=False,
        evidence=(
            f"{last.evidence} [observed after {observed_after:.0f}s and STILL "
            f"unhealthy {waited:.0f}s later, across the full budget; this is a "
            "lane report, not a timing artefact]"
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
    source_error: str = "",
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
    # The source failing to be READ and the lane declaring NOTHING are two
    # different facts, and conflating them is the OMN-18866 production defect:
    # a silently-failed deriving step produced an empty value, and this check
    # reported "the lane declares no groups", which was false and which nobody
    # could act on. The source error is now reported as itself.
    if source_error:
        return ModelLabPassCheck.indeterminate_check(
            CONSUMER_GROUP_LAG_CHECK, source_error
        )
    if not groups:
        return ModelLabPassCheck.indeterminate_check(
            CONSUMER_GROUP_LAG_CHECK,
            "the declared-group source was read successfully and contained no "
            "groups, so there is nothing to measure; an empty declaration is "
            "not a healthy lane, and this is NOT the same as the source being "
            "unreadable, which is reported separately",
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


class GroupSourceError(RuntimeError):
    """The declared-group source could not be read at all."""


def load_declared_groups(path: Path) -> tuple[str, ...]:
    """Read the declared group list from the file the deriver wrote.

    OMN-18866 follow-up. The first wiring passed this list through a WORKFLOW
    STEP OUTPUT, and that is how the probe came to assert something false in
    production. The producing step failed -- silently, exit 1 with no output,
    because its stdout was swallowed by a command substitution -- so the step
    output was never set, the expression delivered an EMPTY STRING, and the
    probe read that as "this lane declares no consumer groups". It then
    reported INDETERMINATE for the right reason applied to the wrong fact, and
    every dev sha got a non-PASS receipt.

    An unset step output and a genuinely empty declaration are the same bytes.
    A MISSING FILE and an EMPTY FILE are not, which is the whole reason this
    reads a path: the two failures now have two different outcomes, and only
    one of them is "this lane declares nothing".

    Raises rather than returning empty when the file is absent, so the caller
    reports "the source could not be read" instead of "there is nothing to
    measure". Both are non-PASS; only one of them is true.
    """
    if not path.exists():
        msg = (
            f"the declared-group source {path} does not exist, so the deriving "
            "step did not run or did not write it. That is a different fact "
            "from a lane declaring no consumer groups, and it is reported as "
            "its own failure rather than folded into that one"
        )
        raise GroupSourceError(msg)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"the declared-group source {path} could not be read: {exc}"
        raise GroupSourceError(msg) from exc
    return parse_group_list_argument(raw)


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
    health_observe_budget_seconds: float | None = None,
    declared_migrations: Sequence[str] | None = None,
    migration_ledger: ModelMigrationLedger | None = None,
    broker_access: ModelBrokerAccess | None = None,
    declared_consumer_groups: Sequence[str] | None = None,
    consumer_group_source_error: str = "",
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
        check_health_dimensions_observed(
            f"{main_url.rstrip('/')}/health",
            timeout_seconds,
            # DERIVED from the settle budget the lane already declares, minus
            # what the readiness wait actually spent -- never a new constant.
            # The job ceiling is already derived from `granted_seconds` plus a
            # reserved tail (OMN-18436), so spending the UNUSED remainder here
            # cannot overrun it, and a second hand-set number would be one more
            # thing to keep in sync with that arithmetic. A caller may override
            # for an ad hoc read or a test.
            (
                max(0.0, outcome.granted_seconds - outcome.waited_seconds)
                if health_observe_budget_seconds is None
                else health_observe_budget_seconds
            ),
        ),
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
                source_error=consumer_group_source_error,
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
    converged_via: str = "",
    subject: ModelLabProofSubject | None = None,
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
        converged_via=converged_via,
        subject=subject,
    )


#: The checks whose failure means THE RUN could not bind its observation, rather
#: than the lane being unhealthy (OMN-18988). Membership alone is not the test:
#: see :func:`is_binding_only_failure`, which reads ``deployed_revision``'s
#: OUTCOME, because the same check also carries the lane's own convergence
#: failure and those two must never be collapsed.
BINDING_CLASS_CHECKS: Final[frozenset[str]] = frozenset(
    {"deployed_revision", "probe_generation_bound"}
)


def is_binding_only_failure(receipt: ModelLabPassReceipt) -> bool:
    """May this FAIL receipt be answered for by a later run?

    Yes only when EVERY failing check is binding-class: the run could not bind
    its observation to the thing under test, rather than the thing under test
    being unhealthy.

    WHY THIS EXISTS. Measured on ``430ff3434``: ``deployed_revision``
    INDETERMINATE with ``commands_ahead`` ZERO, because the deploy agent
    answered HTTP 404 for the run's own correlation id and the lane's budget
    never started; plus ``probe_generation_bound``, because the probe read a
    different container generation than convergence verified. The lane had
    converged and was running the sha. Neither failure says anything about the
    lane, and frozen as FAIL both are inherited by every merge behind them.

    THE LINE THIS DRAWS, and it is the whole function. A receipt whose health
    checks genuinely failed is NEVER re-emittable: turning a real finding into
    a PASS is strictly worse than the wrong FAIL this removes. So:

    * ``deployed_revision`` counts ONLY with outcome ``INDETERMINATE``. The same
      check with outcome ``FAIL`` is the lane having had its whole budget and
      not converged -- a statement about the lane. Keying on the check NAME
      alone would collapse the two, which is the easiest way to get this wrong.
    * ``probe_generation_bound`` is binding by construction: its failure says
      the reads were about a different container generation, never that the
      lane is unwell.
    * ONE non-binding failure poisons the whole receipt. There is no partial
      re-emission.
    """
    if receipt.result is not EnumLabPassResult.FAIL:
        return False
    failing = [c for c in receipt.checks if not c.ok]
    if not failing:
        return False
    for check in failing:
        if check.name not in BINDING_CLASS_CHECKS:
            return False
        if (
            check.name == "deployed_revision"
            and check.outcome is not EnumLabPassCheckOutcome.INDETERMINATE
        ):
            return False
    return True


@dataclass(frozen=True)
class ModelLaneBinding:
    """Which surface established that the lane runs code containing a sha."""

    sha: str
    surface: str
    observed: str

    @property
    def converged_via(self) -> str:
        """The provenance string a re-emitted receipt carries.

        It names the SURFACE as well as the revision, because "the lane was on
        something containing this" and "the container label said so" are
        different strengths of claim, and a reader of a re-emitted PASS is
        entitled to know which one they have.
        """
        return f"{self.observed} via {self.surface}"


def resolve_lane_binding(
    *,
    sha: str,
    container_revision: str | None,
    agent_loaded_code_sha: str | None,
    ready_version_revision: str | None,
    contains: Callable[[str, str], bool],
) -> ModelLaneBinding | None:
    """Establish, from the lane itself, that it runs code containing ``sha``.

    OMN-18988. A re-emission cannot INHERIT the converged run's binding: that
    run's receipt failed precisely because it could not bind its own
    observation. So the binding is re-established here, against the live lane,
    across three independent surfaces read in descending directness:

    1. the container's own revision label,
    2. the deploy agent's recorded ``loaded_code_sha``,
    3. the runtime's ``/ready`` version.

    THE FIRST SURFACE TO ESTABLISH CONTAINMENT WINS, and a surface that merely
    ANSWERS does not veto the others. That asymmetry is deliberate: a container
    label can lag a restart while the agent has already recorded the new code,
    so treating a readable-but-non-containing surface as a refusal would make
    the common case unbindable.

    ALL THREE SILENT IS A REFUSAL, and so is all three answering without
    containment. Nothing bound means nothing may be claimed; emitting on a
    guess is the fabrication this whole change exists to avoid.
    """
    for surface, observed in (
        ("container-revision", container_revision),
        ("agent-loaded-code-sha", agent_loaded_code_sha),
        ("ready-version", ready_version_revision),
    ):
        if not observed:
            continue
        if contains(sha, observed):
            return ModelLaneBinding(sha=sha, surface=surface, observed=observed)
    return None


def read_agent_loaded_code_sha(agent_url: str, timeout_seconds: float = 10.0) -> str:
    """The code sha the deploy agent RECORDED for what it last loaded.

    OMN-18988, surface two of three. Empty on any failure: an unreachable agent
    is one surface declining to answer, never a refusal on its own, because the
    caller has two more to try. A refusal is all three staying silent.
    """
    if not agent_url:
        return ""
    try:
        with urllib.request.urlopen(  # noqa: S310
            f"{agent_url.rstrip('/')}/health", timeout=timeout_seconds
        ) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - a silent surface is data, not an error
        return ""
    value = payload.get("loaded_code_sha") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and _SHA_RE.match(value) else ""


def read_ready_revision(ready_url: str, timeout_seconds: float = 10.0) -> str:
    """The revision the runtime reports for itself on ``/ready``.

    OMN-18988, surface three of three, and the weakest: it reports a package
    VERSION rather than a commit on most shapes, so it answers only when the
    runtime carries an explicit revision. Empty on anything else, on the same
    terms as the surface above.
    """
    if not ready_url:
        return ""
    try:
        with urllib.request.urlopen(  # noqa: S310
            ready_url, timeout=timeout_seconds
        ) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - a silent surface is data, not an error
        return ""
    if not isinstance(payload, dict):
        return ""
    details = payload.get("details")
    for candidate in (payload, details if isinstance(details, dict) else {}):
        value = candidate.get("revision")
        if isinstance(value, str) and _SHA_RE.match(value):
            return value
    return ""


def reemit_receipt(
    source: ModelLabPassReceipt, sha: str, converged_via: str = ""
) -> ModelLabPassReceipt:
    """Re-key a converged receipt onto a sha that was queued behind it.

    OMN-18976 part two. The queued sha's own verify run emitted nothing, so a
    later run that converged on a revision CONTAINING it answers on its behalf.

    THE EVIDENCE IS COPIED, NOT RE-TAKEN, and that is the point. The probes on
    ``source`` were read against the image the lane converged to, and that
    image contains ``sha`` -- so they are exactly the reads a lab pass for
    ``sha`` would have made, taken at the only moment they could be. Re-probing
    here would read a LATER lane state and attribute it to this sha, which is
    the weaker claim.

    ``converged_via`` records whose convergence this was, so the two kinds of
    receipt stay tellable apart. A receipt that merely said PASS would lose
    the distinction between "its own run watched this land" and "a later run
    subsumed it", and those are different evidence.

    REFUSALS. Re-keying onto the source's own sha is a caller error, not a
    no-op: it would race two artifacts of one name from one job. Re-keying a
    receipt that is itself a re-emission is refused too -- provenance that
    chains is provenance nobody can read.
    """
    if sha == source.sha:
        msg = (
            f"refusing to re-emit {source.sha} onto itself; its own run wrote "
            "that receipt"
        )
        raise ValueError(msg)
    if source.converged_via:
        msg = (
            f"refusing to re-emit from a receipt that is itself a re-emission "
            f"(converged_via={source.converged_via}); re-emit from the "
            "converged run's own receipt"
        )
        raise ValueError(msg)
    return ModelLabPassReceipt(
        sha=sha,
        lane=source.lane,
        started_at=source.started_at,
        finished_at=source.finished_at,
        result=source.result,
        checks=source.checks,
        agent_command_id=source.agent_command_id,
        node_inventory=source.node_inventory,
        # OMN-18988: the caller supplies this when it re-established the
        # binding against the live lane, so it names the SURFACE that answered
        # and not merely the sha. Defaulting to the source sha keeps the
        # queued-ahead path, where the converged run bound its own observation
        # and the source sha IS the provenance.
        converged_via=converged_via or source.sha,
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


def _gh_api_post(path: str) -> None:
    """One POST against the GitHub REST API. A non-zero exit RAISES (rule 16)."""
    argv = ["gh", "api", "-X", "POST", path]
    completed = subprocess.run(argv, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        msg = f"`gh api -X POST {path}` exited {completed.returncode}: {stderr}"
        raise ReceiptLookupError(msg)


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


def pr_head_receipt_key(
    receipt: ModelLabPassReceipt,
) -> tuple[str, int, str, str, str]:
    """Return the exact identity of a pre-merge proof receipt."""
    if receipt.lane is not EnumLabLane.PR_HEAD or receipt.subject is None:
        msg = "only a pr-head receipt has a PR-head receipt key"
        raise ValueError(msg)
    return (
        receipt.subject.repo,
        receipt.subject.pr_number,
        receipt.sha,
        receipt.subject.profile_id,
        receipt.subject.profile_version,
    )


def compute_pr_diff_digest(diff_bytes: bytes) -> str:
    """Hash the exact bytes emitted by ``git diff`` for a PR proof."""
    if not isinstance(diff_bytes, bytes):
        msg = f"diff_bytes must be bytes, got {type(diff_bytes).__name__}"
        raise TypeError(msg)
    return f"sha256:{hashlib.sha256(diff_bytes).hexdigest()}"


def compute_pr_diff_digest_from_repo(
    repo_path: Path, merge_base: str, head: str
) -> str:
    """Hash ``git diff merge_base..head`` without color or external drivers."""
    for label, value in (("merge_base", merge_base), ("head", head)):
        if not isinstance(value, str) or not _SHA_RE.match(value):
            msg = f"{label}={value!r} must be a 40-character lowercase sha"
            raise ValueError(msg)
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--no-color",
            "--no-ext-diff",
            f"{merge_base}..{head}",
        ],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return compute_pr_diff_digest(completed.stdout)


class EnumPrHeadVerdict(StrEnum):
    """Stable outcome tokens from the offline PR-head verifier."""

    ACCEPTED = "ACCEPTED"
    NOT_PR_HEAD = "NOT_PR_HEAD"
    HEAD_MISMATCH = "HEAD_MISMATCH"
    RESULT_NOT_PASS = "RESULT_NOT_PASS"
    MISSING_MANDATORY_CHECK = "MISSING_MANDATORY_CHECK"
    RUNNER_IS_VERIFIER = "RUNNER_IS_VERIFIER"
    CARRY_OVER_RUNTIME_PROFILE = "CARRY_OVER_RUNTIME_PROFILE"
    CARRY_OVER_DIGEST_MISMATCH = "CARRY_OVER_DIGEST_MISMATCH"
    PROFILE_MISMATCH = "PROFILE_MISMATCH"


_RUNTIME_PROOF_HANDLER_KINDS: Final[frozenset[EnumLabProofHandlerKind]] = frozenset(
    {
        EnumLabProofHandlerKind.RUNTIME_IMAGE,
        EnumLabProofHandlerKind.FOUNDATION_OVERRIDE,
        EnumLabProofHandlerKind.PYPI_SIBLING_OVERRIDE,
        EnumLabProofHandlerKind.K8S_NAMESPACE,
    }
)


def verify_pr_head_receipt(
    receipt: ModelLabPassReceipt,
    *,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_profile_id: str,
    expected_profile_version: str,
    mandatory_checks: frozenset[str],
    current_pr_diff_digest: str,
) -> tuple[EnumPrHeadVerdict, str]:
    """Verify one PR-head receipt against the current pull-request identity."""
    if receipt.lane is not EnumLabLane.PR_HEAD:
        return (
            EnumPrHeadVerdict.NOT_PR_HEAD,
            f"receipt lane {receipt.lane.value!r} is not 'pr-head'",
        )
    if receipt.sha != expected_head_sha:
        return (
            EnumPrHeadVerdict.HEAD_MISMATCH,
            f"receipt head {receipt.sha} does not match expected head {expected_head_sha}",
        )
    if receipt.result is not EnumLabPassResult.PASS:
        return (
            EnumPrHeadVerdict.RESULT_NOT_PASS,
            f"receipt result is {receipt.result.value}, not PASS",
        )

    # ModelLabPassReceipt refuses a pr-head receipt without this subject. Keep
    # the assertion local so type narrowing does not depend on that invariant.
    subject = receipt.subject
    if subject is None:  # pragma: no cover - construction already refuses it
        raise AssertionError("validated pr-head receipt has no subject")

    mismatches: list[str] = []
    if subject.repo != expected_repo:
        mismatches.append(f"repo={subject.repo!r}, expected {expected_repo!r}")
    if subject.pr_number != expected_pr_number:
        mismatches.append(
            f"pr_number={subject.pr_number}, expected {expected_pr_number}"
        )
    if subject.profile_id != expected_profile_id:
        mismatches.append(
            f"profile_id={subject.profile_id!r}, expected {expected_profile_id!r}"
        )
    if subject.profile_version != expected_profile_version:
        mismatches.append(
            "profile_version="
            f"{subject.profile_version!r}, expected {expected_profile_version!r}"
        )
    if mismatches:
        return (
            EnumPrHeadVerdict.PROFILE_MISMATCH,
            "receipt profile binding does not match: " + "; ".join(mismatches),
        )

    if not mandatory_checks:
        return (
            EnumPrHeadVerdict.MISSING_MANDATORY_CHECK,
            "no mandatory checks were supplied; a PR-head proof is judged against "
            "its profile's mandatory checks and an empty set would accept any PASS",
        )
    passing_names = {
        check.name
        for check in receipt.checks
        if check.outcome is EnumLabPassCheckOutcome.PASS
    }
    missing = sorted(mandatory_checks - passing_names)
    if missing:
        return (
            EnumPrHeadVerdict.MISSING_MANDATORY_CHECK,
            f"mandatory passing check(s) missing: {missing}",
        )

    if (
        subject.runner_identity.strip().casefold()
        == subject.verifier_identity.strip().casefold()
    ):
        return (
            EnumPrHeadVerdict.RUNNER_IS_VERIFIER,
            "runner_identity and verifier_identity name the same identity",
        )

    if subject.carried_from:
        if subject.handler_kind in _RUNTIME_PROOF_HANDLER_KINDS:
            return (
                EnumPrHeadVerdict.CARRY_OVER_RUNTIME_PROFILE,
                f"handler {subject.handler_kind.value!r} must re-prove at every head",
            )
        if subject.pr_diff_digest != current_pr_diff_digest:
            return (
                EnumPrHeadVerdict.CARRY_OVER_DIGEST_MISMATCH,
                "carried proof diff digest "
                f"{subject.pr_diff_digest} does not match current digest "
                f"{current_pr_diff_digest}",
            )

    return (
        EnumPrHeadVerdict.ACCEPTED,
        "PR-head receipt accepted for the exact head, profile, and required checks",
    )


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


#: How long ``gate --wait-seconds`` may poll at most. Six hours is the ceiling
#: GitHub places on a hosted job; a larger request could only ever be cut off
#: by the runner, which would read as a cancellation rather than as this gate's
#: own refusal.
MAX_WAIT_SECONDS: Final[int] = 6 * 60 * 60

#: Default spacing of ``--wait-seconds`` polls. Each poll is one exact-name
#: listing per lane, so this is gentle on the API while still being short next
#: to the rebuild durations a waiting gate is waiting out.
DEFAULT_POLL_SECONDS: Final[float] = 30.0


class EnumGateToken(StrEnum):
    """Why the gate refused, one token per outcome a reader acts on (OMN-19233).

    Plan PS-5 (decision S, 2026-09-23) separates the refusals that a later
    compose-dev PASS can change from the ones it cannot:

    * ``PENDING``: no receipt for the subject, and its rebuild is in flight;
    * ``ABSENT``: no receipt for the subject, and nothing is in flight (the
      PS-3 R1 path: a queued attempt 1 wrote none, a re-run attempt may);
    * ``INDETERMINATE``: a receipt whose only non-passing checks are
      INDETERMINATE (PS-3 R2), which asserts nothing about the lane;
    * ``FAIL``: a receipt carrying at least one FAIL check. It stands until the
      subject changes; no re-run is triggered for it;
    * ``UNREADABLE``: the surface or the artifact could not be read, or its name
      and payload disagree;
    * ``ANY_OF_UNMET``: the rule 24(b) any-of premise failed on the exact sha;
    * ``TIMED_OUT``: the read is past the overall bound from the delivery run's
      first gate read. Never a pass, even when a PASS has since arrived.
    """

    PASS = "PASS"
    PENDING = "PENDING"
    ABSENT = "ABSENT"
    INDETERMINATE = "INDETERMINATE"
    FAIL = "FAIL"
    UNREADABLE = "UNREADABLE"
    ANY_OF_UNMET = "ANY_OF_UNMET"
    TIMED_OUT = "TIMED_OUT"


#: Most severe first. The run's overall token is the most severe of its lanes'.
_TOKEN_SEVERITY: Final[tuple[EnumGateToken, ...]] = (
    EnumGateToken.TIMED_OUT,
    EnumGateToken.FAIL,
    EnumGateToken.UNREADABLE,
    EnumGateToken.ANY_OF_UNMET,
    EnumGateToken.INDETERMINATE,
    EnumGateToken.ABSENT,
    EnumGateToken.PENDING,
)

#: The refusals a later compose-dev PASS for the same subject can turn into a
#: pass, and so the only ones the emitter re-runs a delivery for (PS-5 item 2).
RERUN_ELIGIBLE_TOKENS: Final[frozenset[EnumGateToken]] = frozenset(
    {EnumGateToken.PENDING, EnumGateToken.ABSENT, EnumGateToken.INDETERMINATE}
)

#: PS-5 item 3, the overall bound, measured from the delivery run's first gate
#: read. Basis: the larger measured compose-dev queue bound (10,963 s,
#: 3e4aaded's first attempt at queue position 5) plus one verify-lane-converged
#: run at its 45-minute ceiling (2,700 s) = 13,663 s, rounded up to four hours.
#: Changing it is a ruling, not a tuning.
DELIVERY_OVERALL_BOUND_SECONDS: Final[int] = 14_400

VERDICT_SCHEMA: Final[str] = "lab_pass_gate_verdict.v1"


def verdict_artifact_name(run_id: int, run_attempt: int) -> str:
    """The artifact a delivery run's gate verdict is uploaded under, per attempt."""
    return f"lab-pass-gate-verdict-{run_id}-{run_attempt}"


def _utc_stamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def classify_receipt_token(receipt: ModelLabPassReceipt) -> EnumGateToken:
    """The token a present receipt earns: PASS, FAIL or INDETERMINATE-only."""
    if receipt.result is EnumLabPassResult.PASS:
        return EnumGateToken.PASS
    if any(c.outcome is EnumLabPassCheckOutcome.FAIL for c in receipt.checks):
        return EnumGateToken.FAIL
    return EnumGateToken.INDETERMINATE


def _most_severe(tokens: Sequence[EnumGateToken]) -> EnumGateToken:
    for token in _TOKEN_SEVERITY:
        if token in tokens:
            return token
    return EnumGateToken.PASS


@dataclass(frozen=True)
class ModelLaneRead:
    """What one read of one lane, for one subject sha, established.

    ``kind`` separates the four outcomes a reader acts on differently:
    ``receipt`` (a validated receipt was read; its verdict may still be FAIL),
    ``absent`` (the surface was read and holds no artifact by that name),
    ``unreadable`` (the surface or the artifact could not be read or parsed) and
    ``mismatch`` (an artifact exists but its payload disagrees with its name).
    Only ``absent`` and ``unreadable`` can change by waiting.
    """

    lane: EnumLabLane
    subject_sha: str
    kind: str
    receipt: ModelLabPassReceipt | None = None
    problem: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.receipt is not None and self.receipt.result is EnumLabPassResult.PASS
        )

    @property
    def may_still_arrive(self) -> bool:
        return self.kind in {"absent", "unreadable"}


def read_lane(repo: str, lane: EnumLabLane, sha: str) -> ModelLaneRead:
    """Read the newest receipt for one lane and one EXACT sha. Never raises."""
    name = artifact_name(lane, sha)
    try:
        artifacts = list_artifacts(repo, name)
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad. Rule 16: a verification sweep that errors and is
        # not caught reads as an absence of findings; here it would read as a
        # traceback with no sha in it. Any failure to READ the surface is
        # reported as "unreadable" against the named lane and fails the gate,
        # which is the same verdict as a missing receipt.
        return ModelLaneRead(lane, sha, "unreadable", problem=f"{lane.value}: {exc}")
    if not artifacts:
        return ModelLaneRead(
            lane,
            sha,
            "absent",
            problem=f"{lane.value}: no receipt artifact named {name}",
        )
    # Newest first: a re-run of the emitting job supersedes an earlier attempt
    # for the same sha and lane, in both directions.
    artifacts.sort(key=lambda a: str(a.get("created_at", "")), reverse=True)
    try:
        receipt = download_receipt(repo, int(artifacts[0]["id"]))
    except Exception as exc:  # noqa: BLE001 - same reasoning as above
        return ModelLaneRead(lane, sha, "unreadable", problem=f"{lane.value}: {exc}")
    if receipt.sha != sha:
        return ModelLaneRead(
            lane,
            sha,
            "mismatch",
            problem=(
                f"{lane.value}: artifact {name} carries sha {receipt.sha}; "
                "the name and the payload disagree"
            ),
        )
    if receipt.lane != lane:
        return ModelLaneRead(
            lane,
            sha,
            "mismatch",
            problem=(
                f"{lane.value}: artifact {name} carries lane {receipt.lane.value}; "
                "the name and the payload disagree"
            ),
        )
    return ModelLaneRead(lane, sha, "receipt", receipt=receipt)


def _read_all(
    repo: str,
    sha: str,
    lanes: Sequence[EnumLabLane],
    required: Sequence[EnumLabLane],
    required_sha: str,
) -> tuple[list[ModelLaneRead], list[ModelLaneRead]]:
    any_of = [read_lane(repo, lane, sha) for lane in lanes]
    cache = {(r.lane, r.subject_sha): r for r in any_of}
    all_of: list[ModelLaneRead] = []
    for lane in required:
        key = (lane, required_sha)
        if key not in cache:
            cache[key] = read_lane(repo, lane, required_sha)
        all_of.append(cache[key])
    return any_of, all_of


def _verdict(any_of: Sequence[ModelLaneRead], all_of: Sequence[ModelLaneRead]) -> bool:
    any_of_ok = not any_of or any(r.passed for r in any_of)
    return any_of_ok and all(r.passed for r in all_of)


def evaluate_gate(
    repo: str,
    sha: str,
    lanes: Sequence[EnumLabLane],
    out: Any,
    *,
    required: Sequence[EnumLabLane] = (),
    required_sha: str | None = None,
    required_note: str = "",
    wait_seconds: float = 0.0,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    rebuild_pending: Callable[[str], bool] | None = None,
    first_read_at: datetime | None = None,
    overall_bound_seconds: float | None = None,
    now: Callable[[], datetime] | None = None,
    verdict_out: Path | None = None,
    run_id: int | None = None,
    run_attempt: int | None = None,
) -> int:
    """Fail closed unless the lab-pass premise holds for the EXACT sha.

    Two requirements, both of which must hold:

    * ``lanes`` is ANY-OF (the rule 24(b) premise, unchanged since OMN-17530):
      at least one of them carries a PASS receipt for ``sha``. An empty
      ``lanes`` places no any-of requirement.
    * ``required`` is ALL-OF (OMN-19312): EVERY one of them carries a PASS
      receipt for ``required_sha`` (``sha`` unless a caller resolved an
      ancestor with ``resolve_required_subject``). A PASS on any other lane,
      in either set, never substitutes for a required lane's verdict -- that
      substitution is the defect this requirement exists to remove: before it,
      the onex-lab boot receipt alone satisfied every push delivery.

    ``wait_seconds`` bounds a poll for REQUIRED lanes whose receipt is absent or
    unreadable; the gate re-reads until the verdict passes, until no required
    lane can still change by waiting, or until the bound expires. Expiry is a
    refusal, never a pass. There is no force, skip or override of any kind.

    OMN-19233 (plan PS-5, decision S): every refusal carries one token
    (:class:`EnumGateToken`). A required compose-dev read with no receipt is
    PENDING when ``rebuild_pending(subject)`` says the subject's rebuild is in
    flight and ABSENT otherwise. With ``overall_bound_seconds``, a read more
    than that many seconds after ``first_read_at`` (the delivery run's first
    gate read; this read when ``None``) refuses TIMED_OUT whatever the receipts
    say. ``verdict_out`` receives the verdict as JSON, which the compose-dev
    emitter's re-run selector reads.

    Every terminal branch prints the sha. "The gate failed" with no commit named
    is unactionable at 3am, and the whole point of a sha-keyed receipt is that
    the answer is about one commit.
    """
    subject = sha if required_sha is None else required_sha
    clock = now if now is not None else (lambda: datetime.now(UTC))

    def _record(token: EnumGateToken, lane_tokens: Mapping[str, str]) -> None:
        if verdict_out is None:
            return
        read_at = clock()
        first = first_read_at if first_read_at is not None else read_at
        body = {
            "schema": VERDICT_SCHEMA,
            "repo": repo,
            "sha": sha,
            "subject": subject,
            "required_lanes": [lane.value for lane in required],
            "lanes": dict(lane_tokens),
            "token": token.value,
            "first_read_at": _utc_stamp(first),
            "read_at": _utc_stamp(read_at),
            "elapsed_seconds": int((read_at - first).total_seconds()),
            "bound_seconds": overall_bound_seconds,
            "run_id": run_id,
            "run_attempt": run_attempt,
        }
        verdict_out.parent.mkdir(parents=True, exist_ok=True)
        verdict_out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

    for label, value in (("commit", sha), ("required subject", subject)):
        if not _SHA_RE.match(value):
            print(
                f"::error::lab-pass gate: {label} {value!r} is not a 40-character "
                "lowercase commit sha. Refusing to resolve an abbreviated ref. "
                "token=UNREADABLE",
                file=out,
            )
            _record(EnumGateToken.UNREADABLE, {})
            return 1
    if not lanes and not required:
        print(
            f"::error::lab-pass gate FAILED for {sha}: no lane was named to read, "
            "so nothing could establish a pass. Refusing rather than passing a "
            "gate that checked nothing. token=UNREADABLE",
            file=out,
        )
        _record(EnumGateToken.UNREADABLE, {})
        return 1
    if wait_seconds < 0 or wait_seconds > MAX_WAIT_SECONDS:
        print(
            f"::error::lab-pass gate FAILED for {sha}: --wait-seconds "
            f"{wait_seconds:g} is outside 0..{MAX_WAIT_SECONDS}. token=UNREADABLE",
            file=out,
        )
        _record(EnumGateToken.UNREADABLE, {})
        return 1

    deadline = time.monotonic() + wait_seconds
    polls = 0
    while True:
        polls += 1
        any_of, all_of = _read_all(repo, sha, lanes, required, subject)
        if _verdict(any_of, all_of):
            break
        waiting_on = [r for r in all_of if r.may_still_arrive]
        remaining = deadline - time.monotonic()
        if not waiting_on or remaining <= 0:
            break
        pending = ", ".join(f"{r.lane.value}@{r.subject_sha[:12]}" for r in waiting_on)
        print(
            f"lab-pass gate: waiting for required lane(s) {pending} "
            f"(poll {polls}, {int(remaining)} s of {wait_seconds:g} s left)",
            file=out,
        )
        time.sleep(max(0.0, min(poll_seconds, remaining)))

    # OMN-19233: one token per required lane, and the overall bound.
    lane_tokens: dict[str, EnumGateToken] = {}
    lane_notes: dict[str, str] = {}
    for read in all_of:
        if read.passed:
            lane_tokens[read.lane.value] = EnumGateToken.PASS
        elif read.receipt is not None:
            lane_tokens[read.lane.value] = classify_receipt_token(read.receipt)
        elif read.kind == "absent":
            token = EnumGateToken.ABSENT
            if read.lane is EnumLabLane.COMPOSE_DEV and rebuild_pending is not None:
                try:
                    if rebuild_pending(read.subject_sha):
                        token = EnumGateToken.PENDING
                        lane_notes[read.lane.value] = (
                            "a rebuild run for this subject is in flight"
                        )
                    else:
                        lane_notes[read.lane.value] = (
                            "no rebuild run for this subject is in flight"
                        )
                except Exception as exc:  # noqa: BLE001 - named, and still refuses
                    lane_notes[read.lane.value] = (
                        f"whether a rebuild is in flight could not be read ({exc}), "
                        "so this is ABSENT, not PENDING"
                    )
            lane_tokens[read.lane.value] = token
        else:
            lane_tokens[read.lane.value] = EnumGateToken.UNREADABLE
    read_at = clock()
    first = first_read_at if first_read_at is not None else read_at
    elapsed = (read_at - first).total_seconds()
    timed_out = overall_bound_seconds is not None and elapsed > overall_bound_seconds

    print(f"lab-pass gate (rule 24(b), OMN-17530) for sha {sha}", file=out)
    print(f"  repository : {repo}", file=out)
    if overall_bound_seconds is not None:
        print(
            f"  bound      : first gate read {_utc_stamp(first)}, this read "
            f"{_utc_stamp(read_at)}, {int(elapsed)} s of {overall_bound_seconds:g} s "
            "(OMN-19233, PS-5)",
            file=out,
        )
    print(
        f"  any-of     : {', '.join(lane.value for lane in lanes) or '(none)'}",
        file=out,
    )
    if required:
        print(
            f"  all-of     : {', '.join(lane.value for lane in required)} "
            f"(OMN-19312) for subject {subject}",
            file=out,
        )
        if required_note:
            print(f"  subject    : {required_note}", file=out)
    if wait_seconds:
        print(f"  waited     : up to {wait_seconds:g} s, {polls} read(s)", file=out)
    rendered: set[tuple[EnumLabLane, str]] = set()
    for read in [*any_of, *all_of]:
        key = (read.lane, read.subject_sha)
        if key in rendered:
            continue
        rendered.add(key)
        if read.receipt is not None:
            print("", file=out)
            print(render_receipt(read.receipt), file=out)
        else:
            print(f"  unreadable : {read.problem}", file=out)

    if _verdict(any_of, all_of) and not timed_out:
        lanes_passing = ", ".join(
            dict.fromkeys(r.lane.value for r in [*any_of, *all_of] if r.passed)
        )
        print("", file=out)
        print(
            f"lab-pass gate PASSED for {sha} on lane(s): {lanes_passing}.",
            file=out,
        )
        _record(EnumGateToken.PASS, {k: v.value for k, v in lane_tokens.items()})
        return 0

    # OMN-18573. An INDETERMINATE check is named on its own line, with the sha
    # and the check's own evidence, BEFORE the generic refusal. The two are
    # different findings and a reader acts on them differently: a FAIL is a
    # question for the lane, an INDETERMINATE is a question for the hop that
    # was supposed to establish the fact. Collapsing them into one sentence is
    # what made the 2026-09-16/17 receipts read as lane failures.
    print("", file=out)
    seen: set[tuple[EnumLabLane, str, str]] = set()
    for read in [*any_of, *all_of]:
        if read.receipt is None:
            continue
        for check in read.receipt.checks:
            check_key = (read.lane, read.subject_sha, check.name)
            if (
                check.outcome is not EnumLabPassCheckOutcome.INDETERMINATE
                or check_key in seen
            ):
                continue
            seen.add(check_key)
            print(
                f"::error::lab-pass gate: for sha {read.subject_sha} on lane "
                f"{read.lane.value}, check {check.name!r} is INDETERMINATE and "
                f"asserts nothing about the lab lane: {check.evidence}. The sha is "
                "refused because an unestablished check is not a pass, NOT because "
                "the lane was shown to misbehave.",
                file=out,
            )

    for read in all_of:
        if read.passed:
            continue
        if read.receipt is not None:
            why = f"its newest receipt is {read.receipt.result.value}"
        elif read.kind == "absent":
            why = (
                f"no receipt exists after waiting {wait_seconds:g} s"
                if wait_seconds
                else "no receipt exists"
            )
        else:
            why = read.problem
        note = lane_notes.get(read.lane.value)
        if note:
            why = f"{why} ({note})"
        print(
            f"::error::lab-pass gate: REQUIRED lane {read.lane.value} does not pass "
            f"for sha {read.subject_sha}: {why}. "
            f"token={lane_tokens[read.lane.value].value}. A required lane is all-of "
            "(OMN-19312): a PASS on any other lane does not substitute for it.",
            file=out,
        )

    any_of_ok = not any_of or any(r.passed for r in any_of)
    refusals = [t for t in lane_tokens.values() if t is not EnumGateToken.PASS]
    if not any_of_ok:
        refusals.append(EnumGateToken.ANY_OF_UNMET)
    if timed_out:
        refusals.append(EnumGateToken.TIMED_OUT)
        print(
            f"::error::lab-pass gate: the read at {_utc_stamp(read_at)} is "
            f"{int(elapsed)} s after this delivery run's first gate read at "
            f"{_utc_stamp(first)}, past the {overall_bound_seconds:g} s bound "
            "(OMN-19233, PS-5). token=TIMED_OUT. A PASS that has arrived since "
            "does not pass a read past the bound; a new delivery run reads afresh.",
            file=out,
        )
    overall = _most_severe(refusals)
    print(
        f"::error::lab-pass gate REFUSED for {sha}: token={overall.value} "
        f"subject={subject} lanes="
        + (", ".join(f"{k}:{v.value}" for k, v in lane_tokens.items()) or "(none)"),
        file=out,
    )
    _record(overall, {k: v.value for k, v in lane_tokens.items()})
    if not any_of_ok:
        print(
            f"::error::lab-pass gate FAILED for {sha}: no PASS lab-pass receipt "
            f"exists for this exact sha on any of "
            f"{', '.join(lane.value for lane in lanes)}. "
            "Rule 24(b) — the lab is the first place a change runs; staging is "
            "promotion — so this candidate is not deliverable. This is not a skip: a "
            "missing, unreadable, malformed, INDETERMINATE or FAIL receipt all fail "
            "here, and there is no override flag. Exercise the sha on a lab lane and "
            "let its emitter publish the receipt.",
            file=out,
        )
    elif overall is EnumGateToken.TIMED_OUT and not any(
        t is not EnumGateToken.PASS for t in lane_tokens.values()
    ):
        # Every lane reads PASS now; the refusal is the bound alone, and the
        # TIMED_OUT line above says so. "A required lane does not carry a PASS"
        # would be false here.
        print(
            f"::error::lab-pass gate FAILED for {sha}: the read is past the overall "
            "bound. This is not a skip, and there is no override flag.",
            file=out,
        )
    else:
        print(
            f"::error::lab-pass gate FAILED for {sha}: a required lane does not "
            "carry a PASS receipt. This is not a skip: a missing, unreadable, "
            "malformed, INDETERMINATE or FAIL receipt on a required lane all fail "
            "here, and there is no override flag. Fix what the lane measures and "
            "let its emitter publish a PASS for this commit.",
            file=out,
        )
    return 1


# ---------------------------------------------------------------------------
# Which commit a REQUIRED lane's receipt is about (OMN-19312, reusing OMN-18664)
# ---------------------------------------------------------------------------
def _load_release_train() -> Any:
    """``release_train.py``, loaded by path from beside this module.

    Loaded lazily and only by ``--resolve-runtime-ancestor``: that module needs
    PyYAML, and every other path through this file must stay stdlib-only
    (``test_the_module_imports_nothing_outside_the_stdlib``). The rule is
    REUSED, not restated -- a second copy of the nearest-runtime-affecting-
    ancestor walk could disagree with the release train about which receipt
    describes a commit.
    """
    name = "_lab_pass_release_train"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "release_train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load the release train's ancestor rule from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def resolve_required_subject(
    sha: str,
    *,
    branch_commits: Callable[[], Sequence[str]],
    runtime_affecting: Callable[[str], bool],
) -> tuple[str, str]:
    """The commit a required lane's receipt is asked for, and why.

    Required verdict lanes emit only for RUNTIME-AFFECTING merges (their
    emitters run after the dev lane rebuild, which fires only for those), while
    delivery also fires for commits the classifier calls non-runtime. Asking
    such a commit for its own receipt would block it forever. So the subject is
    the nearest runtime-affecting commit at or below ``sha`` on first parent,
    by the OMN-18664 rule ``release_train.resolve_lab_candidate`` owns: a
    runtime-affecting ``sha`` resolves to itself, and inheritance crosses only
    non-runtime-affecting commits -- the walk stops at the FIRST runtime-
    affecting commit, so one between the subject and ``sha`` is impossible by
    construction.

    Every failure resolves to ``sha`` itself, which asks for the exact-commit
    receipt and refuses rather than inherits. The note says which happened.
    """
    try:
        train = _load_release_train()
        candidate = train.resolve_lab_candidate(
            sha, branch_commits=branch_commits, runtime_affecting=runtime_affecting
        )
    except Exception as exc:  # noqa: BLE001 - fail to the exact commit, named
        return sha, f"{sha} itself: the ancestor rule could not run ({exc})"
    if not candidate.sha:
        return sha, f"{sha} itself: {candidate.unresolved_reason}"
    if not candidate.skipped:
        return candidate.sha, f"{sha} itself (runtime-affecting)"
    skipped = ", ".join(s[:12] for s in candidate.skipped)
    return candidate.sha, (
        f"{candidate.sha}, the nearest runtime-affecting ancestor of {sha}, "
        f"inherited across {len(candidate.skipped)} non-runtime-affecting "
        f"commit(s): {skipped}"
    )


def resolve_required_subject_from_clone(
    sha: str, clone: Path, runtime_path_validator: Path, repo: str
) -> tuple[str, str]:
    """``resolve_required_subject`` over a local clone, with the TRIGGER's predicate.

    The predicate is the one the rebuild trigger and the release train read:
    ``is_runtime_affecting`` in ``scripts/runtime_change_classifier.py``, the
    path rule over omniclaude's deploy-gate validator unioned with the merged
    pull request's ``runtime_change`` label, read from ``repo`` (OMN-19318). A
    classifier that cannot be loaded, and a label that cannot be read, resolve
    to the exact commit.
    """
    try:
        train = _load_release_train()
        predicate = train.load_runtime_affecting(
            clone,
            runtime_path_validator,
            labels_for=lambda commit: train.default_merged_pr_labels(repo, commit),
        )
    except Exception as exc:  # noqa: BLE001 - fail to the exact commit, named
        return (
            sha,
            f"{sha} itself: the runtime-change classifier could not load ({exc})",
        )
    return resolve_required_subject(
        sha,
        branch_commits=lambda: train.default_branch_commits(clone, sha),
        runtime_affecting=predicate,
    )


# ---------------------------------------------------------------------------
# Receipt timing on staging delivery (OMN-19233, plan PS-5, decision S)
# ---------------------------------------------------------------------------
def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _gh_json(path: str) -> Any:
    try:
        raw = _gh_api(path)
    except ReceiptLookupError:
        raise
    except Exception as exc:
        raise ReceiptLookupError(str(exc)) from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"`gh api {path}` did not return JSON: {exc}"
        raise ReceiptLookupError(msg) from exc


def resolve_first_gate_read(
    repo: str,
    run_id: int,
    run_attempt: int,
    job_name: str,
    *,
    now: Callable[[], datetime],
) -> datetime:
    """When the delivery run's gate first read, across all of its attempts.

    The PS-5 bound is measured from the delivery run's FIRST gate read, so a
    re-run (the emitter's, or a person's) must not restart it. Attempt 1 is its
    own first read. A later attempt takes the earliest ``started_at`` of the gate
    job in any earlier attempt, read from the run's own jobs listing, which a
    caller cannot supply. A prior attempt whose gate job never started (skipped
    because an earlier job failed) did not read, and does not count.

    The job's start precedes the receipt read by the job's setup steps, so the
    bound this measures is slightly EARLIER than the literal read: the
    conservative direction. A listing that cannot be read raises
    ``ReceiptLookupError``, and the caller refuses UNREADABLE.
    """
    if run_attempt <= 1:
        return now()
    starts: list[datetime] = []
    for attempt in range(1, run_attempt):
        payload = _gh_json(
            f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100"
        )
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, list):
            msg = f"run {run_id} attempt {attempt}: the jobs listing carries no list"
            raise ReceiptLookupError(msg)
        for job in jobs:
            if not isinstance(job, dict) or job.get("name") != job_name:
                continue
            if job.get("conclusion") == "skipped":
                continue
            started = _parse_utc(job.get("started_at"))
            if started is not None:
                starts.append(started)
    return min(starts) if starts else now()


@dataclass(frozen=True)
class ModelDeliveryRun:
    """One staging delivery run, as the re-run selector reads it."""

    run_id: int
    created_at: datetime
    status: str
    conclusion: str | None
    run_attempt: int
    #: The gate verdict uploaded by this run's LATEST attempt, or None.
    verdict: Mapping[str, Any] | None


def _ineligible(
    run: ModelDeliveryRun,
    subject: str,
    *,
    newest: ModelDeliveryRun,
    now: datetime,
    bound_seconds: float,
) -> str:
    if run is not newest:
        return (
            f"a newer delivery run {newest.run_id} exists; re-running this one "
            "would deliver older code over it"
        )
    if run.status != "completed":
        return f"still {run.status}; it will read afresh"
    if run.conclusion != "failure":
        return f"concluded {run.conclusion}, not a refusal"
    verdict = run.verdict
    if verdict is None:
        return (
            f"attempt {run.run_attempt} uploaded no gate verdict (the gate step "
            "did not run), so nothing says it refused on compose-dev timing"
        )
    if (
        verdict.get("run_id") != run.run_id
        or verdict.get("run_attempt") != run.run_attempt
    ):
        return (
            f"its verdict names run {verdict.get('run_id')} attempt "
            f"{verdict.get('run_attempt')}, not attempt {run.run_attempt}"
        )
    if verdict.get("subject") != subject:
        return f"it refused for subject {verdict.get('subject')}, not {subject}"
    token = str(verdict.get("token"))
    if token not in {t.value for t in RERUN_ELIGIBLE_TOKENS}:
        if token == EnumGateToken.FAIL.value:
            return "it refused on a FAIL check, which stands until the subject changes"
        return f"it refused {token}, which a compose-dev PASS does not change"
    first = _parse_utc(verdict.get("first_read_at"))
    if first is None:
        return "its verdict carries no readable first_read_at"
    elapsed = (now - first).total_seconds()
    if elapsed > bound_seconds:
        return (
            f"its first gate read was {int(elapsed)} s ago, past the "
            f"{bound_seconds:g} s bound; a re-run would refuse TIMED_OUT"
        )
    return ""


def select_refused_deliveries(
    subject: str,
    runs: Sequence[ModelDeliveryRun],
    *,
    now: datetime,
    bound_seconds: float,
) -> tuple[list[ModelDeliveryRun], list[str]]:
    """The delivery runs a compose-dev PASS for ``subject`` should re-run.

    PS-5 item 2: a run is re-run when its latest attempt's gate refused
    PENDING, ABSENT or INDETERMINATE for this exact subject, inside the bound.
    A FAIL refusal is never re-run. One rule beyond the plan: only the NEWEST
    delivery run is ever eligible, because re-running an older one would
    deliver older code over newer code on staging (and would cancel the newer
    run in their shared concurrency group). Every run gets a reason line.
    """
    ordered = sorted(runs, key=lambda r: r.created_at, reverse=True)
    selected: list[ModelDeliveryRun] = []
    reasons: list[str] = []
    for run in ordered:
        why = _ineligible(
            run, subject, newest=ordered[0], now=now, bound_seconds=bound_seconds
        )
        if why:
            reasons.append(f"delivery run {run.run_id}: not re-run: {why}")
            continue
        selected.append(run)
        reasons.append(
            f"delivery run {run.run_id}: RE-RUN: attempt {run.run_attempt} refused "
            f"{run.verdict.get('token') if run.verdict else ''} for subject "
            f"{subject}, which now carries a compose-dev PASS"
        )
    return selected, reasons


def read_gate_verdict(
    repo: str, run_id: int, run_attempt: int
) -> Mapping[str, Any] | None:
    """The gate verdict one delivery attempt uploaded, or None when it has none."""
    artifacts = list_artifacts(repo, verdict_artifact_name(run_id, run_attempt))
    if not artifacts:
        return None
    artifacts.sort(key=lambda a: str(a.get("created_at", "")), reverse=True)
    artifact_id = int(artifacts[0]["id"])
    blob = _gh_api(f"repos/{repo}/actions/artifacts/{artifact_id}/zip")
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            members = [n for n in archive.namelist() if n.endswith("verdict.json")]
            if len(members) != 1:
                msg = (
                    f"verdict artifact {artifact_id} carries {len(members)} "
                    "verdict.json entries; exactly one is required."
                )
                raise ReceiptLookupError(msg)
            body = json.loads(archive.read(members[0]).decode("utf-8"))
    except (zipfile.BadZipFile, json.JSONDecodeError) as exc:
        msg = f"verdict artifact {artifact_id} is unreadable: {exc}"
        raise ReceiptLookupError(msg) from exc
    if not isinstance(body, dict) or body.get("schema") != VERDICT_SCHEMA:
        msg = f"verdict artifact {artifact_id} is not a {VERDICT_SCHEMA} document"
        raise ReceiptLookupError(msg)
    return body


def read_delivery_runs(
    repo: str, workflow: str, branch: str, *, limit: int = 20
) -> list[ModelDeliveryRun]:
    """Recent delivery runs, newest first. Only the newest can be re-run, so
    only its verdict is fetched. A surface that cannot be read raises."""
    payload = _gh_json(
        f"repos/{repo}/actions/workflows/{workflow}/runs?branch={branch}"
        f"&per_page={limit}"
    )
    raw_runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(raw_runs, list):
        msg = f"the {workflow} run listing carries no workflow_runs list"
        raise ReceiptLookupError(msg)
    runs: list[ModelDeliveryRun] = []
    for raw in raw_runs:
        created = _parse_utc(raw.get("created_at")) if isinstance(raw, dict) else None
        if created is None:
            msg = f"a {workflow} run carries no readable created_at: {raw!r}"
            raise ReceiptLookupError(msg)
        runs.append(
            ModelDeliveryRun(
                run_id=int(raw["id"]),
                created_at=created,
                status=str(raw.get("status")),
                conclusion=raw.get("conclusion"),
                run_attempt=int(raw.get("run_attempt") or 1),
                verdict=None,
            )
        )
    runs.sort(key=lambda r: r.created_at, reverse=True)
    if runs and runs[0].status == "completed" and runs[0].conclusion == "failure":
        newest = runs[0]
        runs[0] = ModelDeliveryRun(
            run_id=newest.run_id,
            created_at=newest.created_at,
            status=newest.status,
            conclusion=newest.conclusion,
            run_attempt=newest.run_attempt,
            verdict=read_gate_verdict(repo, newest.run_id, newest.run_attempt),
        )
    return runs


def endpoint_reemit_eligibility(
    subject: str,
    *,
    repo: str,
    workflow: str,
    branch: str,
    bound_seconds: float,
    now: datetime,
    read_runs: Callable[[str, str, str], list[ModelDeliveryRun]] = read_delivery_runs,
) -> str:
    """Decide whether an absent lower-endpoint receipt may be recovered.

    OMN-19563. A queued verify run emits no receipt. When its command deploys
    after that run ends, the next convergence observes the queued sha as its
    exact initial revision -- the bounded lower endpoint -- rather than inside
    the open re-emission window. The caller has already re-established live
    containment for that endpoint; this second predicate proves the newest
    staging delivery is waiting for the same exact subject inside its bound.

    Any unreadable delivery surface stays closed. This function only admits an
    absent receipt; unreadable or health-failing receipts are rejected by the
    existing eligibility path before this function is called.
    """
    if not _SHA_RE.match(subject):
        return "skip:endpoint-subject-invalid"
    try:
        runs = read_runs(repo, workflow, branch)
    except Exception as exc:  # noqa: BLE001 - refusal is the fail-closed result
        detail = " ".join(str(exc).split()) or type(exc).__name__
        return f"skip:endpoint-delivery-unreadable {detail}"
    selected, _reasons = select_refused_deliveries(
        subject,
        runs,
        now=now,
        bound_seconds=bound_seconds,
    )
    if selected:
        return "reemit:absent-endpoint-delivery-waiting"
    return "skip:endpoint-has-no-waiting-delivery"


def rerun_refused_deliveries(
    repo: str,
    subjects: Sequence[str],
    *,
    workflow: str,
    branch: str,
    bound_seconds: float,
    out: Any,
    now: Callable[[], datetime],
) -> int:
    """The compose-dev emitter's half of PS-5: re-run what a PASS unblocks.

    For each subject that now carries a compose-dev PASS (read back through the
    same reader the gate uses, never assumed from the emitting job), re-run the
    failed jobs of the delivery runs :func:`select_refused_deliveries` picks. A
    subject whose receipt is not PASS re-runs nothing. An unreadable surface or
    a refused re-run request fails the step, loudly: a delivery nobody re-reads
    stays refused, and that must be visible.
    """
    failures = 0
    runs: list[ModelDeliveryRun] | None = None
    for subject in dict.fromkeys(subjects):
        if not _SHA_RE.match(subject):
            print(
                f"::error::rerun-refused-deliveries: subject {subject!r} is not a "
                "40-character lowercase commit sha",
                file=out,
            )
            failures += 1
            continue
        read = read_lane(repo, EnumLabLane.COMPOSE_DEV, subject)
        if not read.passed:
            state = read.receipt.result.value if read.receipt is not None else read.kind
            line = (
                f"subject {subject}: compose-dev receipt is {state}, not PASS; "
                "no delivery is re-run for it"
            )
            if read.kind in {"unreadable", "mismatch"}:
                print(
                    f"::error::rerun-refused-deliveries: {line}: {read.problem}",
                    file=out,
                )
                failures += 1
            else:
                print(line, file=out)
            continue
        if runs is None:
            try:
                runs = read_delivery_runs(repo, workflow, branch)
            except Exception as exc:  # noqa: BLE001 - reported, and fails the step
                print(
                    f"::error::rerun-refused-deliveries: the {workflow} runs could "
                    f"not be read: {exc}",
                    file=out,
                )
                return 1
        selected, reasons = select_refused_deliveries(
            subject, runs, now=now(), bound_seconds=bound_seconds
        )
        print(
            f"subject {subject}: compose-dev receipt is PASS; required lane "
            "compose-dev (OMN-19233)",
            file=out,
        )
        for reason in reasons:
            print(f"  {reason}", file=out)
        for run in selected:
            path = f"repos/{repo}/actions/runs/{run.run_id}/rerun-failed-jobs"
            try:
                _gh_api_post(path)
            except Exception as exc:  # noqa: BLE001 - reported, and fails the step
                print(
                    f"::error::rerun-refused-deliveries: re-running delivery run "
                    f"{run.run_id} failed: {exc}",
                    file=out,
                )
                failures += 1
                continue
            print(f"  re-ran the failed jobs of delivery run {run.run_id}", file=out)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# workflow-verdict (OMN-18866, ruling 2026-09-23T17:12:15Z)
# ---------------------------------------------------------------------------
#: The widest freshness window a caller may ask for. The reader exists to bind
#: a RECURRING measurement to a delivery; a window wide enough to span several
#: of its cadences would turn "the newest measurement" into "some measurement",
#: and a wider window is exactly how a stale green would be laundered past a
#: fresh red. 72 hours covers a nightly with two missed nights and nothing more.
WORKFLOW_VERDICT_MAX_AGE_CEILING_HOURS: Final[float] = 72.0

#: How many completed runs are read. The newest attempt wins, and an attempt's
#: start moves forward on a re-run, so the newest attempt is not necessarily the
#: newest-CREATED run; reading a page rather than one row is what lets a re-run
#: of an older run supersede a newer-created one, in both directions.
WORKFLOW_VERDICT_PAGE: Final[int] = 50

#: OMN-19311 AC5 -- the planted-red drill's blast-radius ceiling. `until` may
#: not be set more than this many hours in the future, so a mistyped date
#: cannot leave staging delivery closed for days with nobody watching. The
#: drill exists to PROVE a refusal happens, not to hold one open.
DRILL_MAX_WINDOW_HOURS: Final[float] = 2.0

_TICKET_RE = re.compile(r"^OMN-\d+$")


def _run_started(run: Mapping[str, Any]) -> datetime | None:
    """The start of a run's LATEST attempt, which is when it last measured."""
    raw = run.get("run_started_at") or run.get("created_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return _parse_ts(raw)
    except ValueError:
        return None


def evaluate_workflow_verdict(
    repo: str,
    workflow: str,
    branch: str,
    max_age_hours: float,
    events: Sequence[str],
    out: Any,
    *,
    now: datetime | None = None,
    dispatch_title_contains: str = "",
    drill_red_until: str = "",
    drill_ticket: str = "",
) -> int:
    """Fail closed unless the NEWEST completed measurement of a workflow is green.

    ``drill_red_until`` / ``drill_ticket`` (OMN-19311 AC5) are the
    planted-red drill: a self-expiring switch that makes THIS reader return a
    red, so staging delivery's refusal path can be proven live without
    depending on a real defect existing at proof time (the real D11 red this
    binding was written against, 2026-09-23, is not guaranteed to stay red --
    it turned green at 2026-09-24T05:33:57Z). Both must be set together or
    neither is (a lone flag is a misconfiguration, refused); ``drill_ticket``
    must be an ``OMN-<n>`` id and ``drill_red_until`` a parseable UTC
    timestamp no more than :data:`DRILL_MAX_WINDOW_HOURS` in the future. Past
    its own expiry the drill has NO effect at all -- nobody has to turn it
    off. There is deliberately no symmetric flag that forces a PASS: the drill
    can only make delivery more conservative, never less. When active it does
    not call the GitHub API at all -- it is a synthetic, clearly-labelled red,
    not a spoofed real one.

    Operator ruling 2026-09-23T17:12:15Z (ledger RULING, amending the
    16:10:08Z every-check-blocks ruling) binds four chronically red checks to a
    blocking surface NOW, not once they go green. Two of them -- C15, this
    repository's ``chain-canary.yml``, and D11, omnimarket's delegation
    regression nightly -- measure a DEPLOYED lab lane on a schedule, so no PR
    head can run them and a merge block would stop the fix itself. Their
    surface is staging delivery, and until each emits a sha-keyed receipt of its
    own (the design ``PROBES_NOT_YET_WIRED`` records), the only verdict that
    exists is the workflow's own conclusion. This reads it.

    STATE-KEYED, NOT SHA-KEYED, and said so in the output: the verdict is about
    the lane the workflow measured, at the time it measured it, not about the
    delivered commit. It is the same fact a board reads from a run conclusion.

    NEWEST WINS, IN BOTH DIRECTIONS. The newest completed attempt among the
    admitted events decides. A re-run is a fresh measurement: a red re-run
    supersedes an earlier green and a green one an earlier red. There is no
    "any green in the window" reading, because that is how a check that fails
    three runs in four would pass.

    FAIL-CLOSED EVERYWHERE, each refusal naming the run: an unreadable surface,
    no admitted completed run, a newest conclusion other than ``success``
    (``failure``, ``cancelled``, ``timed_out``, ``startup_failure``,
    ``skipped``, ``neutral``, ``action_required``, ``stale`` -- a cancelled or
    runner-starved run is an instrument fault, and it BLOCKS like any red), a
    newest measurement older than ``max_age_hours``, or an unparseable start.
    There is no override, and a manual dispatch with non-default inputs is not
    admitted unless the caller admits its event.

    ``dispatch_title_contains`` (OMN-19311, D11) narrows an admitted
    ``workflow_dispatch`` further: such a run is a measurement only when its run
    title carries the token. D11's nightly takes a ``lane`` input and renders it
    into its ``run-name``; admitting its dispatches (the fast path to a fresh
    measurement after a fix) without this would let a green dispatch aimed at
    the dev lane stand in for the governed stability-test verdict, dropping the
    red by changing what is measured. Runs of every other admitted event are
    unaffected: a scheduled run takes no inputs and measures the default lane.
    """
    moment = now or datetime.now(tz=UTC)
    admitted = tuple(dict.fromkeys(e.strip() for e in events if e.strip()))
    label = f"{repo} {workflow} on {branch}"

    def refuse(reason: str) -> int:
        print(
            f"::error::workflow verdict FAILED for {label}: {reason} "
            "This is a blocking gate (operator ruling 2026-09-23T17:12:15Z): "
            "delivery stops until the workflow's newest measurement is green "
            "and fresh. There is no override flag; fix the defect it measures, "
            "or the instrument, and let the next run (or a re-run, which is a "
            "fresh measurement) decide.",
            file=out,
        )
        return 1

    # OMN-19311 AC5 -- the planted-red drill. Checked before anything else,
    # including the GitHub API read: an active drill never depends on network
    # access, so it is a reliable, fast, unambiguously-synthetic red.
    if drill_red_until or drill_ticket:
        if not drill_red_until or not drill_ticket:
            return refuse(
                "DRILL misconfigured (OMN-19311 AC5): drill_red_until and "
                "drill_ticket must both be set together, or neither is set "
                f"(drill_red_until={drill_red_until!r}, "
                f"drill_ticket={drill_ticket!r})."
            )
        if not _TICKET_RE.match(drill_ticket):
            return refuse(
                f"DRILL misconfigured (OMN-19311 AC5): drill_ticket "
                f"{drill_ticket!r} is not an OMN-<n> id."
            )
        try:
            drill_until = _parse_ts(drill_red_until)
        except ValueError:
            return refuse(
                f"DRILL misconfigured (OMN-19311 AC5): drill_red_until "
                f"{drill_red_until!r} is not a parseable UTC timestamp."
            )
        window_hours = (drill_until - moment).total_seconds() / 3600.0
        if window_hours > DRILL_MAX_WINDOW_HOURS:
            return refuse(
                "DRILL misconfigured (OMN-19311 AC5): drill_red_until is "
                f"{window_hours:.2f}h in the future, past the "
                f"{DRILL_MAX_WINDOW_HOURS:g}h ceiling ({drill_ticket})."
            )
        if window_hours > 0:
            print(
                f"::warning::workflow verdict DRILL (OMN-19311 AC5) active for "
                f"{label}: a self-expiring switch forcing this reader to return "
                f"red until {drill_red_until} ({window_hours:.2f}h "
                f"remaining), ticket {drill_ticket}. This proves staging "
                "delivery's refusal path live; it is not a real defect and no "
                "GitHub API call was made to produce it.",
                file=out,
            )
            return refuse(
                f"DRILL (OMN-19311 AC5, ticket {drill_ticket}) forcing red "
                f"until {drill_red_until}; this is a synthetic "
                "measurement, not a real one."
            )
        print(
            f"workflow verdict DRILL (OMN-19311 AC5) for {label}: window "
            f"expired at {drill_red_until} ({drill_ticket}); resuming "
            "ordinary evaluation with no further action needed.",
            file=out,
        )

    print(f"workflow verdict (OMN-18866) for {label}", file=out)
    if not (0 < max_age_hours <= WORKFLOW_VERDICT_MAX_AGE_CEILING_HOURS):
        return refuse(
            f"--max-age-hours {max_age_hours} is outside (0, "
            f"{WORKFLOW_VERDICT_MAX_AGE_CEILING_HOURS:g}]; a window that wide "
            "is not a freshness bound."
        )
    if not admitted:
        return refuse("no event is admitted, so no run can decide the verdict.")

    path = (
        f"repos/{repo}/actions/workflows/{workflow}/runs"
        f"?branch={branch}&status=completed&per_page={WORKFLOW_VERDICT_PAGE}"
    )
    try:
        payload = json.loads(_gh_api(path).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - rule 16: an unread surface is a refusal
        return refuse(f"the run surface is unreadable: {exc}.")
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        return refuse("the run listing carries no 'workflow_runs' list.")

    candidates = [
        r
        for r in runs
        if isinstance(r, dict)
        and r.get("status") == "completed"
        and r.get("head_branch") == branch
        and r.get("event") in admitted
        and (
            not dispatch_title_contains
            or r.get("event") != "workflow_dispatch"
            or dispatch_title_contains in str(r.get("display_title", ""))
        )
    ]
    ignored = len(runs) - len(candidates)
    print(
        f"  events read : {', '.join(admitted)} "
        f"({len(candidates)} completed run(s) admitted, {ignored} other(s) ignored)",
        file=out,
    )
    if dispatch_title_contains:
        print(
            f"  dispatches  : admitted only when the run title carries "
            f"{dispatch_title_contains!r}",
            file=out,
        )
    if not candidates:
        return refuse(
            f"no completed {'/'.join(admitted)} run exists on {branch}; an absent "
            "measurement is not a pass."
        )

    dated: list[tuple[datetime, Mapping[str, Any]]] = []
    for run in candidates:
        run_start = _run_started(run)
        if run_start is None:
            return refuse(f"completed run {run.get('id')} has no parseable start time.")
        dated.append((run_start, run))
    started, newest = max(dated, key=lambda pair: pair[0])
    age_hours = (moment - started).total_seconds() / 3600.0
    run_id = newest.get("id")
    conclusion = str(newest.get("conclusion"))
    head = str(newest.get("head_sha", "?"))
    print(
        f"  newest run  : {run_id} ({newest.get('event')}, attempt "
        f"{newest.get('run_attempt', '?')}) conclusion={conclusion}",
        file=out,
    )
    print(
        f"  measured at : {started.isoformat()} (age {age_hours:.2f}h, bound "
        f"{max_age_hours:g}h)",
        file=out,
    )
    print(f"  run head    : {head} (the workflow file's sha, not the lane's)", file=out)
    print(f"  run title   : {newest.get('display_title', '?')}", file=out)
    print(f"  url         : {newest.get('html_url', '?')}", file=out)

    if conclusion != "success":
        return refuse(
            f"its newest measurement, run {run_id} at {head}, concluded {conclusion!r}."
        )
    if age_hours > max_age_hours:
        return refuse(
            f"its newest measurement, run {run_id}, is {age_hours:.2f}h old, past "
            f"the {max_age_hours:g}h bound; a stale green is not a current one."
        )
    if age_hours < -0.25:
        return refuse(
            f"run {run_id} starts {-age_hours:.2f}h in the future; the clock "
            "reading cannot be trusted."
        )
    print(
        f"workflow verdict PASSED for {label}: run {run_id} concluded success "
        f"{age_hours:.2f}h ago.",
        file=out,
    )
    return 0


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
        "--health-observe-budget-seconds",
        type=float,
        default=None,
        help=(
            "seconds to keep asking for details.runtime_health after the lane "
            "is READY (OMN-18886). That block is populated ASYNCHRONOUSLY and "
            "carries its own observed_at, so a single sample taken between "
            "readiness and the first observation finds it absent and reports a "
            "healthy lane as failing. The default of 0 preserves the "
            "single-sample behaviour for an ad hoc read. It never turns an "
            "unhealthy dimension into a pass: the terminal failure is 'STILL "
            "unhealthy after the full budget', naming the dimensions."
        ),
    )
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
        "--consumer-groups-file",
        type=Path,
        default=None,
        help=(
            "a file of declared group names, one per line, as written by "
            "scripts/runtime_build/declared_consumer_groups.py --out. "
            "PREFERRED over --consumer-groups: an absent file and an empty "
            "file are distinguishable, while an unset step output and an "
            "empty declaration are the same bytes -- which is exactly how "
            "OMN-18866's first wiring reported 'this lane declares no groups' "
            "about a lane that declares six."
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
    group_src = sample.add_mutually_exclusive_group(required=True)
    group_src.add_argument("--consumer-groups")
    group_src.add_argument(
        "--consumer-groups-file",
        type=Path,
        help="the file written by declared_consumer_groups.py --out. Preferred, "
        "for the reason load_declared_groups records.",
    )
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
    emit.add_argument(
        "--converged-via",
        default="",
        help=(
            "OMN-18976. The sha whose convergence produced this receipt, when "
            "that is not --sha itself. Set only when answering for a merge "
            "that was queued behind another deploy and emitted no receipt of "
            "its own; the probes must have been taken against an image "
            "containing --sha, which a containing revision is."
        ),
    )
    emit.add_argument("--out", required=True, type=Path)
    emit.add_argument(
        "--event-out",
        type=Path,
        default=None,
        help=(
            "also write the bus event document (OMN-18769) to this path. The "
            "emitting job publishes it with rpk when it has a broker; this "
            "script never opens a broker connection itself, because the "
            "emitting jobs differ in whether they can reach one and a "
            "publish failure must never fail a lab pass that genuinely ran."
        ),
    )

    reemit = sub.add_parser(
        "reemit",
        help="re-key a converged receipt onto a sha that was queued behind it",
    )
    reemit.add_argument(
        "--from",
        dest="source",
        required=True,
        type=Path,
        help="the receipt written by the run that converged",
    )
    reemit.add_argument(
        "--sha",
        required=True,
        help="the queued sha this receipt is being written for",
    )
    reemit.add_argument(
        "--converged-via",
        default="",
        dest="reemit_converged_via",
        help=(
            "OMN-18988. The provenance string from `bind-lane`, naming the sha "
            "AND the surface that established the lane runs code containing "
            "--sha. Omitted on the queued-ahead path, where the converged run "
            "bound its own observation and its sha is the provenance."
        ),
    )
    reemit.add_argument("--out", required=True, type=Path)

    elig = sub.add_parser(
        "reemit-eligible",
        help="decide whether a candidate sha may be answered for",
    )
    elig.add_argument(
        "--existing",
        type=Path,
        default=None,
        help="the candidate's existing receipt, or omitted when it has none",
    )
    elig.add_argument(
        "--endpoint",
        action="store_true",
        help=(
            "this candidate is the window's LOWER endpoint, the revision the "
            "lane was already on. An endpoint with no receipt is eligible only "
            "when the newest staging delivery is waiting for its exact sha."
        ),
    )
    elig.add_argument(
        "--endpoint-subject",
        default="",
        help=(
            "OMN-19563: exact lower-endpoint sha. When its receipt is absent, "
            "re-emission is allowed only while the newest staging delivery is "
            "waiting for this subject."
        ),
    )
    elig.add_argument("--repo", default=DEFAULT_REPO)
    elig.add_argument(
        "--delivery-workflow",
        default="deliver-dev-candidate-to-staging.yml",
    )
    elig.add_argument("--delivery-branch", default="dev")
    elig.add_argument("--overall-bound-seconds", type=float, default=14_400)

    gate = sub.add_parser("gate", help="fail closed unless a PASS receipt exists")
    gate.add_argument("--sha", required=True)
    gate.add_argument("--repo", default=DEFAULT_REPO)
    gate.add_argument(
        "--lane",
        action="append",
        default=[],
        choices=[e.value for e in EnumLabLane],
        help=(
            "repeatable, ANY-OF: one PASS among these satisfies the rule 24(b) "
            "premise. Defaults to compose-dev, onex-lab and onex-lab-k3s"
        ),
    )
    gate.add_argument(
        "--instance-lanes-for",
        default="",
        metavar="REPO",
        help=(
            "OMN-19543: also read, ANY-OF, the receipt lane of every deploy-agent "
            "instance whose row in config/deploy_lane_routing.yaml proves REPO "
            "(omnimarket today: compose-dev-202, compose-dev-200). Needs --lane. "
            "An unreadable table or a lane this enum does not declare refuses"
        ),
    )
    gate.add_argument(
        "--require-lane",
        action="append",
        default=[],
        choices=[e.value for e in EnumLabLane],
        help=(
            "repeatable, ALL-OF (OMN-19312): every named lane must carry its own "
            "PASS receipt; no other lane's PASS substitutes"
        ),
    )
    gate.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help=(
            "poll up to this long for a REQUIRED lane's receipt that is absent "
            f"or unreadable (0..{MAX_WAIT_SECONDS}); expiry refuses"
        ),
    )
    gate.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="spacing of the --wait-seconds polls",
    )
    gate.add_argument(
        "--resolve-runtime-ancestor",
        action="store_true",
        help=(
            "ask REQUIRED lanes for the nearest runtime-affecting first-parent "
            "ancestor's receipt (the OMN-18664 release-train rule); needs "
            "--clone and --runtime-path-validator"
        ),
    )
    gate.add_argument("--clone", type=Path, default=None)
    gate.add_argument("--runtime-path-validator", type=Path, default=None)
    gate.add_argument(
        "--detect-pending",
        action="store_true",
        help=(
            "OMN-19233: a required compose-dev receipt that is absent refuses "
            "PENDING while the subject's rebuild run is in flight, ABSENT otherwise"
        ),
    )
    gate.add_argument(
        "--overall-bound-seconds",
        type=float,
        default=None,
        help=(
            "OMN-19233 (PS-5): refuse TIMED_OUT when this read is more than this "
            "many seconds after the delivery run's first gate read"
        ),
    )
    gate.add_argument("--run-id", type=int, default=None)
    gate.add_argument("--run-attempt", type=int, default=None)
    gate.add_argument(
        "--gate-job-name",
        default="",
        help="the gate job's name, whose earliest start is the first gate read",
    )
    gate.add_argument(
        "--verdict-out",
        type=Path,
        default=None,
        help="write the gate verdict as JSON (the re-run selector reads it)",
    )

    rerun = sub.add_parser(
        "rerun-refused-deliveries",
        help=(
            "OMN-19233 (PS-5 item 2): after a compose-dev PASS, re-run the staging "
            "delivery that refused PENDING, ABSENT or INDETERMINATE for it"
        ),
    )
    rerun.add_argument("--repo", default=DEFAULT_REPO)
    rerun.add_argument("--subject", action="append", required=True)
    rerun.add_argument("--workflow", required=True)
    rerun.add_argument("--branch", default="dev")
    rerun.add_argument("--overall-bound-seconds", type=float, required=True)

    verdict = sub.add_parser(
        "workflow-verdict",
        help=(
            "fail closed unless the newest completed run of a workflow on a "
            "branch concluded success within a freshness bound (OMN-18866)"
        ),
    )
    verdict.add_argument("--repo", required=True)
    verdict.add_argument("--workflow", required=True, help="the workflow file name")
    verdict.add_argument("--branch", required=True)
    verdict.add_argument("--max-age-hours", required=True, type=float)
    verdict.add_argument(
        "--event",
        action="append",
        required=True,
        help=(
            "repeatable; the run events admitted as a measurement. A run of any "
            "other event is ignored, never counted as a pass."
        ),
    )
    verdict.add_argument(
        "--dispatch-title-contains",
        default="",
        help=(
            "OMN-19311: an admitted workflow_dispatch run is a measurement only "
            "when its run title carries this token (the lane it measured)"
        ),
    )
    verdict.add_argument(
        "--drill-red-until",
        default="",
        help=(
            "OMN-19311 AC5: the planted-red drill. A UTC timestamp "
            "(YYYY-MM-DDTHH:MM:SSZ), no more than DRILL_MAX_WINDOW_HOURS in "
            "the future, past which this flag has no effect at all. Requires "
            "--drill-ticket. Forces this reader to return red until then, "
            "proving staging delivery's refusal path live without depending "
            "on a real red existing."
        ),
    )
    verdict.add_argument(
        "--drill-ticket",
        default="",
        help="OMN-19311 AC5: the OMN-<n> ticket authorizing the drill. Required with --drill-red-until.",
    )

    verify = sub.add_parser(
        "verify",
        help="refuse a receipt file that is not the one this job emitted",
    )
    verify.add_argument("--path", required=True, type=Path)
    verify.add_argument("--sha", required=True)
    verify.add_argument("--lane", required=True, choices=[e.value for e in EnumLabLane])

    verify_pr_head = sub.add_parser(
        "verify-pr-head",
        help="verify an exact-head PR lab-proof receipt offline",
    )
    verify_pr_head.add_argument("--receipt", required=True, type=Path)
    verify_pr_head.add_argument("--repo", required=True)
    verify_pr_head.add_argument("--pr", required=True, type=int)
    verify_pr_head.add_argument("--head-sha", required=True)
    verify_pr_head.add_argument("--profile-id", required=True)
    verify_pr_head.add_argument("--profile-version", required=True)
    verify_pr_head.add_argument(
        "--mandatory-check",
        action="append",
        default=[],
        help="repeatable check name that must be present with a pass outcome",
    )
    verify_pr_head.add_argument("--current-pr-diff-digest", required=True)

    return parser


def _instance_receipt_lanes() -> Any:
    """``scripts/ci/instance_receipt_lanes.py``, loaded by path (OMN-19543).

    Lazily, and only by ``gate --instance-lanes-for``: that module needs PyYAML,
    and every other path through this file stays stdlib-only, as with
    ``_release_train_module``.
    """
    name = "_lab_pass_instance_receipt_lanes"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "instance_receipt_lanes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load the instance receipt lanes from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def instance_lanes_for(repo: str) -> list[EnumLabLane]:
    """The receipt lanes of the instances that prove ``repo``, as enum values.

    Raises ``ValueError`` for a table lane this enum does not declare, and the
    reader's own error for an unreadable table; ``gate`` refuses on both.
    """
    return [
        EnumLabLane(value)
        for value in _instance_receipt_lanes().receipt_lanes_for(repo)
    ]


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
        if args.consumer_groups_file is not None:
            try:
                sample_groups = load_declared_groups(args.consumer_groups_file)
            except GroupSourceError as exc:
                # A baseline that cannot be taken is not a failure: the growth
                # arm simply has nothing to compare against and the check says
                # so. Failing here would turn a missing baseline into a missing
                # RECEIPT.
                print(f"::warning::lag baseline not taken: {exc}", file=sys.stderr)
                return 0
        else:
            sample_groups = parse_group_list_argument(args.consumer_groups or "")
        readings = sample_group_lag(sample_access, sample_groups)
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

        declared_groups = parse_group_list_argument(args.consumer_groups)
        group_source_error = ""
        if args.consumer_groups_file is not None:
            try:
                declared_groups = load_declared_groups(args.consumer_groups_file)
            except GroupSourceError as exc:
                # NOT a hard exit. The other checks in this receipt are still
                # worth recording, and an unwritten receipt is the one outcome
                # worse than a failing one. The lag check reports the cause.
                group_source_error = str(exc)

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
            health_observe_budget_seconds=args.health_observe_budget_seconds,
            declared_migrations=declared_migrations,
            migration_ledger=migration_ledger,
            broker_access=broker_access,
            declared_consumer_groups=declared_groups,
            consumer_group_source_error=group_source_error,
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
                converged_via=args.converged_via,
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            print(
                f"::error::refusing to emit an invalid receipt: {exc}", file=sys.stderr
            )
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(receipt.to_json(indent=2), encoding="utf-8")
        if args.event_out is not None:
            # Written for a FAIL receipt exactly as for a PASS: the record of a
            # failed lab pass is as valuable as the record of a passing one.
            args.event_out.parent.mkdir(parents=True, exist_ok=True)
            args.event_out.write_text(
                json.dumps(build_bus_event(receipt), indent=2), encoding="utf-8"
            )
            print(f"wrote lab-pass bus event -> {args.event_out}")
        print(f"wrote {artifact_name(receipt.lane, receipt.sha)} -> {args.out}")
        print(render_receipt(receipt))
        # A FAIL receipt is still EMITTED — the record of a failed lab pass is
        # exactly as valuable as the record of a passing one — but the emitting
        # step goes red so the failure is visible on the run that produced it.
        return 0 if receipt.result == EnumLabPassResult.PASS else 1

    if args.command == "verify":
        return verify_emitted(args.path, args.sha, EnumLabLane(args.lane), sys.stdout)

    if args.command == "verify-pr-head":
        try:
            receipt = parse_receipt(args.receipt.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ReceiptLookupError) as exc:
            print(
                json.dumps(
                    {
                        "token": "UNREADABLE",
                        "reason": f"receipt input is unreadable: {exc}",
                    },
                    sort_keys=True,
                )
            )
            return 2
        verdict, reason = verify_pr_head_receipt(
            receipt,
            expected_repo=args.repo,
            expected_pr_number=args.pr,
            expected_head_sha=args.head_sha,
            expected_profile_id=args.profile_id,
            expected_profile_version=args.profile_version,
            mandatory_checks=frozenset(args.mandatory_check),
            current_pr_diff_digest=args.current_pr_diff_digest,
        )
        print(json.dumps({"token": verdict.value, "reason": reason}, sort_keys=True))
        return 0 if verdict is EnumPrHeadVerdict.ACCEPTED else 1

    if args.command == "reemit-eligible":
        # Prints one word, because the workflow branches on it and a decision
        # a shell has to parse out of prose is a decision nothing tests.
        existing = args.existing
        if existing is None or not existing.is_file():
            if args.endpoint:
                if not args.endpoint_subject:
                    print("skip:endpoint-has-no-receipt")
                    return 0
                print(
                    endpoint_reemit_eligibility(
                        args.endpoint_subject,
                        repo=args.repo,
                        workflow=args.delivery_workflow,
                        branch=args.delivery_branch,
                        bound_seconds=args.overall_bound_seconds,
                        now=datetime.now(UTC),
                    )
                )
                return 0
            print("reemit:absent")
            return 0
        try:
            receipt = ModelLabPassReceipt.from_json(
                existing.read_text(encoding="utf-8")
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            # Unreadable is NOT eligible. A receipt nobody can parse might be a
            # real FAIL, and guessing in the permissive direction is the one
            # mistake this feature must not make.
            print(f"skip:unreadable-existing-receipt: {exc}")
            return 0
        if receipt.result is EnumLabPassResult.PASS:
            print("skip:already-passing")
            return 0
        if is_binding_only_failure(receipt):
            print("reemit:binding-failure")
            return 0
        failed = sorted(c.name for c in receipt.checks if not c.ok)
        print(f"skip:health-failure {failed}")
        return 0
    if args.command == "reemit":
        try:
            source = ModelLabPassReceipt.from_json(
                args.source.read_text(encoding="utf-8")
            )
            receipt = reemit_receipt(source, args.sha, args.reemit_converged_via)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            print(f"::error::refusing to re-emit: {exc}", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(receipt.to_json(indent=2), encoding="utf-8")
        print(
            f"re-emitted {receipt.result.value} for {receipt.sha} "
            f"via {receipt.converged_via}"
        )
        return 0
    if args.command == "gate":
        lanes = (
            [EnumLabLane(value) for value in args.lane]
            if args.lane
            else list(ANY_OF_DEFAULT_LANES)
        )
        if args.instance_lanes_for:
            if not args.lane:
                print(
                    f"::error::lab-pass gate FAILED for {args.sha}: "
                    "--instance-lanes-for needs --lane; refusing rather than "
                    "adding instance lanes to the any-of default.",
                    file=sys.stdout,
                )
                return 1
            try:
                extra = instance_lanes_for(args.instance_lanes_for)
            except Exception as exc:  # noqa: BLE001 - an unread table refuses
                print(
                    f"::error::lab-pass gate FAILED for {args.sha}: the instance "
                    f"receipt lanes for {args.instance_lanes_for} could not be read "
                    f"({exc}); refusing rather than reading fewer lanes.",
                    file=sys.stdout,
                )
                return 1
            lanes.extend(lane for lane in extra if lane not in lanes)
        required = list(dict.fromkeys(EnumLabLane(v) for v in args.require_lane))
        required_sha: str | None = None
        required_note = ""
        if args.resolve_runtime_ancestor:
            if args.clone is None or args.runtime_path_validator is None:
                print(
                    f"::error::lab-pass gate FAILED for {args.sha}: "
                    "--resolve-runtime-ancestor needs --clone and "
                    "--runtime-path-validator; refusing rather than guessing.",
                    file=sys.stdout,
                )
                return 1
            required_sha, required_note = resolve_required_subject_from_clone(
                args.sha, args.clone, args.runtime_path_validator, args.repo
            )
        first_read_at: datetime | None = None
        if args.run_attempt is not None and args.run_attempt > 1:
            if args.run_id is None or not args.gate_job_name:
                print(
                    f"::error::lab-pass gate FAILED for {args.sha}: --run-attempt "
                    f"{args.run_attempt} needs --run-id and --gate-job-name to find "
                    "the first gate read. token=UNREADABLE",
                    file=sys.stdout,
                )
                return 1
            try:
                first_read_at = resolve_first_gate_read(
                    args.repo,
                    args.run_id,
                    args.run_attempt,
                    args.gate_job_name,
                    now=lambda: datetime.now(UTC),
                )
            except ReceiptLookupError as exc:
                print(
                    f"::error::lab-pass gate FAILED for {args.sha}: this delivery "
                    f"run's first gate read could not be read ({exc}), so the "
                    "overall bound cannot be applied. token=UNREADABLE",
                    file=sys.stdout,
                )
                if args.verdict_out is not None:
                    # No lane named: evaluate_gate's own refusal branch records
                    # the verdict as UNREADABLE, in the one verdict shape.
                    evaluate_gate(
                        args.repo,
                        args.sha,
                        [],
                        io.StringIO(),
                        required_sha=required_sha,
                        verdict_out=args.verdict_out,
                        run_id=args.run_id,
                        run_attempt=args.run_attempt,
                    )
                return 1
        rebuild_pending: Callable[[str], bool] | None = None
        if args.detect_pending:
            repo_name = args.repo.split("/")[-1]

            def rebuild_pending(subject: str) -> bool:
                train = _load_release_train()
                return bool(train.default_rebuild_pending(repo_name, subject))

        return evaluate_gate(
            args.repo,
            args.sha,
            lanes,
            sys.stdout,
            required=required,
            required_sha=required_sha,
            required_note=required_note,
            wait_seconds=args.wait_seconds,
            poll_seconds=args.poll_seconds,
            rebuild_pending=rebuild_pending,
            first_read_at=first_read_at,
            overall_bound_seconds=args.overall_bound_seconds,
            verdict_out=args.verdict_out,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
        )
    if args.command == "rerun-refused-deliveries":
        return rerun_refused_deliveries(
            args.repo,
            args.subject,
            workflow=args.workflow,
            branch=args.branch,
            bound_seconds=args.overall_bound_seconds,
            out=sys.stdout,
            now=lambda: datetime.now(UTC),
        )
    if args.command == "workflow-verdict":
        return evaluate_workflow_verdict(
            args.repo,
            args.workflow,
            args.branch,
            args.max_age_hours,
            args.event,
            sys.stdout,
            dispatch_title_contains=args.dispatch_title_contains,
            drill_red_until=args.drill_red_until,
            drill_ticket=args.drill_ticket,
        )

    raise AssertionError(f"unreachable subcommand {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
