# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed gate for the tenant_scoped_ingress opt-in (OMN-14360 / OMN-14208).

``event_bus.tenant_scoped_ingress: true`` opts a contract into the OMN-14349
(OMN-14208 Path A) tenant-stamp: auto-wiring derives ``tenant_id`` from the
matched ``tenant-<slug>.`` topic prefix and overwrites ``payload["tenant_id"]``
before dispatch. That derivation is only sound when a subscribe topic actually
carries a ``tenant-<slug>.`` wire prefix. A bare (un-prefixed) subscribe topic
is left unstamped, so a client-supplied ``tenant_id`` survives into the payload
unverified — a silent cross-tenant identity leak.

This gate is the mechanical enforcement companion (CLAUDE.md Rule #5) to the
runtime stamp in
``omnibase_infra.runtime.auto_wiring.handler_wiring._stamp_tenant_id_from_topic_prefix``
(OMN-14349). It scans node ``contract.yaml`` files and fails when an opted-in
contract violates any invariant below:

condition (c) — topic shape:
    Every ``event_bus.subscribe_topics`` entry MUST match the tenant wire
    prefix ``^tenant-<slug>.`` (same regex the runtime stamp matches) UNLESS it
    is carved out as a *trusted-internal* topic (see the OMN-14482 carve-out
    below). A bare topic that is neither prefixed nor a verified trusted-internal
    topic fails closed — no opt-in may bind a topic the stamp cannot cover.

    An opted-in contract must ALSO bind at least one genuinely
    ``tenant-<slug>.``-prefixed subscribe topic; a flag-true contract whose only
    subscribe topics are trusted-internal bare topics is a silent no-op opt-in
    (the stamp never fires) — the OMN-14208 failure class.

condition (b) — proving-seam attestation:
    Any contract that flips the flag true MUST be named in the config-as-data
    allowlist (``config/validation/tenant_scoped_ingress_allowlist.yaml``) with
    a non-empty ``seam_test`` path. This mechanizes "no opt-in flip until a
    real cross-boundary seam test exists" — the OMN-14208 near-miss was two
    individually-green PRs that were a silent 100% runtime no-op because no
    test drove the actual seam. The allowlist is fail-closed: a missing file,
    or a flag-true contract with no entry, is a violation.

trusted-internal carve-out (OMN-14482) — how a mixed contract opts in safely:
    An opted-in node whose reader also serves a *trusted internal* bare topic
    (e.g. the delegate-skill orchestrator, which consumes an internal command
    topic no external tenant can publish to) may keep that bare topic alongside
    its ``tenant-<slug>.``-prefixed external topics. That is only sound when the
    bare topic is genuinely unreachable by an untrusted producer, so it is
    NOT a self-declared contract flag: the carve-out lives in a
    CODEOWNERS-protected allowlist entry
    (``trusted_internal_topics: [<bare topic>]``) AND the gate mechanically
    verifies that NO contract under the producer-scan root publishes to that
    topic. A topic that any node republishes to (an in-cluster producer exists)
    can carry a client-supplied ``tenant_id`` on that leg and so is REJECTED for
    the carve-out — the mislabel cannot dodge the leak check. The mechanical
    scan is per-repo; the CODEOWNERS approval on the allowlist file is the
    cross-repo backstop (a producer in another repo is caught by review).

Usage (pre-commit, per-file):
    uv run python -m omnibase_infra.validators.tenant_scoped_ingress_schema \\
        src/omnibase_infra/nodes/<node>/contract.yaml

Usage (CI, whole tree):
    uv run python -m omnibase_infra.validators.tenant_scoped_ingress_schema \\
        src/omnibase_infra

Usage (another repo's contracts, e.g. omnimarket, against its own allowlist):
    uv run python -m omnibase_infra.validators.tenant_scoped_ingress_schema \\
        src/omnimarket \\
        --allowlist config/validation/tenant_scoped_ingress_allowlist.yaml \\
        --producer-scan-root src/omnimarket
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")
DEFAULT_ALLOWLIST_PATH = Path("config/validation/tenant_scoped_ingress_allowlist.yaml")

# SYNC with the canonical runtime stamp regex
# ``omnibase_infra.runtime.auto_wiring.handler_wiring._TENANT_WIRE_PREFIX_RE``
# (OMN-14349). The runtime form captures the slug; this gate only needs the
# match, so the capture group is dropped. Both accept ``tenant-<slug>.`` where
# ``<slug>`` is a 3-63 char DNS-compatible lowercase label.
_TENANT_WIRE_PREFIX_RE: re.Pattern[str] = re.compile(
    r"^tenant-[a-z][a-z0-9-]{1,61}[a-z0-9]\."
)


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal entry
class AllowlistEntry:
    """A single seam-test attestation allowlist entry.

    ``seam_test`` is the proving cross-boundary test path (condition b).
    ``trusted_internal_topics`` is the OMN-14482 per-topic condition-(c)
    carve-out: bare subscribe topics this node is permitted to keep BECAUSE
    they are trusted-internal (no in-cluster producer; CODEOWNERS-approved).
    """

    seam_test: str = ""
    trusted_internal_topics: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class TenantScopedIngressFinding:
    path: Path
    node_id: str
    kind: str
    detail: str

    def format(self) -> str:
        return f"{self.path} [{self.node_id}]: {self.kind} — {self.detail}"


def load_allowlist(path: Path) -> dict[str, AllowlistEntry]:
    """Load the seam-test attestation allowlist as ``{node_id: AllowlistEntry}``.

    Fail-closed: a missing or malformed file yields an EMPTY mapping, so every
    ``tenant_scoped_ingress: true`` contract then fails condition (b) until a
    real allowlist entry exists.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("allowlist")
    if not isinstance(entries, list):
        return {}
    mapping: dict[str, AllowlistEntry] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        node_id = entry.get("node_id")
        if not isinstance(node_id, str):
            continue
        seam_test = entry.get("seam_test")
        trusted_raw = entry.get("trusted_internal_topics")
        trusted = (
            frozenset(t for t in trusted_raw if isinstance(t, str))
            if isinstance(trusted_raw, list)
            else frozenset()
        )
        mapping[node_id] = AllowlistEntry(
            seam_test=seam_test if isinstance(seam_test, str) else "",
            trusted_internal_topics=trusted,
        )
    return mapping


def collect_published_topics(root: Path) -> frozenset[str]:
    """Return every topic any contract under ``root`` declares in publish_topics.

    This is the mechanical "in-cluster producer" registry the trusted-internal
    carve-out checks against: a bare topic that some node republishes to is NOT
    a pure trusted-internal entry point (a client value could ride in on that
    producer's leg), so it is ineligible for the condition-(c) carve-out. The
    scan is intentionally decoupled from the (possibly per-file) validation
    ``paths`` so pre-commit's single-file invocation still sees the full
    producer corpus.
    """
    published: set[str] = set()
    for contract_path in _iter_contract_files((root,)):
        try:
            raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict):
            continue
        event_bus = raw.get("event_bus")
        if not isinstance(event_bus, dict):
            continue
        publish_raw = event_bus.get("publish_topics")
        if isinstance(publish_raw, list):
            published.update(t for t in publish_raw if isinstance(t, str))
    return frozenset(published)


def validate_file(
    path: Path,
    allowlist: Mapping[str, AllowlistEntry],
    published_topics: frozenset[str],
) -> list[TenantScopedIngressFinding]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        # Malformed / unreadable YAML is another gate's concern; do not crash here.
        return []

    if not isinstance(raw, dict):
        return []

    event_bus = raw.get("event_bus")
    if not isinstance(event_bus, dict):
        return []

    # Only an explicit boolean True opts in; a truthy string/int does not.
    if event_bus.get("tenant_scoped_ingress") is not True:
        return []

    node_id = str(raw.get("name") or path.parent.name)
    entry = allowlist.get(node_id)
    trusted_internal = (
        entry.trusted_internal_topics if entry is not None else frozenset()
    )
    findings: list[TenantScopedIngressFinding] = []

    subscribe_raw = event_bus.get("subscribe_topics")
    topics = subscribe_raw if isinstance(subscribe_raw, list) else []

    # (c) topic shape: an opted-in contract with nothing to stamp is a silent
    # no-op opt-in (the OMN-14208 failure class); require at least one topic.
    if not topics:
        findings.append(
            TenantScopedIngressFinding(
                path=path,
                node_id=node_id,
                kind="NO_SUBSCRIBE_TOPICS",
                detail=(
                    "sets event_bus.tenant_scoped_ingress: true but declares no "
                    "subscribe_topics — the tenant stamp never fires (a silent "
                    "no-op opt-in)."
                ),
            )
        )

    # (c) topic shape: every subscribe topic must carry the tenant-<slug>. prefix
    # OR be a verified trusted-internal bare topic (OMN-14482).
    prefixed_count = 0
    for topic in topics:
        if isinstance(topic, str) and _TENANT_WIRE_PREFIX_RE.match(topic) is not None:
            prefixed_count += 1
            continue
        # Bare (or non-string) topic. Allowed only via the trusted-internal carve-out.
        if isinstance(topic, str) and topic in trusted_internal:
            # (OMN-14482) the carve-out is only sound when NO in-cluster producer
            # republishes to this topic — otherwise a client tenant_id can ride
            # in on that producer's leg. Fail closed if a producer exists.
            if topic in published_topics:
                findings.append(
                    TenantScopedIngressFinding(
                        path=path,
                        node_id=node_id,
                        kind="TRUSTED_INTERNAL_HAS_PRODUCER",
                        detail=(
                            f"subscribe topic {topic!r} is allowlisted as "
                            "trusted_internal, but a contract under the "
                            "producer-scan root PUBLISHES to it — so it is not a "
                            "pure trusted-internal entry point (a client-supplied "
                            "tenant_id could survive on the producer's leg). The "
                            "trusted-internal carve-out is REJECTED (OMN-14482)."
                        ),
                    )
                )
            # else: verified trusted-internal — carve-out granted, no finding.
            continue
        findings.append(
            TenantScopedIngressFinding(
                path=path,
                node_id=node_id,
                kind="BARE_OR_MIXED_TOPIC",
                detail=(
                    f"subscribe topic {topic!r} is not tenant-<slug>.-prefixed and "
                    "is not an allowlisted trusted_internal topic. A bare/mixed "
                    "topic is left unstamped, so a client-supplied tenant_id "
                    "survives into the payload unverified (cross-tenant identity "
                    "leak, OMN-14208). Prefix the topic, or — if it is a trusted "
                    "internal topic no untrusted producer can reach — add it to "
                    "this node's trusted_internal_topics allowlist entry "
                    "(OMN-14482)."
                ),
            )
        )

    # (c) anti-no-op: an opted-in contract must actually stamp SOMETHING. A
    # flag-true contract whose only subscribe topics are trusted-internal bare
    # topics never fires the stamp — the OMN-14208 silent-no-op class.
    if topics and prefixed_count == 0:
        findings.append(
            TenantScopedIngressFinding(
                path=path,
                node_id=node_id,
                kind="NO_PREFIXED_TOPIC",
                detail=(
                    "sets event_bus.tenant_scoped_ingress: true but binds no "
                    "tenant-<slug>.-prefixed subscribe topic — the stamp never "
                    "fires (a silent no-op opt-in). At least one prefixed topic "
                    "is required (OMN-14482)."
                ),
            )
        )

    # trusted_internal_topics hygiene: every carved-out topic must actually be a
    # bare subscribe topic of THIS contract. A carve-out for a topic the node
    # does not subscribe to, or for an already-prefixed topic, is a dead/wrong
    # designation that should be corrected rather than silently accepted.
    subscribe_set = {t for t in topics if isinstance(t, str)}
    for topic in sorted(trusted_internal):
        if topic not in subscribe_set:
            findings.append(
                TenantScopedIngressFinding(
                    path=path,
                    node_id=node_id,
                    kind="TRUSTED_INTERNAL_NOT_SUBSCRIBED",
                    detail=(
                        f"trusted_internal_topics names {topic!r}, which this "
                        "contract does not subscribe to. Remove the stale "
                        "designation (OMN-14482)."
                    ),
                )
            )
        elif _TENANT_WIRE_PREFIX_RE.match(topic) is not None:
            findings.append(
                TenantScopedIngressFinding(
                    path=path,
                    node_id=node_id,
                    kind="TRUSTED_INTERNAL_IS_PREFIXED",
                    detail=(
                        f"trusted_internal_topics names {topic!r}, which is already "
                        "tenant-<slug>.-prefixed and thus stamped — it needs no "
                        "carve-out. Remove it from trusted_internal_topics "
                        "(OMN-14482)."
                    ),
                )
            )

    # (b) proving-seam attestation: the flag-true contract must name a seam test.
    if entry is None or not entry.seam_test:
        findings.append(
            TenantScopedIngressFinding(
                path=path,
                node_id=node_id,
                kind="MISSING_SEAM_TEST",
                detail=(
                    "sets event_bus.tenant_scoped_ingress: true but names no proving "
                    "seam test in the tenant_scoped_ingress allowlist. "
                    "Opt-in is forbidden until a real cross-boundary seam test drives "
                    "the stamp→consumer boundary (OMN-14208 / OMN-14360)."
                ),
            )
        )

    return findings


def validate_paths(
    paths: Sequence[Path],
    allowlist: Mapping[str, AllowlistEntry],
    published_topics: frozenset[str],
) -> list[TenantScopedIngressFinding]:
    findings: list[TenantScopedIngressFinding] = []
    for path in _iter_contract_files(paths):
        findings.extend(validate_file(path, allowlist, published_topics))
    return findings


def _iter_contract_files(paths: Sequence[Path]) -> Iterator[Path]:
    scan_paths = paths or (DEFAULT_SCAN_ROOT,)
    for path in scan_paths:
        if path.is_file():
            if path.name == "contract.yaml" or path.name.endswith(".contract.yaml"):
                yield path
        elif path.is_dir():
            yield from sorted(
                p for p in path.rglob("contract.yaml") if "__pycache__" not in p.parts
            )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed gate: reject event_bus.tenant_scoped_ingress: true "
            "contracts with bare/mixed subscribe topics or no proving seam test "
            "(OMN-14360 / OMN-14208 / OMN-14482)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_SCAN_ROOT],
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST_PATH,
        help=(
            "Path to the seam-test attestation allowlist "
            "(default: config/validation/tenant_scoped_ingress_allowlist.yaml)."
        ),
    )
    parser.add_argument(
        "--producer-scan-root",
        type=Path,
        default=None,
        help=(
            "Root scanned to build the in-cluster producer index for the "
            "trusted-internal carve-out check (OMN-14482). Defaults to the first "
            "positional directory path, else src/omnibase_infra. Set this "
            "explicitly in per-file (pre-commit) mode so the producer corpus is "
            "still complete."
        ),
    )
    return parser.parse_args(argv)


def _resolve_producer_scan_root(args: argparse.Namespace) -> Path:
    explicit: Path | None = args.producer_scan_root
    if explicit is not None:
        return explicit
    paths: Sequence[Path] = args.paths
    for path in paths:
        if path.is_dir():
            return path
    return DEFAULT_SCAN_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else list(argv))
    allowlist = load_allowlist(args.allowlist)
    published_topics = collect_published_topics(_resolve_producer_scan_root(args))
    findings = validate_paths(args.paths, allowlist, published_topics)

    if not findings:
        return 0

    for finding in findings:
        sys.stderr.write(f"  {finding.format()}\n")

    sys.stderr.write(
        f"\n[tenant-scoped-ingress-gate] {len(findings)} tenant_scoped_ingress "
        "violation(s) found.\n"
        "A contract that sets event_bus.tenant_scoped_ingress: true must:\n"
        "  1. bind at least one tenant-<slug>.-prefixed subscribe_topic, and every "
        "other subscribe_topic must be prefixed OR an allowlisted, producer-free "
        "trusted_internal topic (OMN-14482), and\n"
        "  2. be named in the tenant_scoped_ingress allowlist with a seam_test "
        "path proving the stamp→consumer boundary.\n"
        "An unprefixed, un-carved-out topic lets a client-supplied tenant_id "
        "survive unverified; an un-attested opt-in can be a silent runtime no-op "
        "(OMN-14208).\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
