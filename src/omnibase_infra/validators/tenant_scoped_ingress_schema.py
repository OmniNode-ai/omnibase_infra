# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed gate for the tenant_scoped_ingress opt-in (OMN-14360 / OMN-14208).

``event_bus.tenant_scoped_ingress: true`` opts a contract into the OMN-14349
(OMN-14208 Path A) tenant-stamp: auto-wiring derives ``tenant_id`` from the
matched ``tenant-<slug>.`` topic prefix and overwrites ``payload["tenant_id"]``
before dispatch. That derivation is only sound when EVERY subscribe topic the
contract binds actually carries a ``tenant-<slug>.`` wire prefix. A bare or
mixed subscribe topic (no prefix) is left unstamped, so a client-supplied
``tenant_id`` survives into the payload unverified — a silent cross-tenant
identity leak.

This gate is the mechanical enforcement companion (CLAUDE.md Rule #5) to the
runtime stamp in
``omnibase_infra.runtime.auto_wiring.handler_wiring._stamp_tenant_id_from_topic_prefix``
(OMN-14349). It scans node ``contract.yaml`` files and fails when an opted-in
contract violates either invariant:

condition (c) — topic shape:
    Every ``event_bus.subscribe_topics`` entry MUST match the tenant wire
    prefix ``^tenant-<slug>.`` (same regex the runtime stamp matches). A bare
    or mixed topic fails closed — no opt-in may bind a topic the stamp cannot
    cover.

condition (b) — proving-seam attestation:
    Any contract that flips the flag true MUST be named in the config-as-data
    allowlist (``config/validation/tenant_scoped_ingress_allowlist.yaml``) with
    a non-empty ``seam_test`` path. This mechanizes "no opt-in flip until a
    real cross-boundary seam test exists" — the OMN-14208 near-miss was two
    individually-green PRs that were a silent 100% runtime no-op because no
    test drove the actual seam. The allowlist is fail-closed: a missing file,
    or a flag-true contract with no entry, is a violation.

No contract sets the flag today, so this is a forward-looking gate: it costs
nothing for every existing contract and blocks the first unsafe opt-in.

Usage (pre-commit, per-file):
    uv run python -m omnibase_infra.validators.tenant_scoped_ingress_schema \\
        src/omnibase_infra/nodes/<node>/contract.yaml

Usage (CI, whole tree):
    uv run python -m omnibase_infra.validators.tenant_scoped_ingress_schema \\
        src/omnibase_infra
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class TenantScopedIngressFinding:
    path: Path
    node_id: str
    kind: str
    detail: str

    def format(self) -> str:
        return f"{self.path} [{self.node_id}]: {self.kind} — {self.detail}"


def load_allowlist(path: Path) -> dict[str, str]:
    """Load the seam-test attestation allowlist as ``{node_id: seam_test}``.

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
    mapping: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        node_id = entry.get("node_id")
        seam_test = entry.get("seam_test")
        if not isinstance(node_id, str):
            continue
        mapping[node_id] = seam_test if isinstance(seam_test, str) else ""
    return mapping


def validate_file(
    path: Path, allowlist: Mapping[str, str]
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

    # (c) topic shape: every subscribe topic must carry the tenant-<slug>. prefix.
    for topic in topics:
        if not isinstance(topic, str) or _TENANT_WIRE_PREFIX_RE.match(topic) is None:
            findings.append(
                TenantScopedIngressFinding(
                    path=path,
                    node_id=node_id,
                    kind="BARE_OR_MIXED_TOPIC",
                    detail=(
                        f"subscribe topic {topic!r} is not tenant-<slug>.-prefixed. "
                        "A bare/mixed topic is left unstamped, so a client-supplied "
                        "tenant_id survives into the payload unverified "
                        "(cross-tenant identity leak, OMN-14208)."
                    ),
                )
            )

    # (b) proving-seam attestation: the flag-true contract must name a seam test.
    if not allowlist.get(node_id):
        findings.append(
            TenantScopedIngressFinding(
                path=path,
                node_id=node_id,
                kind="MISSING_SEAM_TEST",
                detail=(
                    "sets event_bus.tenant_scoped_ingress: true but names no proving "
                    "seam test in config/validation/tenant_scoped_ingress_allowlist.yaml. "
                    "Opt-in is forbidden until a real cross-boundary seam test drives "
                    "the stamp→consumer boundary (OMN-14208 / OMN-14360)."
                ),
            )
        )

    return findings


def validate_paths(
    paths: Sequence[Path], allowlist: Mapping[str, str]
) -> list[TenantScopedIngressFinding]:
    findings: list[TenantScopedIngressFinding] = []
    for path in _iter_contract_files(paths):
        findings.extend(validate_file(path, allowlist))
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
            "(OMN-14360 / OMN-14208)."
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else list(argv))
    allowlist = load_allowlist(args.allowlist)
    findings = validate_paths(args.paths, allowlist)

    if not findings:
        return 0

    for finding in findings:
        sys.stderr.write(f"  {finding.format()}\n")

    sys.stderr.write(
        f"\n[tenant-scoped-ingress-gate] {len(findings)} tenant_scoped_ingress "
        "violation(s) found.\n"
        "A contract that sets event_bus.tenant_scoped_ingress: true must:\n"
        "  1. bind ONLY tenant-<slug>.-prefixed subscribe_topics (so the stamp "
        "covers every topic), and\n"
        "  2. be named in config/validation/tenant_scoped_ingress_allowlist.yaml "
        "with a seam_test path proving the stamp→consumer boundary.\n"
        "An unprefixed topic lets a client-supplied tenant_id survive unverified; "
        "an un-attested opt-in can be a silent runtime no-op (OMN-14208).\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
