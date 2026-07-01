# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Repository scanner + ratchet for ARCH-004 (imperative-orchestrator debt).

This module turns the per-node ARCH-004 analysis
(``validator_contract_declared_orchestrator_workflow.analyze_node_directory``)
into a repository-wide gate with three modes:

    * ``--check-all --report`` — scan every node directory, emit one report row
      per hard-fail. Never hides debt; intended for CI reporting and baseline
      generation. Exit 0 (reporting) unless ``--ratchet`` is also requested.
    * ``--check-changed --ratchet`` — scan only the node directories touched by
      a changeset. Fail (exit 1) if a touched node introduces a NEW hard-fail,
      WORSENS its risk score / grows its max handler beyond the baseline, or is
      an untracked hard-fail with no baseline entry.
    * ``--strict`` — after a remediation wave, treat any baselined node that the
      live scan no longer reproduces as a recurrence failure if it reappears;
      with ``--strict`` a baselined node that still hard-fails is itself a
      failure (the baseline is being dropped). Use to ratchet the baseline down.

Baseline format: ``architecture-handshakes/imperative-orchestrator-baseline.yaml``.
This mirrors the repo's existing handshake-baseline pattern
(``validator-requirements-baseline.yaml``): the baseline can only SHRINK; a new
or worsened finding fails the gate.

Related:
    - OMN-13472 (ARCH-004 ratchet — Workstream B)
    - OMN-13471 (delegation decomposition epic — baseline owner ticket)
    - OMN-12550 / OMN-13325 (validator wiring / ratchet-enforcement epic)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from omnibase_infra.nodes.node_architecture_validator.validators.validator_contract_declared_orchestrator_workflow import (
    analyze_node_directory,
)

#: Canonical baseline location relative to a repo root.
BASELINE_RELATIVE_PATH = "architecture-handshakes/imperative-orchestrator-baseline.yaml"

#: Owner ticket recorded against the delegation hard-fail in the baseline
#: (the dedicated decomposition epic).
DELEGATION_OWNER_TICKET = "OMN-13471"

#: Owner ticket for every other current hard-fail (same anti-pattern class,
#: tracked under the imperative-orchestrator decomposition epic). The audit
#: (docs/audits/2026-06-22-imperative-orchestrator-audit.md) treats these as
#: baseline debt under the same epic, with delegation as the sole P0.
FLEET_OWNER_TICKET = "OMN-13471"


@dataclass(frozen=True)  # internal-dataclass-ok: scanner-internal baseline YAML row
class BaselineEntry:
    """One accepted hard-fail in the ratchet baseline."""

    repo: str
    node: str
    max_handler_path: str
    line_count: int
    risk_score: int
    finding_codes: tuple[str, ...]
    owner_ticket: str
    #: Optional one-line justification for keeping this hard-fail in the
    #: baseline (e.g. a thin handler that delegates the state decision to a
    #: pure-function reducer and is accepted as baseline rather than Class A).
    #: Empty string when no rationale is recorded.
    accepted_rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "repo": self.repo,
            "node": self.node,
            "max_handler_path": self.max_handler_path,
            "line_count": self.line_count,
            "risk_score": self.risk_score,
            "finding_codes": list(self.finding_codes),
            "owner_ticket": self.owner_ticket,
        }
        if self.accepted_rationale:
            out["accepted_rationale"] = self.accepted_rationale
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BaselineEntry:
        def _as_int(value: object) -> int:
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return 0

        codes_raw = data.get("finding_codes", [])
        codes: tuple[str, ...] = (
            tuple(str(c) for c in codes_raw) if isinstance(codes_raw, list) else ()
        )
        return cls(
            repo=str(data.get("repo", "")),
            node=str(data["node"]),
            max_handler_path=str(data.get("max_handler_path", "")),
            line_count=_as_int(data.get("line_count", 0)),
            risk_score=_as_int(data.get("risk_score", 0)),
            finding_codes=codes,
            owner_ticket=str(data.get("owner_ticket", "")),
            accepted_rationale=str(data.get("accepted_rationale", "")),
        )


@dataclass  # internal-dataclass-ok: scanner-internal aggregate scan result
class ScanResult:
    """Aggregated scan result across a set of node directories."""

    repo: str
    hard_fails: list[BaselineEntry] = field(default_factory=list)
    #: Per-node finding detail for reporting (node -> messages).
    report_lines: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_node_dirs(repo_root: Path) -> list[Path]:
    """Return every ``nodes/<node>/`` directory under ``repo_root`` with a contract."""
    src = repo_root / "src"
    search_root = src if src.is_dir() else repo_root
    node_dirs: set[Path] = set()
    for contract in search_root.rglob("contract.yaml"):
        parent = contract.parent
        # A node directory's parent is conventionally ``nodes/``.
        if parent.parent.name == "nodes" or "nodes" in parent.parts:
            node_dirs.add(parent)
    return sorted(node_dirs)


def node_dirs_for_changed_files(
    repo_root: Path, changed_files: list[str]
) -> list[Path]:
    """Map changed file paths to the node directories that contain them."""
    node_dirs: set[Path] = set()
    for raw in changed_files:
        if not raw:
            continue
        candidate = (repo_root / raw).resolve()
        # Walk up until we find a directory holding a contract.yaml.
        cur = candidate if candidate.is_dir() else candidate.parent
        while True:
            if (cur / "contract.yaml").is_file() and "nodes" in cur.parts:
                node_dirs.add(cur)
                break
            if cur == repo_root or cur.parent == cur:
                break
            cur = cur.parent
    return sorted(node_dirs)


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def _relative_handler_path(handler_path: Path | None, repo_root: Path | None) -> str:
    """Render a handler path repo-relative (never a machine-absolute path).

    Absolute ``/Users/...`` / ``/Volumes/...`` paths in committed artifacts are a
    portability bug (Operating Rule #6). When ``repo_root`` is known and the
    handler lives under it, the stored path is relative to the repo root.
    """
    if handler_path is None:
        return ""
    if repo_root is not None:
        try:
            return str(handler_path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            pass
    # Fall back to the path from the ``src/`` segment onward, never absolute.
    parts = handler_path.parts
    if "src" in parts:
        idx = parts.index("src")
        return str(Path(*parts[idx:]))
    return handler_path.name


def scan_node_dirs(
    repo: str, node_dirs: list[Path], repo_root: Path | None = None
) -> ScanResult:
    """Run ARCH-004 over ``node_dirs`` and collect hard-fails + report lines.

    Args:
        repo: Repo name recorded on each baseline entry.
        node_dirs: Node directories to analyze.
        repo_root: Repo root used to render ``max_handler_path`` repo-relative
            (avoids committing machine-absolute paths).
    """
    result = ScanResult(repo=repo)
    for node_dir in node_dirs:
        analysis = analyze_node_directory(node_dir)
        if analysis is None:
            continue
        if not analysis.has_hard_fail:
            continue
        entry = BaselineEntry(
            repo=repo,
            node=analysis.node_name,
            max_handler_path=_relative_handler_path(
                analysis.max_handler_path, repo_root
            ),
            line_count=analysis.max_handler_lines,
            risk_score=analysis.risk_score,
            finding_codes=tuple(analysis.finding_codes),
            owner_ticket=(
                DELEGATION_OWNER_TICKET
                if "delegation" in analysis.node_name
                else FLEET_OWNER_TICKET
            ),
        )
        result.hard_fails.append(entry)
        for v in analysis.violations():
            result.report_lines.append(
                f"  [{analysis.node_name}] {v.severity.value.upper()}: {v.message}"
            )
    return result


# --------------------------------------------------------------------------- #
# Baseline I/O
# --------------------------------------------------------------------------- #
def _baseline_key(repo: str, node: str) -> str:
    """Composite baseline key (``repo::node``) so cross-repo nodes never collide."""
    return f"{repo}::{node}"


def load_baseline(baseline_path: Path) -> dict[str, BaselineEntry]:
    """Load the baseline keyed by ``repo::node``. Missing file => empty baseline."""
    if not baseline_path.is_file():
        return {}
    raw = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("entries", [])
    out: dict[str, BaselineEntry] = {}
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict) and "node" in item:
                entry = BaselineEntry.from_dict(item)
                out[_baseline_key(entry.repo, entry.node)] = entry
    return out


def render_baseline_yaml(repo: str, entries: list[BaselineEntry]) -> str:
    """Render a baseline YAML document from scan entries."""
    doc = {
        "schema_version": {"major": 1, "minor": 0, "patch": 0},
        "repo": repo,
        "rule": "ARCH-004",
        "description": (
            "Accepted imperative-orchestrator hard-fails for the ARCH-004 "
            "ratchet (OMN-13472). One entry per current hard-fail. This list can "
            "only SHRINK: a new/worsened/untracked finding on a touched node "
            "fails the changed-node ratchet. Each entry carries an owner ticket "
            "for its decomposition (delegation -> OMN-13471)."
        ),
        "entries": [e.to_dict() for e in sorted(entries, key=lambda e: e.node)],
    }
    header = (
        "# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.\n"
        "# SPDX-License-Identifier: MIT\n"
        "#\n"
        "# ARCH-004 imperative-orchestrator ratchet baseline (OMN-13472).\n"
        "# Generated by:\n"
        "#   uv run python -m omnibase_infra.nodes.node_architecture_validator."
        "validators.scanner_imperative_orchestrator_ratchet \\\n"
        "#       --check-all --report --write-baseline\n"
        "# Owner epic: OMN-13471. Wiring: OMN-12550 / OMN-13325.\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def write_baseline(
    baseline_path: Path, repo: str, entries: list[BaselineEntry]
) -> None:
    """Write the baseline YAML to ``baseline_path`` (creating the dir if needed)."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(render_baseline_yaml(repo, entries), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Ratchet comparison
# --------------------------------------------------------------------------- #
def ratchet_violations(
    scanned: list[BaselineEntry], baseline: dict[str, BaselineEntry]
) -> list[str]:
    """Return ratchet failures for the scanned (changed) hard-fails.

    A scanned hard-fail fails the ratchet when it:
      * is NOT in the baseline (new / untracked hard-fail), OR
      * worsens its risk score vs the baseline, OR
      * materially grows its max handler line count vs the baseline, OR
      * adds a finding code not present in the baseline entry, OR
      * has a baseline entry with no owner ticket.
    """
    failures: list[str] = []
    for entry in scanned:
        base = baseline.get(_baseline_key(entry.repo, entry.node))
        if base is None:
            failures.append(
                f"{entry.node}: NEW imperative-orchestrator hard-fail "
                f"(codes={list(entry.finding_codes)}, risk={entry.risk_score}, "
                f"lines={entry.line_count}) — not in baseline. Decompose it or, "
                f"if intentional debt, add a baseline entry with an owner ticket."
            )
            continue
        if entry.risk_score > base.risk_score:
            failures.append(
                f"{entry.node}: risk score WORSENED "
                f"{base.risk_score} -> {entry.risk_score}."
            )
        if entry.line_count > base.line_count:
            failures.append(
                f"{entry.node}: max handler GREW "
                f"{base.line_count} -> {entry.line_count} lines."
            )
        new_codes = set(entry.finding_codes) - set(base.finding_codes)
        if new_codes:
            failures.append(
                f"{entry.node}: NEW finding codes {sorted(new_codes)} "
                f"(baseline had {list(base.finding_codes)})."
            )
        if not base.owner_ticket:
            failures.append(
                f"{entry.node}: baseline entry has no owner_ticket; "
                f"every accepted hard-fail must cite a decomposition ticket."
            )
    return failures


def strict_violations(
    scanned: list[BaselineEntry], baseline: dict[str, BaselineEntry]
) -> list[str]:
    """Return strict-mode failures: a baselined node that still hard-fails.

    In ``--strict`` mode the baseline is being dropped: any node still on the
    baseline that the live scan reproduces is a failure (it must be remediated
    before the baseline entry is removed).
    """
    scanned_keys = {_baseline_key(e.repo, e.node) for e in scanned}
    return [
        f"{entry.node}: still hard-fails ARCH-004 but is baselined; "
        f"--strict requires the baseline be dropped (decompose it). "
        f"Owner: {entry.owner_ticket or 'UNASSIGNED'}."
        for key, entry in baseline.items()
        if key in scanned_keys
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    # Default: repo root = four parents up from this file
    # (.../src/omnibase_infra/nodes/node_architecture_validator/validators/<file>)
    return Path(__file__).resolve().parents[5]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the ARCH-004 ratchet."""
    parser = argparse.ArgumentParser(
        prog="imperative-orchestrator-ratchet",
        description=(
            "ARCH-004 imperative-orchestrator ratchet (OMN-13472). Scans node "
            "directories for contract-declared-but-unbound orchestrator FSMs "
            "driven imperatively by a handler."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-all",
        action="store_true",
        help="Scan every node directory in the repo.",
    )
    mode.add_argument(
        "--check-changed",
        action="store_true",
        help="Scan only node directories containing the given --files.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print one report row per hard-fail (does not block on its own).",
    )
    parser.add_argument(
        "--ratchet",
        action="store_true",
        help="Fail on a new/worsened/untracked finding for a touched node.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Drop-baseline mode: a baselined node that still hard-fails fails.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write/refresh the baseline YAML from a --check-all scan.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (defaults to this repo).",
    )
    parser.add_argument(
        "--repo-name",
        default="omnibase_infra",
        help="Repo name recorded in baseline entries.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Changed files (relative to repo root) for --check-changed.",
    )
    args = parser.parse_args(argv)

    repo_root = _resolve_repo_root(args.repo_root)
    baseline_path = repo_root / BASELINE_RELATIVE_PATH
    baseline = load_baseline(baseline_path)

    if args.check_all:
        node_dirs = discover_node_dirs(repo_root)
    else:
        node_dirs = node_dirs_for_changed_files(repo_root, args.files)

    result = scan_node_dirs(args.repo_name, node_dirs, repo_root=repo_root)

    if args.report or args.check_all:
        print(
            f"ARCH-004 imperative-orchestrator scan ({args.repo_name}): "
            f"{len(result.hard_fails)} hard-fail node(s) across "
            f"{len(node_dirs)} node dir(s)."
        )
        for line in result.report_lines:
            print(line)

    if args.write_baseline:
        # A baseline must capture the FULL current debt; writing one from a
        # --check-changed scan would silently drop every hard-fail not in the
        # changeset, shrinking the ratchet baseline below true state.
        if not args.check_all:
            print(
                "--write-baseline requires --check-all (a baseline must record "
                "the full current debt, not just the changed nodes).",
                file=sys.stderr,
            )
            return 1
        write_baseline(baseline_path, args.repo_name, result.hard_fails)
        print(f"Wrote baseline: {baseline_path}")
        return 0

    failures: list[str] = []
    if args.strict:
        failures.extend(strict_violations(result.hard_fails, baseline))
    if args.ratchet:
        failures.extend(ratchet_violations(result.hard_fails, baseline))

    if failures:
        print("\nARCH-004 ratchet FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    return 0


__all__ = [
    "BaselineEntry",
    "ScanResult",
    "discover_node_dirs",
    "node_dirs_for_changed_files",
    "scan_node_dirs",
    "load_baseline",
    "render_baseline_yaml",
    "write_baseline",
    "ratchet_violations",
    "strict_violations",
    "main",
    "BASELINE_RELATIVE_PATH",
    "DELEGATION_OWNER_TICKET",
]


if __name__ == "__main__":
    raise SystemExit(main())
