# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Repository scanner + ratchet for ARCH-004 Signal B (orchestration-monolith debt).

Signal B is the orchestration-monolith complexity rule (OMN-13486), DISTINCT
from Signal A (the imperative ``_transition`` / decorative-FSM triad, OMN-13472).
This scanner turns the per-node Signal B analysis
(:func:`validator_orchestration_monolith.analyze_monolith`) into a
repository-wide gate with the same three modes as the Signal A ratchet, but with
its OWN baseline dimension:

    ``architecture-handshakes/orchestration-monolith-baseline.yaml``

The two baselines are deliberately separate files so the two decomposition
tracks (corruption-risk vs monolith complexity) never commingle finding codes or
``accepted_rationale`` semantics.

Modes:
    * ``--check-all --report`` — scan every node directory, one report row per
      hard-fail. Exit 0 (reporting) unless ``--ratchet`` is requested.
    * ``--check-changed --ratchet`` — scan only changed node directories; fail
      (exit 1) on a NEW / worsened / untracked monolith.
    * ``--strict`` — drop-baseline mode: a baselined node that still hard-fails
      is a failure (the baseline must be dropped after decomposition).

Related:
    - OMN-13486 (Signal B — orchestration-monolith complexity)
    - OMN-13485 (sibling decomposition epic — owner tickets)
    - OMN-13472 (Signal A ratchet)
    - OMN-12550 / OMN-13325 (validator wiring / ratchet-enforcement epic)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from omnibase_infra.nodes.node_architecture_validator.validators.scanner_imperative_orchestrator_ratchet import (
    discover_node_dirs,
    node_dirs_for_changed_files,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_orchestration_monolith import (
    MONOLITH_RULE_ID,
    analyze_monolith,
)

#: Canonical Signal B baseline location relative to a repo root. DISTINCT from
#: the Signal A baseline (``imperative-orchestrator-baseline.yaml``).
MONOLITH_BASELINE_RELATIVE_PATH = (
    "architecture-handshakes/orchestration-monolith-baseline.yaml"
)

# ``discover_node_dirs`` and ``node_dirs_for_changed_files`` are re-exported in
# ``__all__`` below so callers (validate.py / tests) can map changed files to
# node dirs and discover all node dirs through the Signal B module surface.


@dataclass(frozen=True)  # internal-dataclass-ok: scanner-internal baseline YAML row
class MonolithBaselineEntry:
    """One accepted orchestration-monolith finding (or assessed not-applicable)."""

    repo: str
    node: str
    max_handler_path: str
    line_count: int
    complexity_score: int
    fan_in: int
    fan_out: int
    finding_codes: tuple[str, ...]
    owner_ticket: str
    #: One-line justification for keeping this monolith in the baseline.
    accepted_rationale: str = ""
    #: One-line justification recorded when a newly-visible node was assessed and
    #: found NOT to be Class B (executor-bound / thin / below threshold). Such an
    #: entry carries empty ``finding_codes`` and documents why it is not gated.
    not_applicable_rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "repo": self.repo,
            "node": self.node,
            "max_handler_path": self.max_handler_path,
            "line_count": self.line_count,
            "complexity_score": self.complexity_score,
            "fan_in": self.fan_in,
            "fan_out": self.fan_out,
            "finding_codes": list(self.finding_codes),
            "owner_ticket": self.owner_ticket,
        }
        if self.accepted_rationale:
            out["accepted_rationale"] = self.accepted_rationale
        if self.not_applicable_rationale:
            out["not_applicable_rationale"] = self.not_applicable_rationale
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MonolithBaselineEntry:
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
            complexity_score=_as_int(data.get("complexity_score", 0)),
            fan_in=_as_int(data.get("fan_in", 0)),
            fan_out=_as_int(data.get("fan_out", 0)),
            finding_codes=codes,
            owner_ticket=str(data.get("owner_ticket", "")),
            accepted_rationale=str(data.get("accepted_rationale", "")),
            not_applicable_rationale=str(data.get("not_applicable_rationale", "")),
        )


@dataclass  # internal-dataclass-ok: scanner-internal aggregate scan result
class MonolithScanResult:
    """Aggregated Signal B scan result across a set of node directories."""

    repo: str
    hard_fails: list[MonolithBaselineEntry] = field(default_factory=list)
    report_lines: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def _relative_handler_path(handler_path: Path | None, repo_root: Path | None) -> str:
    """Render a handler path repo-relative (never machine-absolute; Rule #6)."""
    if handler_path is None:
        return ""
    if repo_root is not None:
        try:
            return str(handler_path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            pass
    parts = handler_path.parts
    if "src" in parts:
        idx = parts.index("src")
        return str(Path(*parts[idx:]))
    return handler_path.name


def scan_node_dirs_for_monolith(
    repo: str, node_dirs: list[Path], repo_root: Path | None = None
) -> MonolithScanResult:
    """Run Signal B over ``node_dirs`` and collect hard-fails + report lines."""
    result = MonolithScanResult(repo=repo)
    for node_dir in node_dirs:
        analysis = analyze_monolith(node_dir)
        if analysis is None:
            continue
        if not analysis.has_hard_fail:
            continue
        entry = MonolithBaselineEntry(
            repo=repo,
            node=analysis.node_name,
            max_handler_path=_relative_handler_path(
                analysis.max_handler_path, repo_root
            ),
            line_count=analysis.max_handler_lines,
            complexity_score=analysis.complexity_score,
            fan_in=analysis.fan_in,
            fan_out=analysis.fan_out,
            finding_codes=tuple(analysis.finding_codes),
            owner_ticket="",
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
    return f"{repo}::{node}"


def load_monolith_baseline(baseline_path: Path) -> dict[str, MonolithBaselineEntry]:
    """Load the Signal B baseline keyed by ``repo::node``."""
    if not baseline_path.is_file():
        return {}
    raw = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("entries", [])
    out: dict[str, MonolithBaselineEntry] = {}
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict) and "node" in item:
                entry = MonolithBaselineEntry.from_dict(item)
                out[_baseline_key(entry.repo, entry.node)] = entry
    return out


def render_monolith_baseline_yaml(
    repo: str, entries: list[MonolithBaselineEntry]
) -> str:
    """Render a Signal B baseline YAML document from scan entries."""
    doc = {
        "schema_version": {"major": 1, "minor": 0, "patch": 0},
        "repo": repo,
        "rule": MONOLITH_RULE_ID,
        "description": (
            "Accepted orchestration-monolith findings for the ARCH-004 Signal B "
            "ratchet (OMN-13486). DISTINCT dimension from the Signal A "
            "imperative-orchestrator baseline. One entry per current monolith "
            "hard-fail (finding code B1) OR a not-applicable assessment for a "
            "node confirmed executor-bound/thin. This list can only SHRINK: a "
            "new/worsened/untracked monolith on a touched node fails the "
            "changed-node ratchet. Each gated entry carries an owner ticket for "
            "its decomposition."
        ),
        "entries": [e.to_dict() for e in sorted(entries, key=lambda e: e.node)],
    }
    header = (
        "# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.\n"
        "# SPDX-License-Identifier: MIT\n"
        "#\n"
        "# ARCH-004 Signal B orchestration-monolith ratchet baseline (OMN-13486).\n"
        "# DISTINCT from imperative-orchestrator-baseline.yaml (Signal A).\n"
        "# Generated by:\n"
        "#   uv run python -m omnibase_infra.nodes.node_architecture_validator."
        "validators.scanner_orchestration_monolith_ratchet \\\n"
        "#       --check-all --report --write-baseline\n"
        "# Owner epic: OMN-13485. Wiring: OMN-12550 / OMN-13325.\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def write_monolith_baseline(
    baseline_path: Path, repo: str, entries: list[MonolithBaselineEntry]
) -> None:
    """Write the Signal B baseline YAML (creating the dir if needed)."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        render_monolith_baseline_yaml(repo, entries), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Ratchet comparison
# --------------------------------------------------------------------------- #
def monolith_ratchet_violations(
    scanned: list[MonolithBaselineEntry],
    baseline: dict[str, MonolithBaselineEntry],
) -> list[str]:
    """Return ratchet failures for the scanned (changed) monolith hard-fails."""
    failures: list[str] = []
    for entry in scanned:
        base = baseline.get(_baseline_key(entry.repo, entry.node))
        if base is None:
            failures.append(
                f"{entry.node}: NEW orchestration-monolith hard-fail "
                f"(codes={list(entry.finding_codes)}, "
                f"complexity={entry.complexity_score}, lines={entry.line_count}, "
                f"fan_in={entry.fan_in}, fan_out={entry.fan_out}) — not in "
                f"Signal B baseline. Decompose it or, if intentional debt, add a "
                f"baseline entry with an owner ticket."
            )
            continue
        if entry.complexity_score > base.complexity_score:
            failures.append(
                f"{entry.node}: complexity score WORSENED "
                f"{base.complexity_score} -> {entry.complexity_score}."
            )
        if entry.line_count > base.line_count:
            failures.append(
                f"{entry.node}: max handler GREW "
                f"{base.line_count} -> {entry.line_count} lines."
            )
        if entry.fan_out > base.fan_out:
            failures.append(
                f"{entry.node}: routing fan_out GREW {base.fan_out} -> {entry.fan_out}."
            )
        new_codes = set(entry.finding_codes) - set(base.finding_codes)
        if new_codes:
            failures.append(
                f"{entry.node}: NEW finding codes {sorted(new_codes)} "
                f"(baseline had {list(base.finding_codes)})."
            )
        if not base.owner_ticket:
            failures.append(
                f"{entry.node}: baseline entry has no owner_ticket; every "
                f"accepted monolith must cite a decomposition ticket."
            )
    return failures


def monolith_strict_violations(
    scanned: list[MonolithBaselineEntry],
    baseline: dict[str, MonolithBaselineEntry],
) -> list[str]:
    """Return strict-mode failures: a baselined node that still hard-fails."""
    scanned_keys = {_baseline_key(e.repo, e.node) for e in scanned}
    return [
        f"{entry.node}: still hard-fails Signal B but is baselined; --strict "
        f"requires the baseline be dropped (decompose it). "
        f"Owner: {entry.owner_ticket or 'UNASSIGNED'}."
        for key, entry in baseline.items()
        if key in scanned_keys and entry.finding_codes
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    return Path(__file__).resolve().parents[5]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the ARCH-004 Signal B ratchet."""
    parser = argparse.ArgumentParser(
        prog="orchestration-monolith-ratchet",
        description=(
            "ARCH-004 Signal B orchestration-monolith ratchet (OMN-13486). Scans "
            "node directories for Class-B orchestration monoliths (oversized + "
            "complex + not executor-bound), independent of FSM decorativeness."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-all", action="store_true")
    mode.add_argument("--check-changed", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--ratchet", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--repo-name", default="omnibase_infra")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)

    repo_root = _resolve_repo_root(args.repo_root)
    baseline_path = repo_root / MONOLITH_BASELINE_RELATIVE_PATH
    baseline = load_monolith_baseline(baseline_path)

    if args.check_all:
        node_dirs = discover_node_dirs(repo_root)
    else:
        node_dirs = node_dirs_for_changed_files(repo_root, args.files)

    result = scan_node_dirs_for_monolith(args.repo_name, node_dirs, repo_root=repo_root)

    if args.report or args.check_all:
        print(
            f"ARCH-004 Signal B orchestration-monolith scan ({args.repo_name}): "
            f"{len(result.hard_fails)} monolith node(s) across "
            f"{len(node_dirs)} node dir(s)."
        )
        for line in result.report_lines:
            print(line)

    if args.write_baseline:
        if not args.check_all:
            print(
                "--write-baseline requires --check-all (a baseline must record "
                "the full current debt, not just the changed nodes).",
                file=sys.stderr,
            )
            return 1
        write_monolith_baseline(baseline_path, args.repo_name, result.hard_fails)
        print(f"Wrote Signal B baseline: {baseline_path}")
        return 0

    failures: list[str] = []
    if args.strict:
        failures.extend(monolith_strict_violations(result.hard_fails, baseline))
    if args.ratchet:
        failures.extend(monolith_ratchet_violations(result.hard_fails, baseline))

    if failures:
        print("\nARCH-004 Signal B ratchet FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    return 0


__all__ = [
    "MonolithBaselineEntry",
    "MonolithScanResult",
    "discover_node_dirs",
    "node_dirs_for_changed_files",
    "scan_node_dirs_for_monolith",
    "load_monolith_baseline",
    "render_monolith_baseline_yaml",
    "write_monolith_baseline",
    "monolith_ratchet_violations",
    "monolith_strict_violations",
    "main",
    "MONOLITH_BASELINE_RELATIVE_PATH",
]


if __name__ == "__main__":
    raise SystemExit(main())
