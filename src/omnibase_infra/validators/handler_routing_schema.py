# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed gate for the handler_routing flat-schema defect class (OMN-14141).

A ``handler_routing.handlers[]`` entry MUST carry a nested
``handler: {name, module}`` mapping. The historical FLAT
``handler_class:`` / ``handler_module:`` string schema is not understood by the
auto-wiring discovery parser (``ModelHandlerRoutingEntry``): it silently parsed
to ``handlers=()`` — zero handlers, no error — and auto-wiring then reported
``EnumWiringOutcome.WIRED`` with zero dispatchers while STILL subscribing to and
committing Kafka offsets on the topic. That is silent phantom-wiring: the topic
looks consumed but no handler ever runs (the WI-14 root cause, OMN-14139 /
OMN-14135).

This gate is the mechanical enforcement companion (CLAUDE.md Rule #5) to the
runtime parse guard in
``omnibase_infra.runtime.auto_wiring.discovery._parse_handler_routing``. It scans
node ``contract.yaml`` files and fails when any ``handler_routing.handlers[]``
entry cannot be parsed into a dispatcher (missing nested ``handler`` mapping).

Legacy contracts that use the top-level ``handler: {module, class}`` fallback are
NOT flagged — that fallback declares an EMPTY / ABSENT ``handlers`` list, so it
has no offending ``handlers[]`` entry.

Usage (pre-commit, per-file):
    uv run python -m omnibase_infra.validators.handler_routing_schema \\
        src/omnibase_infra/nodes/<node>/contract.yaml

Usage (CI, whole tree):
    uv run python -m omnibase_infra.validators.handler_routing_schema src/omnibase_infra
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class RoutingSchemaFinding:
    path: Path
    handler_index: int
    operation: str
    found_keys: tuple[str, ...]

    def format(self) -> str:
        op = f" operation={self.operation!r}" if self.operation else ""
        return (
            f"{self.path}: handler_routing.handlers[{self.handler_index}]{op} "
            f"has no nested 'handler: {{name, module}}' mapping "
            f"(found keys: {list(self.found_keys)}). Flat "
            f"'handler_class'/'handler_module' entries silently parse to ZERO "
            f"dispatchers and phantom-wire the subscribed topic (OMN-14141)."
        )


def validate_file(path: Path) -> list[RoutingSchemaFinding]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        # Malformed / unreadable YAML is another gate's concern; do not crash here.
        return []

    if not isinstance(raw, dict):
        return []

    hr_raw = raw.get("handler_routing")
    if not isinstance(hr_raw, dict):
        return []

    handlers = hr_raw.get("handlers")
    if not isinstance(handlers, list):
        return []

    findings: list[RoutingSchemaFinding] = []
    for index, entry in enumerate(handlers):
        if not isinstance(entry, dict):
            findings.append(
                RoutingSchemaFinding(
                    path=path,
                    handler_index=index,
                    operation="",
                    found_keys=(type(entry).__name__,),
                )
            )
            continue
        if isinstance(entry.get("handler"), dict):
            continue  # valid nested shape — the only accepted form
        findings.append(
            RoutingSchemaFinding(
                path=path,
                handler_index=index,
                operation=str(entry.get("operation", "")),
                found_keys=tuple(sorted(entry.keys())),
            )
        )
    return findings


def validate_paths(paths: Sequence[Path]) -> list[RoutingSchemaFinding]:
    findings: list[RoutingSchemaFinding] = []
    for path in _iter_contract_files(paths):
        findings.extend(validate_file(path))
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
            "Fail-closed gate: reject handler_routing.handlers[] entries using "
            "the flat handler_class/handler_module schema (OMN-14141)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_SCAN_ROOT],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else list(argv))
    findings = validate_paths(args.paths)

    if not findings:
        return 0

    for finding in findings:
        sys.stderr.write(f"  {finding.format()}\n")

    sys.stderr.write(
        f"\n[handler-routing-schema-gate] {len(findings)} flat-schema "
        "handler_routing entr(y/ies) found.\n"
        "Every routed handler must use the nested shape:\n"
        "    handler:\n"
        "      name: <HandlerClassName>\n"
        "      module: <module.path>\n"
        "The flat 'handler_class'/'handler_module' shape silently produces zero "
        "dispatchers and phantom-wires the subscribed topic (OMN-14141).\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
