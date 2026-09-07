# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The burned-down baselines stay deleted (OMN-18013).

OPERATOR RULING (2026-09-06), ITEM 5
------------------------------------
Burn the subscriber-dispatcher-resolution and contract-topic-graph baselines to
zero and DELETE the files, so the gate cannot be re-frozen.

WHY DELETION, NOT AN EMPTY FILE
-------------------------------
A shrink-only ratchet is a genuine improvement over an amnesty list, but it has
one failure mode that an empty list does not close: the next person who cannot
make a gate pass can add a row. The OMN-14605 mixed-category baseline was
authored with "THIS IS A RATCHET, NOT AN EXEMPTION" in its own header and still
sat at 4 rows for four months; the OMN-16939 baseline sat at 22 and its
omnimarket twin at 50, and every one of the OMN-16939 live victims — a projection
taking 174 messages and DLQ'ing 174 over six hours — was a baselined row. A file
that does not exist cannot take a row.

WHAT THIS REFUSES
-----------------
The recreation of any burned baseline file, and any ``--baseline`` flag
re-appearing on a validator that no longer has one. Both are checked from the
repo tree, so a PR that adds either fails before it merges.

Usage (pre-commit / CI):
    uv run python -m omnibase_infra.validators.no_baseline_refreeze
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Baseline files OMN-18013 burned to zero and deleted, in EVERY repo that carried
# one. The path is relative to the repo root the checker is pointed at, so the
# same list guards omnibase_infra and omnimarket without a second copy.
BURNED_BASELINES: tuple[str, ...] = (
    "config/validation/subscriber_dispatcher_resolution_baseline.yaml",
    "config/validation/mixed_category_routing_baseline.yaml",
    # OMN-16088 fan-out ratchet. Its last five rows were cleared by the
    # per-topic `topic:` scoping OMN-18013 applied, and its own text named that
    # split as the fix for each one. Burned to zero and deleted with the rest.
    "config/validation/operation_match_fanout_baseline.yaml",
    "src/omnimarket/validators/data/contract_topic_graph_baseline.yaml",
    "src/omnimarket/validators/data/contract_topic_graph_orphan_classification.yaml",
    "src/omnibase_infra/validators/data/contract_topic_graph_baseline.yaml",
)

# Validators that must never regrow a baseline flag.
BASELINE_FREE_VALIDATORS: tuple[str, ...] = (
    "src/omnibase_infra/validators/subscriber_dispatcher_resolution.py",
    "src/omnibase_infra/validators/contract_topic_category.py",
    "src/omnibase_infra/validators/contract_topic_graph.py",
    "src/omnibase_infra/validators/handler_event_type_source.py",
    "src/omnibase_infra/validators/no_literal_event_type_in_tests.py",
    "src/omnibase_infra/validators/operation_match_fanout.py",
)

_BASELINE_FLAG = re.compile(r'["\']--baseline["\']')


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class RefreezeFinding:
    """One attempt to bring a burned baseline back."""

    path: str
    detail: str


def findings(repo_root: Path) -> list[RefreezeFinding]:
    """Every burned baseline that exists again, or validator that regrew the flag."""
    out: list[RefreezeFinding] = []
    for rel in BURNED_BASELINES:
        candidate = repo_root / rel
        if candidate.exists():
            out.append(
                RefreezeFinding(
                    path=rel,
                    detail=(
                        "this baseline was burned to zero and DELETED by OMN-18013. "
                        "Its existence means a gate has been re-frozen. Fix the "
                        "contract instead — declare the publisher, declare the sink, "
                        "or delete the dead subscription."
                    ),
                )
            )
    for rel in BASELINE_FREE_VALIDATORS:
        candidate = repo_root / rel
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8")
        if _BASELINE_FLAG.search(text):
            out.append(
                RefreezeFinding(
                    path=rel,
                    detail=(
                        "this validator regrew a `--baseline` flag. OMN-18013 removed "
                        "it deliberately: a gate that accepts a baseline is a gate that "
                        "will be frozen the first time it is inconvenient."
                    ),
                )
            )
    return out


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refuse the recreation of a baseline OMN-18013 burned and deleted."
    )
    parser.add_argument("repo_root", nargs="?", default=".")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    repo_root = Path(args.repo_root)
    found = findings(repo_root)
    if found:
        sys.stderr.write(
            "[no-baseline-refreeze] FAIL: a burned gate baseline is back (OMN-18013):\n"
        )
        for f in found:
            sys.stderr.write(f"  - {f.path}\n      {f.detail}\n")
        return 1
    sys.stderr.write(
        f"[no-baseline-refreeze] OK: all {len(BURNED_BASELINES)} burned baselines "
        f"remain deleted and all {len(BASELINE_FREE_VALIDATORS)} validators remain "
        "baseline-free.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
