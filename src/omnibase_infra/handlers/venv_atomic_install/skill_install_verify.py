# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Steps 3 and 4 of ``scripts/install-node-skill-package.sh``, moved here (OMN-22712).

Run (by the node runtime's ``verify`` command) under the interpreter being checked, so the entry points it reads are that
interpreter's. Step 3: the nodes behind the operator skills this install must
keep resolvable are still advertised. Step 4 (OMN-18675 AC2): every advertised
``onex.cli`` entry point still LOADS; a co-install that leaves the venv
importable-but-broken is not a success, because one entry point that raises takes
down every ``onex`` subcommand.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, distribution, entry_points
from typing import TextIO

# merge_sweep -> node_pr_lifecycle_orchestrator, session -> node_session_orchestrator,
# aislop_sweep kept as a broad-coverage canary.
REQUIRED_NODES = frozenset(
    {"node_pr_lifecycle_orchestrator", "node_session_orchestrator", "node_aislop_sweep"}
)


# Anything an entry point's import can raise; any of them breaks the CLI.
_LOAD_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    ValueError,
    TypeError,
    OSError,
    RuntimeError,
    SyntaxError,
)


def _say(text: str = "", *, file: TextIO | None = None) -> None:
    """One line to stdout, or to ``file``; the tool's output is its interface."""
    (file or sys.stdout).write(text + "\n")


def verify_nodes() -> int:
    _say("== step 3: verify node resolution ==")
    eps = {e.name for e in entry_points(group="onex.nodes")}
    missing = sorted(REQUIRED_NODES - eps)
    if missing:
        _say(f"FAIL: mapped nodes still unresolved: {missing}", file=sys.stderr)
        return 1
    _say(
        f"OK: {len(eps)} onex.nodes entry points; required nodes resolved: {sorted(REQUIRED_NODES)}"
    )
    return 0


def verify_cli() -> int:
    _say("== step 4: verify every advertised onex.cli entry point imports ==")
    failures: list[tuple[str, str]] = []
    eps = list(entry_points(group="onex.cli"))
    for ep in eps:
        try:
            ep.load()
        except _LOAD_ERRORS as exc:
            failures.append((ep.name, f"{type(exc).__name__}: {exc}"))
    if failures:
        _say(
            "FAIL: the onex CLI does not load — these entry points raise:",
            file=sys.stderr,
        )
        for name, reason in failures:
            _say(f"  {name}: {reason}", file=sys.stderr)
        _say(
            "\nThe venv is left in this state; `onex <any subcommand>` will raise.\n"
            "Re-install the dependency the failing entry point imports.",
            file=sys.stderr,
        )
        return 1
    _say(f"OK: all {len(eps)} onex.cli entry points import cleanly.")
    return 0


def installed_market_commit() -> int:
    """Print the commit omnimarket was installed from ("" when absent): the
    build a dispatch from this interpreter would actually use."""
    try:
        raw = distribution("omnimarket").read_text("direct_url.json") or ""
    except PackageNotFoundError:
        _say("")
        return 0
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {}
    _say(data.get("vcs_info", {}).get("commit_id", ""))
    return 0
