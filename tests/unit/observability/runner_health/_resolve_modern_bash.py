# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared bash>=5 interpreter resolution for the runner-monitor.sh harness.

OMN-15617: on stickybeatz-studio (.200, the rule-11a default gate host),
non-interactive ssh resolves the system bash 3.2.57 first on PATH, silently
failing every `declare -A` predicate in runner-monitor.sh. These tests drive
the REAL script end-to-end (deliberately -- see the module docstrings in the
two callers), so they need a real bash>=5, resolved explicitly rather than
trusted to ambient PATH order.

Deliberately NOT a Python re-implementation of the bash version check: it
shells out to scripts/ci/resolve_modern_bash.sh, the single source of truth
also used by the pre-push canary (scripts/hooks/prepush_smart_tests.sh), so
the two can never drift apart (DRY, root CLAUDE.md operating discipline).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[4]
RESOLVE_SCRIPT = REPO_ROOT / "scripts" / "ci" / "resolve_modern_bash.sh"


def resolve_modern_bash() -> str:
    """Return an absolute path to a bash>=5 interpreter, or fail loud.

    Fails the test via ``pytest.fail`` (never a silent skip or a quiet
    fallback to whatever "bash" resolves first on PATH) when no bash>=5
    interpreter can be found anywhere -- that quiet-fallback shape is the
    exact bug OMN-15617 exists to close.
    """
    if not RESOLVE_SCRIPT.is_file():
        pytest.fail(f"resolver script missing: {RESOLVE_SCRIPT}")
    # The resolver script itself is bash-3.2-safe by construction (it cannot
    # presuppose the modern bash it is trying to find), so it is safe to
    # invoke via the bare "bash" on whatever PATH this process inherited.
    result = subprocess.run(
        ["bash", str(RESOLVE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    resolved = result.stdout.strip()
    if result.returncode != 0 or not resolved:
        pytest.fail(
            "no bash>=5 interpreter resolvable for the runner-monitor.sh "
            "harness (declare -A requires bash>=4; OMN-15617).\n"
            f"resolver stderr:\n{result.stderr}"
        )
    return resolved
