# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer-seam tests for scripts/hooks/prepush_smart_tests.sh (OMN-15245).

The governed selector is fail-closed about changed test modules: any changed
path under tests/ is unconditionally selected, including tests/integration/**.
The pre-push hook is unit-scoped by design and passes
`--ignore=tests/integration` to pytest, so it must filter those paths out of
the pytest invocation -- otherwise pytest exits 5 ("no tests ran") whenever a
diff's only selected path is an integration directory, and every such push is
blocked by a gate that ran nothing.

These tests EXECUTE the real bash function extracted from the hook (the
OMN-15218 executed-seam pattern); they do not grep for it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK = REPO_ROOT / "scripts/hooks/prepush_smart_tests.sh"
FUNCTION_NAME = "filter_prepush_runnable_paths"


def _extract_function(source: str, name: str) -> str:
    lines = source.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"{name}() {{")),
        None,
    )
    assert start is not None, f"{name}() not found in {HOOK}"
    end = next((i for i in range(start + 1, len(lines)) if lines[i] == "}"), None)
    assert end is not None, f"unterminated {name}() in {HOOK}"
    return "\n".join(lines[start : end + 1])


def _run_filter(paths: list[str], tmp_path: Path) -> list[str]:
    bash = shutil.which("bash")
    assert bash is not None, "bash not available"
    fragment = tmp_path / "fragment.sh"
    fragment.write_text(
        "set -euo pipefail\n" + _extract_function(HOOK.read_text(), FUNCTION_NAME),
        encoding="utf-8",
    )
    result = subprocess.run(
        [bash, "-c", f'. "{fragment}"; {FUNCTION_NAME}'],
        input="".join(f"{p}\n" for p in paths),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line]


def test_hook_exists_and_defines_the_filter() -> None:
    assert HOOK.is_file()
    assert f"{FUNCTION_NAME}() {{" in HOOK.read_text()


def test_integration_paths_are_filtered_out(tmp_path: Path) -> None:
    kept = _run_filter(
        [
            "tests/ci/",
            "tests/integration/infra/",
            "tests/unit/cli/",
            "tests/integration/runtime/",
            "tests/scripts/",
        ],
        tmp_path,
    )
    assert kept == ["tests/ci/", "tests/unit/cli/", "tests/scripts/"]


def test_integration_only_selection_yields_no_pytest_paths(tmp_path: Path) -> None:
    # The exit-5 case: without the filter this hands pytest a path it also
    # ignores, and pytest exits 5, blocking a push on a gate that ran nothing.
    assert _run_filter(["tests/integration/infra/"], tmp_path) == []


def test_non_integration_paths_pass_through_unchanged(tmp_path: Path) -> None:
    paths = ["tests/unit/", "tests/unit/scripts/", "tests/replay/", "tests/ci/"]
    assert _run_filter(paths, tmp_path) == paths


def test_hook_still_ignores_integration_in_the_pytest_invocation() -> None:
    # The filter complements --ignore=tests/integration; it does not replace it.
    # A source diff can still pull an integration test in transitively.
    assert HOOK.read_text().count("--ignore=tests/integration") >= 2
