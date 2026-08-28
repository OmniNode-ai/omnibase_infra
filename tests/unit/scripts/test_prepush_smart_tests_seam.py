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

OMN-16825 narrows that filter. "Under tests/integration/" was standing in for
"needs a live service", and the two are not the same set: tests/integration/
carries service-dependent suites AND service-free ones. The chain gates
(tests/integration/chains/) are integration-SHAPED -- they wire several real
components together -- but run entirely on ``EventBusInmemory`` and touch no
Postgres, no Kafka, no lane endpoint. They are also collected wholesale by the
REQUIRED ``Event Chain Gate`` job, so the path heuristic made the local
selector structurally blind to a required merge gate: a chain regression was
observable only after the push.

The classifier now carries an explicit allowlist of locally-runnable
integration prefixes. The default for everything else under tests/integration/
is unchanged and remains DEFER -- an unrecognised subtree is deferred, never
silently included, so the narrowing is additive and fail-closed. The tests
below pin all three halves: the allowlisted prefix is kept, a genuinely
service-dependent subtree is still dropped (the negative control), and an
unclassified subtree defers rather than leaking through.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK = REPO_ROOT / "scripts/hooks/prepush_smart_tests.sh"
FUNCTION_NAME = "filter_prepush_runnable_paths"

# The one integration subtree the hook is allowed to run locally (OMN-16825).
# Kept here as a literal so a directory-wide reversion in the hook turns this
# module red instead of silently restoring the blind spot (AC5).
LOCALLY_RUNNABLE_INTEGRATION_PREFIX = "tests/integration/chains/"

# Markers from pyproject.toml [tool.pytest.ini_options] that declare a
# dependency on something the local pre-push hook cannot provide. The
# allowlisted subtree must never acquire one of these: if it does, the premise
# of the allowlist ("these suites need no live service") is false and this
# module must go red rather than let the hook run a suite that will fail or
# hang on a developer's machine.
LIVE_SERVICE_MARKERS = frozenset(
    {
        "consul",
        "database",
        "e2e",
        "heavy",
        "kafka",
        "linear",
        "live_github_api",
        "llm",
        "postgres",
        "real_mcp",
        "runtime",
    }
)


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


# ---------------------------------------------------------------------------
# OMN-16825: narrowed live-service classification
# ---------------------------------------------------------------------------


def test_chain_suites_are_locally_runnable(tmp_path: Path) -> None:
    """AC1/AC2: the allowlisted chain subtree survives the filter.

    Before OMN-16825 this returned `[]` and the hook logged the directory in
    its "deferred to CI (integration needs live services...)" line -- for a
    suite that needs no live service and gates the merge.
    """
    assert _run_filter([LOCALLY_RUNNABLE_INTEGRATION_PREFIX], tmp_path) == [
        LOCALLY_RUNNABLE_INTEGRATION_PREFIX
    ]


def test_chain_suite_file_paths_are_locally_runnable(tmp_path: Path) -> None:
    """File-grain selections of the same subtree are runnable too."""
    module = f"{LOCALLY_RUNNABLE_INTEGRATION_PREFIX}test_event_chain_gate.py"
    assert _run_filter([module], tmp_path) == [module]


def test_chain_suites_survive_alongside_deferred_integration_paths(
    tmp_path: Path,
) -> None:
    """A mixed selection keeps the chains and drops the service-dependent rest."""
    kept = _run_filter(
        [
            "tests/unit/runtime/",
            "tests/integration/db/",
            LOCALLY_RUNNABLE_INTEGRATION_PREFIX,
            "tests/integration/docker/",
        ],
        tmp_path,
    )
    assert kept == ["tests/unit/runtime/", LOCALLY_RUNNABLE_INTEGRATION_PREFIX]


@pytest.mark.parametrize(
    "service_dependent_path",
    [
        # Postgres-backed
        "tests/integration/db/",
        "tests/integration/migrations/test_forward_migrations.py",
        # Kafka / broker-backed
        "tests/integration/event_bus/",
        "tests/integration/dlq/",
        # lane endpoint / container-backed
        "tests/integration/docker/",
        "tests/integration/runtime/",
        "tests/integration/gateway/",
    ],
)
def test_service_dependent_integration_paths_remain_deferred(
    service_dependent_path: str, tmp_path: Path
) -> None:
    """AC3 negative control: the narrowing did not open the whole directory.

    Each of these needs something the local hook cannot provide (Postgres,
    a broker, a running lane). They must still be handed to CI.
    """
    assert _run_filter([service_dependent_path], tmp_path) == []


@pytest.mark.parametrize(
    "unclassified_path",
    [
        # A subtree that does not exist today: the classifier has no evidence
        # either way, so it must defer, not guess.
        "tests/integration/some_future_subtree/",
        # Prefix-collision probes: "chains" is a path SEGMENT, not a substring.
        "tests/integration/chains_experimental/",
        "tests/integration/chain/",
        "tests/integration/chainsaw/test_x.py",
        # A file sitting directly under tests/integration/ is unclassified.
        "tests/integration/test_agent_identity_e2e.py",
    ],
)
def test_unclassified_integration_paths_defer_rather_than_leak(
    unclassified_path: str, tmp_path: Path
) -> None:
    """AC3: fail-closed default. Unrecognised == deferred, never included."""
    assert _run_filter([unclassified_path], tmp_path) == []


def test_hook_pins_the_allowlist_literally() -> None:
    """AC5: a directory-wide reversion in the hook is red, not invisible."""
    source = HOOK.read_text()
    assert LOCALLY_RUNNABLE_INTEGRATION_PREFIX in source
    assert "OMN-16825" in source


def test_chain_suites_declare_no_live_service_marker() -> None:
    """AC3/AC5: the allowlist's premise is asserted, not assumed.

    The allowlist says "this subtree needs no live service". Nothing stops a
    future commit from dropping a Postgres-backed test into it. This scans the
    subtree's real marker declarations so that commit reddens here instead of
    reddening every developer's pre-push hook.
    """
    chains_dir = REPO_ROOT / LOCALLY_RUNNABLE_INTEGRATION_PREFIX
    assert chains_dir.is_dir(), f"{chains_dir} is missing; update the allowlist"

    modules = sorted(chains_dir.glob("test_*.py"))
    assert modules, "no chain suites found; the allowlist would be pointless"

    offenders: list[str] = []
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            # Matches `pytest.mark.<name>` in both decorators and pytestmark.
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "mark"
                and isinstance(value.value, ast.Name)
                and value.value.id == "pytest"
                and node.attr in LIVE_SERVICE_MARKERS
            ):
                offenders.append(f"{module.relative_to(REPO_ROOT)}: {node.attr}")

    assert not offenders, (
        "live-service markers found inside the locally-runnable allowlist "
        f"{LOCALLY_RUNNABLE_INTEGRATION_PREFIX}: {offenders}. Either the test "
        "does not belong in chains/, or the allowlist must be narrowed."
    )


def test_blanket_integration_ignore_does_not_suppress_an_explicit_chain_target(
    tmp_path: Path,
) -> None:
    """The filter's decision must survive the hook's own pytest flags.

    The hook keeps `--ignore=tests/integration` as belt-and-braces against a
    transitively-pulled parent directory. `--ignore` prunes recursion below a
    collection root; an explicitly-named argument IS a root, so the chains
    target still collects. That interaction is what makes the classifier fix
    sufficient on its own -- pin it, so a pytest upgrade that changes it turns
    this red instead of quietly producing a gate that runs nothing.
    """
    env = dict(os.environ)
    # The hook unsets these before invoking pytest for the same reason
    # (OMN-15071): git exports them into hook processes and they override cwd.
    for leaked in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
    ):
        env.pop(leaked, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            LOCALLY_RUNNABLE_INTEGRATION_PREFIX,
            "--ignore=tests/integration",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{LOCALLY_RUNNABLE_INTEGRATION_PREFIX}test_event_chain_gate.py" in (
        result.stdout
    ), result.stdout[-4000:]
