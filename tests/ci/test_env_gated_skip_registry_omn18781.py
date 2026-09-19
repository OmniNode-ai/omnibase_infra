# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18781 — a suite may not skip on an env var nobody is accountable for.

WHY THIS EXISTS
---------------
``tests/integration/event_bus/test_kafka_event_bus_integration.py`` and
``tests/integration/handlers/test_handler_qdrant_integration.py`` each carried a
module-level ``pytest.mark.skipif`` on an opt-in environment variable that **no
workflow in this repository set**. Neither module carried a marker the PR test
splits deselect, so all 27 cases were collected on every pull request, skipped
in full, and counted toward a green Tests job. The event-bus one is the bus's
own integration proof; it had never executed in CI.

Nothing surfaced that, because a skipped test and a deselected test look the
same from outside, and the OMN-14172 silent-skip guard only ever read the
curated Postgres junit.

WHAT IS PINNED HERE
-------------------
``test_every_env_gated_module_skip_is_registered``
    the STRUCTURAL ratchet. Every module-level skip whose condition reads an
    environment variable must appear in ``config/env_gated_test_skips.yaml``,
    either as a suite with an execution path — naming the workflow, which must
    PROVABLY set every variable the skip reads — or as an explicitly
    unprovisioned suite carrying a ticket. A registered workflow that stops
    setting the variable is therefore a red test, not a discovery months later.

``test_registry_has_no_stale_entries``
    the other direction. An entry for a module that no longer has an env-gated
    skip is removed, so the file cannot rot into a list nobody reads.

``test_scanner_detects_the_shape_it_exists_to_catch``
    the POSITIVE CONTROL. A scanner that silently matches nothing would make
    every other assertion here vacuous, so it is driven against a synthetic
    module in the exact shape of the original defect and must find it.

``test_workflow_setter_detection_rejects_a_var_the_workflow_does_not_set``
    the second positive control, for the half that does the real work: the
    setter detector must say no when asked about a variable a workflow does not
    set. Without this, a detector that returned True unconditionally would pass
    every registry entry.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Final

import pytest
import yaml

pytestmark = [pytest.mark.ci]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
TESTS_ROOT: Final[Path] = REPO_ROOT / "tests"
WORKFLOWS_ROOT: Final[Path] = REPO_ROOT / ".github" / "workflows"
REGISTRY_PATH: Final[Path] = REPO_ROOT / "config" / "env_gated_test_skips.yaml"

# An env read, in either spelling. Restricted to SHOUTY names so a lowercase
# local dict lookup is not mistaken for an environment variable.
_ENV_READ: Final[re.Pattern[str]] = re.compile(
    r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Z][A-Z0-9_]*)["']"""
    r"""|os\.environ\[\s*["']([A-Z][A-Z0-9_]*)["']\s*\]"""
)

_TICKET: Final[re.Pattern[str]] = re.compile(r"^OMN-\d+$")


def _env_names(source_segment: str) -> set[str]:
    return {first or second for first, second in _ENV_READ.findall(source_segment)}


def env_gated_skips(source: str) -> dict[str, set[str]]:
    """Return ``{skip site name: env vars its condition reads}`` for one module.

    A "site" is a module-level construct that can skip the WHOLE module or every
    test in it:

    * a module-level assignment whose value builds a ``pytest.mark.skipif`` —
      ``pytestmark`` itself, and also a reusable marker constant such as
      ``requires_kafka`` that is then applied per test;
    * a module-level ``if`` that calls ``pytest.skip(..., allow_module_level=True)``.

    Env reads are resolved THROUGH module constants: the common shape assigns
    ``KAFKA_AVAILABLE = os.getenv(...) == "1"`` and then references that name in
    the ``skipif``, so a scanner that only read the ``skipif`` expression itself
    would see no variable at all and report a clean tree.
    """
    tree = ast.parse(source)
    constant_env: dict[str, set[str]] = {}
    sites: dict[str, set[str]] = defaultdict(set)

    def resolved(segment: str) -> set[str]:
        names = _env_names(segment)
        for constant, constant_names in constant_env.items():
            if re.search(rf"\b{re.escape(constant)}\b", segment):
                names |= constant_names
        return names

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name) or node.value is None:
                continue
            segment = ast.get_source_segment(source, node.value) or ""
            names = resolved(segment)
            if names:
                constant_env[target.id] = names
            if ("skipif" in segment or "allow_module_level" in segment) and names:
                sites[target.id] |= names
        elif isinstance(node, ast.If):
            segment = ast.get_source_segment(source, node) or ""
            if "allow_module_level" not in segment:
                continue
            names = resolved(segment)
            if names:
                sites["<module-level skip>"] |= names

    return dict(sites)


def scan_tests_tree() -> dict[str, set[str]]:
    """Return ``{repo-relative module path: env vars its module-level skips read}``."""
    found: dict[str, set[str]] = {}
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - unreadable file
            continue
        if "skipif" not in source and "allow_module_level" not in source:
            continue
        try:
            sites = env_gated_skips(source)
        except SyntaxError:  # pragma: no cover - not this test's job to report
            continue
        if not sites:
            continue
        names: set[str] = set()
        for site_names in sites.values():
            names |= site_names
        found[str(path.relative_to(REPO_ROOT))] = names
    return found


def workflow_sets(workflow_source: str, variable: str) -> bool:
    """True when this workflow provably exports ``variable`` to a test process.

    Three spellings count, and they are the three this repository actually uses:
    a YAML ``env:`` key, a value written into ``$GITHUB_ENV``, and a
    ``--env VAR=`` passed to a container. A bare mention in a comment or in prose
    does not count — that is the difference between this check and a grep.
    """
    patterns = (
        rf"^\s*{re.escape(variable)}\s*:",  # env: key
        rf"^\s*echo\s+[\"']?{re.escape(variable)}=",  # >> $GITHUB_ENV
        rf"{re.escape(variable)}=.*>>\s*[\"']?\$GITHUB_ENV",  # same, one line
        rf"--env\s+{re.escape(variable)}=",  # docker run
    )
    return any(re.search(p, workflow_source, re.MULTILINE) for p in patterns)


def load_registry() -> dict[str, dict[str, object]]:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    suites = raw.get("suites")
    if not isinstance(suites, dict):
        raise AssertionError(
            f"{REGISTRY_PATH} must declare a `suites:` mapping; got {type(suites)!r}"
        )
    return suites


def test_every_env_gated_module_skip_is_registered() -> None:
    """A module-level env-gated skip must name where it runs, or name a ticket.

    RED proof: add a module-level ``pytestmark = [pytest.mark.skipif(
    not os.getenv("SOME_VAR"), reason=...)]`` to any test module and this test
    names that module and that variable.
    """
    scanned = scan_tests_tree()
    registry = load_registry()
    violations: list[str] = []

    for module, variables in sorted(scanned.items()):
        entry = registry.get(module)
        if entry is None:
            violations.append(
                f"UNREGISTERED: {module} skips on {sorted(variables)} and is absent "
                f"from {REGISTRY_PATH.relative_to(REPO_ROOT)}. Either give it an "
                f"execution path (a workflow that provisions its dependency and "
                f"sets the variable) or register it as unprovisioned with a ticket."
            )
            continue

        declared_raw = entry.get("env_vars")
        declared = set(declared_raw) if isinstance(declared_raw, list) else set()
        if declared != variables:
            violations.append(
                f"DRIFT: {module} skips on {sorted(variables)} but its registry "
                f"entry declares {sorted(declared)}. A variable added to a skip "
                f"condition must be accounted for in the same change."
            )

        ticket = entry.get("unprovisioned_ticket")
        if ticket is not None:
            if not _TICKET.match(str(ticket)):
                violations.append(
                    f"BAD TICKET: {module} declares unprovisioned_ticket "
                    f"{ticket!r}, which is not an OMN-<number> reference."
                )
            if not str(entry.get("reason", "")).strip():
                violations.append(
                    f"NO REASON: {module} is registered as unprovisioned with no "
                    f"reason. An unprovisioned suite is a residual, and a residual "
                    f"with no stated cost is indistinguishable from an oversight."
                )
            continue

        workflow_name = entry.get("workflow")
        if not workflow_name:
            violations.append(
                f"NO HOME: {module} declares neither a workflow nor an "
                f"unprovisioned_ticket. Every env-gated skip needs one or the other."
            )
            continue

        workflow_path = REPO_ROOT / str(workflow_name)
        if not workflow_path.is_file():
            violations.append(
                f"MISSING WORKFLOW: {module} names {workflow_name}, which does not "
                f"exist."
            )
            continue

        workflow_source = workflow_path.read_text(encoding="utf-8")
        for variable in sorted(variables):
            if not workflow_sets(workflow_source, variable):
                violations.append(
                    f"NOT SET: {module} skips on {variable}, and {workflow_name} — "
                    f"the workflow its registry entry names as the surface that "
                    f"runs it — does not set that variable. This is the exact "
                    f"condition OMN-18781 closed: a suite gated on a variable no "
                    f"workflow sets, skipping in silence."
                )

    assert not violations, "\n".join(violations)


def test_registry_has_no_stale_entries() -> None:
    """An entry whose module no longer has an env-gated skip must be removed."""
    scanned = scan_tests_tree()
    stale = [module for module in load_registry() if module not in scanned]
    assert not stale, (
        "Stale entries in "
        f"{REGISTRY_PATH.relative_to(REPO_ROOT)}: {stale}. These modules no longer "
        "carry a module-level env-gated skip. Remove them, so the registry stays a "
        "description of the tree rather than a list nobody reads."
    )


def test_scanner_detects_the_shape_it_exists_to_catch() -> None:
    """Positive control: the scanner finds the original defect's exact shape.

    Without this, a scanner that matched nothing would make the ratchet above
    pass on an empty set and read as a clean bill of health.
    """
    synthetic = (
        "import os\n"
        "import pytest\n"
        'SOMETHING_AVAILABLE = os.getenv("SOMETHING_INTEGRATION_TESTS") == "1"\n'
        "pytestmark = [\n"
        "    pytest.mark.skipif(\n"
        "        not SOMETHING_AVAILABLE,\n"
        '        reason="Something not available",\n'
        "    ),\n"
        "]\n"
    )
    sites = env_gated_skips(synthetic)
    assert sites == {"pytestmark": {"SOMETHING_INTEGRATION_TESTS"}}, sites

    module_level = (
        "import os\n"
        "import pytest\n"
        'if not os.environ.get("OTHER_REQUIRE_FLAG"):\n'
        '    pytest.skip("flag unset", allow_module_level=True)\n'
    )
    assert env_gated_skips(module_level) == {
        "<module-level skip>": {"OTHER_REQUIRE_FLAG"}
    }

    # And the live tree is not silently empty: the registry describes real
    # modules, so a scan that returned nothing would mean the scanner broke.
    assert len(scan_tests_tree()) >= 10


def test_workflow_setter_detection_rejects_a_var_the_workflow_does_not_set() -> None:
    """Positive control for the half that does the work.

    A detector that answered True unconditionally would admit every registry
    entry, including one naming a workflow that sets nothing.
    """
    source = (
        "jobs:\n"
        "  demo:\n"
        "    env:\n"
        "      SET_BY_YAML_KEY: '1'\n"
        "    steps:\n"
        "      - run: |\n"
        '          echo "SET_BY_GITHUB_ENV=yes" >> "$GITHUB_ENV"\n'
        "      - run: docker run --env SET_BY_DOCKER=1 image\n"
        "      # MENTIONED_IN_A_COMMENT is not set by this workflow\n"
    )
    assert workflow_sets(source, "SET_BY_YAML_KEY")
    assert workflow_sets(source, "SET_BY_GITHUB_ENV")
    assert workflow_sets(source, "SET_BY_DOCKER")
    assert not workflow_sets(source, "MENTIONED_IN_A_COMMENT")
    assert not workflow_sets(source, "NEVER_APPEARS_AT_ALL")


def test_the_two_repaired_suites_no_longer_skip_on_an_env_var() -> None:
    """The suites this ticket repaired must not reappear in the scan.

    They are selected by marker now — ``kafka`` and ``qdrant``, both deselected
    by the PR splits — and their CI disposition is a FAILURE, not a skip. If a
    later change reintroduces a module-level env skipif on either, it belongs in
    the registry with a workflow behind it, and this names them first.
    """
    scanned = scan_tests_tree()
    for repaired in (
        "tests/integration/event_bus/test_kafka_event_bus_integration.py",
        "tests/integration/handlers/test_handler_qdrant_integration.py",
    ):
        assert repaired not in scanned, (
            f"{repaired} has a module-level env-gated skip again. OMN-18781 "
            f"replaced it with marker selection plus a fail-closed CI check "
            f"precisely so this suite cannot silently skip."
        )
