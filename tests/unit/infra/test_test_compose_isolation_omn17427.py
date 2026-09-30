# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No test may run a mutating ``docker compose`` verb on a declared lane (OMN-17427).

On 2026-09-29T23:57:58Z ``tests/integration/test_catalog_roundtrip.py`` ran
``docker compose -f docker/docker-compose.generated.yml up -d`` and then
``down`` on the .201 lab host. The generated file's top-level ``name:`` was
``omnibase-infra`` and its container names were the dev lane's, so the test
recreated and then removed the dev lane's postgres, redpanda, valkey, keycloak
and infisical containers.

Two ratchets, both static (no docker daemon, so they run in unit CI):

1. Every ``subprocess`` call in ``tests/`` whose argv resolves to
   ``docker compose ... <up|down|start|stop|restart|rm|kill|create|run|pause>``
   must name its project with ``-p``/``--project-name``, and that project must
   not be a declared lane's (the lane set comes from
   ``deploy/lane-census/lane-manifest.yaml``). Without ``-p`` the compose file's
   own ``name:`` decides, which is how the incident happened.
2. The document the catalog round-trip test starts is isolated: its project,
   container names, volumes and networks are the test's own.

Each ratchet carries a red control that reconstructs the incident's shape and
shows the check rejects it.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from tests.helpers.compose_isolation import (
    compose_isolation_violations,
    declared_lane_projects,
    isolated_project_name,
    without_host_ports,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = REPO_ROOT / "tests"
CATALOG_DIR = REPO_ROOT / "docker" / "catalog"

MUTATING_VERBS = frozenset(
    {"up", "down", "start", "stop", "restart", "rm", "kill", "create", "run", "pause"}
)
_SUBPROCESS_FUNCS = frozenset({"run", "call", "check_call", "check_output", "Popen"})
_GLOBAL_OPTS_WITH_VALUE = frozenset(
    {
        "-p",
        "--project-name",
        "-f",
        "--file",
        "--env-file",
        "--project-directory",
        "--profile",
        "--ansi",
        "--progress",
        "--parallel",
    }
)
_UNRESOLVED = object()


def _assignments(scope: ast.AST) -> dict[str, ast.expr]:
    found: dict[str, ast.expr] = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                found[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                found[node.target.id] = node.value
    return found


def _flatten(
    expr: ast.expr, names: dict[str, ast.expr], depth: int = 0
) -> list[object]:
    """An argv expression as constants, ``ast`` nodes for dynamic parts, or _UNRESOLVED."""
    if depth > 4:
        return [_UNRESOLVED]
    if isinstance(expr, ast.Name) and expr.id in names:
        return _flatten(names[expr.id], names, depth + 1)
    if isinstance(expr, ast.List | ast.Tuple):
        out: list[object] = []
        for elt in expr.elts:
            if isinstance(elt, ast.Starred):
                out.extend(_flatten(elt.value, names, depth + 1))
            elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            else:
                out.append(elt)
        return out
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _flatten(expr.left, names, depth + 1) + _flatten(
            expr.right, names, depth + 1
        )
    return [_UNRESOLVED]


def _project_is_lane(value: object, names: dict[str, ast.expr]) -> bool | None:
    """True/False for a resolvable project value; None when it is dynamic."""
    if isinstance(value, str):
        return value in declared_lane_projects()
    if isinstance(value, ast.Name) and value.id in names:
        return _project_is_lane(names[value.id], names)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value in declared_lane_projects()
    return None  # an f-string, a call or an attribute: computed per run


def compose_violations(source: str, label: str) -> list[str]:
    """Every subprocess call in ``source`` that runs a mutating compose verb unsafely."""
    tree = ast.parse(source)
    module_names = _assignments(tree)
    problems: list[str] = []
    scopes: list[ast.AST] = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    seen: set[int] = set()
    for scope in [*scopes, tree]:
        names = {**module_names, **(_assignments(scope) if scope is not tree else {})}
        for node in ast.walk(scope):
            if id(node) in seen or not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in _SUBPROCESS_FUNCS
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
            ):
                continue
            seen.add(id(node))
            argv = _flatten(node.args[0], names)
            if argv[:2] == ["docker", "compose"]:
                rest = argv[2:]
            elif argv[:1] == ["docker-compose"]:
                rest = argv[1:]
            else:
                continue
            project: object = None
            verb: object = None
            i = 0
            while i < len(rest):
                item = rest[i]
                if isinstance(item, str) and item.startswith("-"):
                    if item in ("-p", "--project-name") and i + 1 < len(rest):
                        project = rest[i + 1]
                    if item.startswith("--project-name="):
                        project = item.split("=", 1)[1]
                    i += 2 if item in _GLOBAL_OPTS_WITH_VALUE else 1
                    continue
                verb = item
                break
            if not (isinstance(verb, str) and verb in MUTATING_VERBS):
                continue
            where = f"{label}:{node.lineno}"
            if project is None:
                problems.append(
                    f"{where}: docker compose {verb} with no -p, so the compose "
                    "file's own name: picks the project"
                )
            elif _project_is_lane(project, names):
                problems.append(
                    f"{where}: docker compose {verb} on a declared lane project"
                )
    return problems


# ----------------------------------------------------------------------------- ratchet 1


def test_lane_set_comes_from_the_lane_manifest() -> None:
    """The declared lanes the incident named are all in the manifest-derived set."""
    assert {
        "omnibase-infra",
        "omnibase-infra-stability-test",
        "omnibase-infra-judge",
        "omnibase-infra-lakshman",
        "omnibase-infra-dogfood",
        "omninode-ci-bus",
    } <= declared_lane_projects()


def test_no_test_runs_a_mutating_compose_verb_on_a_lane_project() -> None:
    problems: list[str] = []
    scanned = 0
    for path in sorted(TESTS_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "compose" not in source:
            continue
        scanned += 1
        problems.extend(compose_violations(source, str(path.relative_to(REPO_ROOT))))
    assert scanned > 0, "positive control: the scan found no file mentioning compose"
    assert not problems, "\n".join(problems)


PRE_FIX_ROUNDTRIP = """
import subprocess
def test_x():
    subprocess.run(["docker", "compose", "-f", "docker/docker-compose.generated.yml", "up", "-d"])
    subprocess.run(["docker", "compose", "-f", "docker/docker-compose.generated.yml", "down"])
"""


def test_red_control_the_incident_shape_is_rejected() -> None:
    """The exact pre-fix argv (no -p, generated file named omnibase-infra) fails."""
    problems = compose_violations(PRE_FIX_ROUNDTRIP, "pre-fix")
    assert len(problems) == 2, problems


@pytest.mark.parametrize(
    ("source", "bad"),
    [
        (
            'import subprocess\nsubprocess.run(["docker", "compose", "-p", '
            '"omnibase-infra-judge", "down"])\n',
            True,
        ),
        (
            'import subprocess\nP = "omninode-ci-bus"\nC = ["docker", "compose", "-p", P]\n'
            'subprocess.run([*C, "stop"])\n',
            True,
        ),
        (
            'import os, subprocess\ndef t():\n    p = f"x-proof-{os.getpid()}"\n'
            '    c = ["docker", "compose", "-p", p, "-f", "f.yml"]\n'
            '    subprocess.run([*c, "up", "-d"])\n    subprocess.run([*c, "down", "-v"])\n',
            False,
        ),
        (
            'import subprocess\nsubprocess.run(["docker", "compose", "-p", '
            '"omnibase-infra-lakshman", "config"])\n',
            False,
        ),
    ],
)
def test_checker_controls(source: str, bad: bool) -> None:
    assert bool(compose_violations(source, "control")) is bad


# ----------------------------------------------------------------------------- ratchet 2


def _roundtrip_compose(project: str | None) -> dict[str, object]:
    resolved = CatalogResolver(catalog_dir=str(CATALOG_DIR)).resolve(["core"])
    if project is not None:
        resolved.project = project
    return without_host_ports(generate_compose(resolved, environment=os.environ))


def test_catalog_roundtrip_document_is_isolated() -> None:
    """The document the integration test starts touches nothing a lane owns."""
    compose = _roundtrip_compose(isolated_project_name("catalog-roundtrip"))
    assert compose_isolation_violations(compose) == []
    services = compose["services"]
    assert isinstance(services, dict)
    for svc in services.values():
        assert "ports" not in svc


def test_red_control_default_project_document_is_rejected() -> None:
    """The pre-fix document (default project, lane container names) is refused."""
    problems = compose_isolation_violations(_roundtrip_compose(None))
    assert any("project 'omnibase-infra'" in p for p in problems), problems
    assert any("omnibase-infra-postgres" in p for p in problems), problems
