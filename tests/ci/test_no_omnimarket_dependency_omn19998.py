# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnibase_infra must not depend on omnimarket (OMN-19998).

Repo layering puts omnimarket ABOVE omnibase_infra: market depends on infra,
never the other way round (operator ruling 2026-10-15T21:52:44Z). This module
is the guard, run by the ``no-omnimarket-dependency`` pre-commit hook and by
the Lint job of ``ci.yml`` (a strict member of the required CI Summary gate).

It refuses omnimarket on the package dependency surface:

* ``pyproject.toml``: any requirement naming omnimarket in
  ``[build-system] requires``, ``[project] dependencies``,
  ``[project.optional-dependencies]``, ``[dependency-groups]``, or the
  ``[tool.uv]`` dev, override and constraint lists; a ``[tool.uv.sources]``
  key for omnimarket; an entry point whose target is an omnimarket module.
* ``uv.lock``: a locked package named omnimarket, or any locked package
  depending on it.
* ``src/omnibase_infra``: an ``import omnimarket`` or ``from omnimarket ...``
  executed at import time, that is anywhere outside a function body.

OMN-21279 extends the guard past the package surface to the three places the
ruling (RULING 2026-10-15T21:52:44Z) also names:

* ``docker/Dockerfile.runtime``: any instruction line naming omnimarket, so the
  runtime image installs, copies from, pins or imports no omnimarket.
* CI install steps: a workflow or composite-action ``run`` step that installs
  omnimarket (``pip install``, ``uv pip install``, ``uv sync``, ``uv add``).
  The checker refuses any workflow text. The live-tree scan covers ``ci.yml``
  (the workflow behind the required CI Summary gate) and every composite
  action. Three other workflows (``pr-title-check``, ``r1-front-door-probe``,
  ``evidence-autoclose-sweep``) still install omnimarket to run market nodes
  from infra's CI; moving those jobs into omnimarket is the remaining work of
  OMN-21279 AC1 and is not scanned until they are moved.
* test fixtures: any ``import omnimarket`` / ``from omnimarket ...`` anywhere
  in ``tests/`` (function scope included), or an ``importlib.import_module`` /
  ``pytest.importorskip`` call naming omnimarket.

There is no allowlist and no suppression marker. The negative controls below
feed the same checkers a manifest, a lock and a module that do depend on
omnimarket and assert each one fires.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "omnibase_infra"
FORBIDDEN = "omnimarket"

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    return _normalize(match.group(1)) if match else ""


def _requirements(pyproject: Mapping[str, object]) -> Iterator[tuple[str, str]]:
    """Yield ``(table, requirement)`` for every requirement string declared."""
    build = pyproject.get("build-system", {})
    assert isinstance(build, dict)
    for req in build.get("requires", []):
        yield "build-system.requires", req
    project = pyproject.get("project", {})
    assert isinstance(project, dict)
    for req in project.get("dependencies", []):
        yield "project.dependencies", req
    for extra, reqs in project.get("optional-dependencies", {}).items():
        for req in reqs:
            yield f"project.optional-dependencies.{extra}", req
    groups = pyproject.get("dependency-groups", {})
    assert isinstance(groups, dict)
    for group, reqs in groups.items():
        for req in reqs:
            if isinstance(req, str):  # {include-group = ...} names no package
                yield f"dependency-groups.{group}", req
    tool = pyproject.get("tool", {})
    assert isinstance(tool, dict)
    uv = tool.get("uv", {})
    for key in ("dev-dependencies", "override-dependencies", "constraint-dependencies"):
        for req in uv.get(key, []):
            yield f"tool.uv.{key}", req


def pyproject_findings(text: str) -> list[str]:
    """Every place a pyproject.toml text makes omnimarket a dependency."""
    data = tomllib.loads(text)
    findings = [
        f"{table}: {req!r}"
        for table, req in _requirements(data)
        if _requirement_name(req) == FORBIDDEN
    ]
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    findings += [
        f"tool.uv.sources: {name!r}"
        for name in sources
        if _normalize(name) == FORBIDDEN
    ]
    project = data.get("project", {})
    entry_tables: dict[str, Mapping[str, str]] = {
        "project.scripts": project.get("scripts", {}),
        "project.gui-scripts": project.get("gui-scripts", {}),
    }
    for group, points in project.get("entry-points", {}).items():
        entry_tables[f"project.entry-points.{group}"] = points
    for table, points in entry_tables.items():
        for name, target in points.items():
            if target.strip().split(".")[0].split(":")[0] == FORBIDDEN:
                findings.append(f"{table}: {name} = {target!r}")
    return findings


def lock_findings(text: str) -> list[str]:
    """Every locked package that is, or depends on, omnimarket."""
    findings: list[str] = []
    for package in tomllib.loads(text).get("package", []):
        name = _normalize(package.get("name", ""))
        if name == FORBIDDEN:
            findings.append(f"uv.lock: package {name} is locked")
        deps = list(package.get("dependencies", []))
        for extra in package.get("optional-dependencies", {}).values():
            deps += extra
        for group in package.get("dev-dependencies", {}).values():
            deps += group
        for dep in deps:
            if _normalize(dep.get("name", "")) == FORBIDDEN:
                findings.append(f"uv.lock: package {name} depends on {FORBIDDEN}")
    return findings


class _ImportTimeImports(ast.NodeVisitor):
    """Collect omnimarket imports that run when the module is imported."""

    def __init__(self) -> None:
        self.depth = 0
        self.lines: list[int] = []

    def _function(self, node: ast.AST) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._function(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self.depth == 0 and any(
            alias.name.split(".")[0] == FORBIDDEN for alias in node.names
        ):
            self.lines.append(node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if (
            self.depth == 0
            and node.level == 0
            and (node.module or "").split(".")[0] == FORBIDDEN
        ):
            self.lines.append(node.lineno)


def source_findings(source: str, label: str) -> list[str]:
    """``label:line`` for each import-time omnimarket import in ``source``."""
    visitor = _ImportTimeImports()
    visitor.visit(ast.parse(source, filename=label))
    return [
        f"{label}:{line}: import-time import of {FORBIDDEN}" for line in visitor.lines
    ]


_FORBIDDEN_PATTERN = re.compile(FORBIDDEN, re.IGNORECASE)
_INSTALL_VERB = re.compile(
    r"\b(?:pip3?\s+install|uv\s+pip\s+install|uv\s+sync|uv\s+add|uv\s+tool\s+install"
    r"|poetry\s+add)\b"
)


def _logical_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(first physical line, text)`` per backslash-joined logical line.

    Whole-line ``#`` comments are dropped, as Docker and the shell both do.
    """
    pending: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        if not pending and raw.lstrip().startswith("#"):
            continue
        if raw.lstrip().startswith("#"):
            continue
        if not pending:
            start = number
        body = raw.rstrip()
        if body.endswith("\\"):
            pending.append(body[:-1])
            continue
        pending.append(body)
        yield start, " ".join(part.strip() for part in pending)
        pending = []
    if pending:
        yield start, " ".join(part.strip() for part in pending)


def dockerfile_findings(text: str, label: str) -> list[str]:
    """``label:line`` for each Dockerfile instruction line naming omnimarket."""
    return [
        f"{label}:{line}: names {FORBIDDEN}: {logical[:100]}"
        for line, logical in _logical_lines(text)
        if _FORBIDDEN_PATTERN.search(logical)
    ]


def _workflow_steps(document: object) -> Iterator[Mapping[str, object]]:
    """Every step of every job, plus the steps of a composite action."""
    if not isinstance(document, dict):
        return
    jobs = document.get("jobs", {})
    containers = list(jobs.values()) if isinstance(jobs, dict) else []
    runs = document.get("runs")
    if isinstance(runs, dict):
        containers.append(runs)
    for container in containers:
        steps = container.get("steps", []) if isinstance(container, dict) else []
        for step in steps:
            if isinstance(step, dict):
                yield step


def workflow_findings(text: str, label: str) -> list[str]:
    """Each ``run`` step in a workflow or action that installs omnimarket."""
    findings: list[str] = []
    for step in _workflow_steps(yaml.safe_load(text)):
        run = step.get("run")
        if not isinstance(run, str):
            continue
        for _, logical in _logical_lines(run):
            if _INSTALL_VERB.search(logical) and _FORBIDDEN_PATTERN.search(logical):
                findings.append(
                    f"{label}: step {step.get('name', '<unnamed>')!r} installs "
                    f"{FORBIDDEN}: {logical[:100]}"
                )
    return findings


class _AnyScopeImports(ast.NodeVisitor):
    """Collect every omnimarket import and dynamic-import call, any scope."""

    def __init__(self) -> None:
        self.lines: list[int] = []

    def visit_Import(self, node: ast.Import) -> None:
        if any(alias.name.split(".")[0] == FORBIDDEN for alias in node.names):
            self.lines.append(node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and (node.module or "").split(".")[0] == FORBIDDEN:
            self.lines.append(node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0] if node.args else None
        if (
            name in {"import_module", "importorskip", "__import__"}
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.split(".")[0] == FORBIDDEN
        ):
            self.lines.append(node.lineno)
        self.generic_visit(node)


def fixture_findings(source: str, label: str) -> list[str]:
    """``label:line`` for each omnimarket import in a test module, any scope."""
    visitor = _AnyScopeImports()
    visitor.visit(ast.parse(source, filename=label))
    return [f"{label}:{line}: test imports {FORBIDDEN}" for line in visitor.lines]


def _src_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _workflow_files() -> list[Path]:
    github = REPO_ROOT / ".github"
    return [
        github / "workflows" / "ci.yml",
        *sorted(github.glob("actions/**/action.yml")),
    ]


def _test_files() -> list[Path]:
    return sorted((REPO_ROOT / "tests").rglob("*.py"))


# --- the live tree ---------------------------------------------------------


def test_pyproject_declares_no_omnimarket_dependency() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Positive control: the walk reads the real requirement tables.
    assert len(list(_requirements(tomllib.loads(text)))) > 10
    assert pyproject_findings(text) == []


def test_lock_resolves_no_omnimarket() -> None:
    text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    # Positive control: the lock parses into its real package list.
    assert len(tomllib.loads(text)["package"]) > 50
    assert lock_findings(text) == []


def test_src_imports_no_omnimarket_at_import_time() -> None:
    files = _src_files()
    # Positive control: the scan reaches the package's modules.
    assert len(files) > 500
    findings = [
        finding
        for path in files
        for finding in source_findings(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))
        )
    ]
    assert findings == [], (
        "omnibase_infra must not depend on omnimarket (OMN-19998); move the code "
        "to omnimarket or invert the dependency:\n" + "\n".join(findings)
    )


def test_runtime_dockerfile_names_no_omnimarket() -> None:
    path = REPO_ROOT / "docker" / "Dockerfile.runtime"
    text = path.read_text(encoding="utf-8")
    # Positive control: the walk reads real instruction lines.
    assert len(list(_logical_lines(text))) > 100
    assert dockerfile_findings(text, "docker/Dockerfile.runtime") == []


def test_ci_install_steps_install_no_omnimarket() -> None:
    files = _workflow_files()
    # Positive control: the scan reaches the real jobs, steps and sibling installs.
    assert len(files) > 5
    assert (
        sum(len(list(_workflow_steps(yaml.safe_load(p.read_text())))) for p in files)
        > 300
    )
    assert "uv pip install --no-deps -e ../omnibase_compat" in files[0].read_text(
        encoding="utf-8"
    )
    findings = [
        finding
        for path in files
        for finding in workflow_findings(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))
        )
    ]
    assert findings == [], (
        "omnibase_infra CI must not install omnimarket (OMN-21279); run the job "
        "from omnimarket or invert the dependency:\n" + "\n".join(findings)
    )


def test_test_fixtures_import_no_omnimarket() -> None:
    files = _test_files()
    # Positive control: the scan reaches the real test modules.
    assert len(files) > 1000
    findings = [
        finding
        for path in files
        for finding in fixture_findings(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))
        )
    ]
    assert findings == [], (
        "omnibase_infra tests must not import omnimarket (OMN-21279); move the "
        "test to omnimarket:\n" + "\n".join(findings)
    )


# --- negative controls: each checker fires on a real dependency ------------

_CLEAN_PYPROJECT = """
[build-system]
requires = ["hatchling"]
[project]
name = "omnibase-infra"
dependencies = ["omnibase-core>=0.47,<0.48"]
"""


@pytest.mark.parametrize(
    "fragment",
    [
        '[project]\nname = "x"\ndependencies = ["omnimarket>=0.1"]\n',
        '[project]\nname = "x"\ndependencies = ["OmniMarket[extra] @ git+https://example.invalid/omnimarket"]\n',
        '[project]\nname = "x"\n[project.optional-dependencies]\nlab = ["omnimarket"]\n',
        '[dependency-groups]\ndev = ["pytest", "omnimarket==1.0"]\n',
        '[tool.uv]\noverride-dependencies = ["omnimarket==1.0"]\n',
        '[tool.uv]\nconstraint-dependencies = ["omnimarket<2"]\n',
        '[tool.uv]\ndev-dependencies = ["omnimarket"]\n',
        '[tool.uv.sources]\nomnimarket = { git = "https://example.invalid/omnimarket" }\n',
        '[build-system]\nrequires = ["omnimarket"]\n',
        '[project]\nname = "x"\n[project.scripts]\nonex-x = "omnimarket.cli:main"\n',
        '[project]\nname = "x"\n[project.entry-points."onex.nodes"]\nn = "omnimarket.nodes.node_x"\n',
    ],
)
def test_pyproject_checker_fires_on_omnimarket(fragment: str) -> None:
    assert pyproject_findings(fragment) != []


def test_pyproject_checker_passes_a_clean_manifest() -> None:
    assert pyproject_findings(_CLEAN_PYPROJECT) == []
    # A package whose name only starts with the forbidden one is not it.
    assert (
        pyproject_findings('[project]\nname = "x"\ndependencies = ["omnimarketing"]\n')
        == []
    )


def test_lock_checker_fires_on_omnimarket() -> None:
    locked = (
        '[[package]]\nname = "omnimarket"\nversion = "1.0"\n'
        '[[package]]\nname = "omnibase-infra"\nversion = "1.0"\n'
        'dependencies = [{ name = "omnimarket" }]\n'
    )
    assert lock_findings(locked) == [
        "uv.lock: package omnimarket is locked",
        "uv.lock: package omnibase-infra depends on omnimarket",
    ]
    assert lock_findings('[[package]]\nname = "omnibase-core"\nversion = "1.0"\n') == []


@pytest.mark.parametrize(
    "source",
    [
        "import omnimarket\n",
        "import omnimarket.nodes as nodes\n",
        "from omnimarket.nodes import node_x\n",
        "try:\n    import omnimarket\nexcept ImportError:\n    pass\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from omnimarket import x\n",
        "class C:\n    from omnimarket import x\n",
    ],
)
def test_source_checker_fires_on_import_time_import(source: str) -> None:
    assert source_findings(source, "m.py") != []


def test_source_checker_ignores_function_scope_and_lookalikes() -> None:
    assert source_findings("def f():\n    import omnimarket\n", "m.py") == []
    assert (
        source_findings("import omnimarketing\nfrom . import omnimarket\n", "m.py")
        == []
    )


# --- recorded contact: the real dependent declares the allowed direction ---


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn19998/omnimarket_dev_pyproject_requirements.json"
)
def test_recorded_omnimarket_manifest_depends_on_infra(
    recorded_response: dict[str, object],
) -> None:
    """omnimarket's own dev manifest names omnibase-infra; the reverse is refused.

    The recording is the ``[project] dependencies`` list of omnimarket's
    ``pyproject.toml`` on dev, read from its canonical clone. The checker parses
    that real requirement list, finds the market-on-infra edge, and fires once
    the same requirement is pointed the other way.
    """
    assert recorded_response["project_name"] == FORBIDDEN
    deps = recorded_response["project_dependencies"]
    assert isinstance(deps, list)
    infra = [req for req in deps if _requirement_name(req) == "omnibase-infra"]
    assert len(infra) == 1
    manifest = '[project]\nname = "omnibase-infra"\ndependencies = {}\n'
    assert pyproject_findings(manifest.format(json.dumps(deps))) == []
    reverse = FORBIDDEN + infra[0][len("omnibase-infra") :]
    assert pyproject_findings(manifest.format(json.dumps([*deps, reverse]))) == [
        f"project.dependencies: {reverse!r}"
    ]


# --- negative controls for the image, CI and test-fixture surfaces ---------

_CLEAN_DOCKERFILE = """\
FROM python:3.12-slim AS builder
# omnimarket named in a comment is history, not an install.
RUN uv-with-retry pip install --no-deps \\
    "https://example.invalid/omnibase_compat.tar.gz"
"""


@pytest.mark.parametrize(
    "line",
    [
        "ARG OMNIMARKET_REF=dev\n",
        "RUN uv pip install --no-deps omnimarket\n",
        'RUN uv-with-retry pip install --no-deps \\\n    "git+https://github.com/OmniNode-ai/omnimarket.git@dev#egg=omnimarket"\n',
        "COPY --from=builder /app/.venv/lib/python3.12/site-packages/omnimarket/configs/x.yaml /app/x.yaml\n",
        "RUN python -c 'from omnimarket.nodes import node_x'\n",
        'RUN uv pip install "omnimarket @ file:///workspace/sibling-repos/omnimarket"\n',
    ],
)
def test_dockerfile_checker_fires_on_omnimarket(line: str) -> None:
    assert dockerfile_findings(_CLEAN_DOCKERFILE + line, "Dockerfile.runtime") != []


def test_dockerfile_checker_passes_a_clean_image_and_comments() -> None:
    # Positive control: the clean fixture parses into real instruction lines.
    assert len(list(_logical_lines(_CLEAN_DOCKERFILE))) == 2
    assert dockerfile_findings(_CLEAN_DOCKERFILE, "Dockerfile.runtime") == []


def _workflow(run: str) -> str:
    body = "\n".join(f"          {line}" for line in run.splitlines())
    return (
        "name: x\non: push\njobs:\n  j:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: install\n        run: |\n" + body + "\n"
    )


@pytest.mark.parametrize(
    "run",
    [
        "uv pip install --no-deps omnimarket",
        'uv pip install --quiet --no-deps "omnimarket==0.4.310"',
        "pip install -e ../omnimarket",
        "uv pip install --no-deps -e ../omnibase_compat -e ../omnimarket -e ../omnimemory",
        'uv pip install --python "$PY" --no-deps \\\n  "omnimarket @ git+https://example.invalid/omnimarket@abc"',
        "uv sync --project omnimarket",
        "uv add OmniMarket",
    ],
)
def test_workflow_checker_fires_on_omnimarket_install(run: str) -> None:
    assert workflow_findings(_workflow(run), "wf.yml") != []


def test_workflow_checker_fires_in_a_composite_action() -> None:
    action = (
        "name: a\nruns:\n  using: composite\n  steps:\n"
        "    - shell: bash\n      run: uv pip install omnimarket\n"
    )
    assert workflow_findings(action, "action.yml") != []


def test_workflow_checker_ignores_non_install_mentions() -> None:
    clean = _workflow("uv pip install --no-deps -e ../omnibase_compat")
    assert workflow_findings(clean, "wf.yml") == []
    # Positive control: the same step shape does fire once omnimarket is added.
    assert workflow_findings(clean + "          uv pip install omnimarket\n", "wf.yml")
    mention = _workflow("# uv pip install omnimarket\necho omnimarket is not installed")
    assert workflow_findings(mention, "wf.yml") == []


@pytest.mark.parametrize(
    "source",
    [
        "import omnimarket\n",
        "def test_x():\n    import omnimarket\n",
        "def test_x():\n    from omnimarket.nodes import node_x\n",
        'import importlib\nm = importlib.import_module("omnimarket")\n',
        'import pytest\npytest.importorskip("omnimarket.nodes")\n',
        'm = __import__("omnimarket")\n',
    ],
)
def test_fixture_checker_fires_on_omnimarket_import(source: str) -> None:
    assert fixture_findings(source, "test_x.py") != []


def test_fixture_checker_ignores_lookalikes() -> None:
    assert (
        fixture_findings(
            "import omnimarketing\nfrom . import omnimarket\n"
            'import importlib\nimportlib.import_module("omnibase_core")\n',
            "test_x.py",
        )
        == []
    )
