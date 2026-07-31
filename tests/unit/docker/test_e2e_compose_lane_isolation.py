# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15565: the e2e compose stack must never alias a live .201 runtime lane.

`docker/docker-compose.e2e.yml` is an ephemeral CI stack that is brought up and
then torn down with `down -v --remove-orphans` on a host that also runs the live
lab/dev, prod, judge and stability-test lanes. Its compose project name used to
default to `omnibase-infra` -- the lab lane's own project -- and it reused that
lane's `container_name`s, volume names and network name. A bare invocation
therefore resolved *into* the live lane, and the teardown deleted the lane's
containers and its Postgres/Redpanda/Valkey data volumes. It ran nightly, on
cron, for at least eight consecutive nights.

The collision was invisible at review time because the project name was a
*default*, not a literal, and because compose projects, container names, volume
names and network names are four separate global namespaces on the Docker
daemon -- renaming only one of them does not isolate anything.

This module is the ratchet. The protected namespaces are derived from
`deploy/lane-census/lane-manifest.yaml` (the versioned desired-state census) and
from the lane compose files it names, so a lane added later is covered without
editing this file. Nothing about lane identity is hardcoded here.

Related but insufficient: `tests/test_compose_profile_teardown_policy.py`
explicitly classifies `docker-compose.e2e.yml` as "NOT a runtime lane" and
asserts a bare `docker compose -f docker/docker-compose.e2e.yml down -v` is
clean. That is correct *given* the e2e file is isolated -- these tests are what
make that premise true.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
E2E_COMPOSE_PATH = REPO_ROOT / "docker" / "docker-compose.e2e.yml"
LANE_MANIFEST_PATH = REPO_ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
DOCKER_DIR = REPO_ROOT / "docker"

# The dev/lab lane's manifest entry points at docker-compose.generated.yml, which
# is produced at deploy time and is not tracked. docker-compose.infra.yml is the
# base that generated file is rendered from, so it carries the lab lane's real
# container/volume/network names and must be part of the protected corpus.
_DEV_LANE_BASE_COMPOSE = DOCKER_DIR / "docker-compose.infra.yml"


# ---------------------------------------------------------------------------
# Shell-style ${VAR:-default} resolution
# ---------------------------------------------------------------------------

# Matches ${NAME}, ${NAME:-default} and ${NAME:?message}. `[^{}]*` deliberately
# refuses nested braces so that nesting is resolved inner-first by _resolve_env
# looping to a fixed point.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::(-|\?)([^{}]*))?\}")


def _substitute_once(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        operator, value = match.group(2), match.group(3)
        # `:-` supplies a default; `:?` is a fail-fast with a message, which
        # resolves to nothing when the variable is unset.
        return value if operator == "-" and value is not None else ""

    return _VAR_RE.sub(_repl, text)


def _resolve_env(text: str) -> str:
    """Resolve every ``${VAR:-default}`` as if no environment variable were set.

    This is what a bare ``docker compose -f <file> ...`` on a clean shell sees --
    the exact invocation shape that produced the OMN-15565 lane destruction.
    """
    for _ in range(10):
        resolved = _substitute_once(text)
        if resolved == text:
            return resolved
        text = resolved
    raise AssertionError("variable substitution did not reach a fixed point")


# ---------------------------------------------------------------------------
# e2e compose facts (as a bare invocation resolves them)
# ---------------------------------------------------------------------------


def _e2e_compose() -> dict[str, Any]:
    doc = yaml.safe_load(_resolve_env(E2E_COMPOSE_PATH.read_text(encoding="utf-8")))
    assert isinstance(doc, dict), "e2e compose file did not parse into a mapping"
    return doc


def _e2e_project() -> str:
    project = _e2e_compose().get("name")
    assert isinstance(project, str) and project, (
        "docker/docker-compose.e2e.yml must declare a top-level `name:` (compose "
        "project). Without it compose derives the project from the directory "
        "name, which is not reviewable."
    )
    return project


def _e2e_container_names() -> dict[str, str]:
    services = _e2e_compose().get("services") or {}
    return {
        name: body["container_name"]
        for name, body in services.items()
        if isinstance(body, dict) and body.get("container_name")
    }


def _e2e_named_resources(key: str) -> dict[str, str]:
    """Resolved ``name:`` values under a top-level ``volumes:``/``networks:`` block."""
    block = _e2e_compose().get(key) or {}
    return {
        alias: body["name"]
        for alias, body in block.items()
        if isinstance(body, dict) and body.get("name")
    }


# ---------------------------------------------------------------------------
# Protected lane namespaces (policy-derived, never hardcoded)
# ---------------------------------------------------------------------------


def _lane_manifest() -> dict[str, Any]:
    manifest = yaml.safe_load(LANE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), "lane-manifest.yaml did not parse into a mapping"
    return manifest


def _lanes() -> dict[str, Any]:
    lanes = _lane_manifest().get("lanes") or {}
    assert lanes, "lane-manifest.yaml declared no lanes — protected set would be empty"
    return lanes


def _lane_projects() -> set[str]:
    return {
        lane["compose_project"]
        for lane in _lanes().values()
        if lane.get("compose_project")
    }


def _lane_networks() -> set[str]:
    return {lane["network"] for lane in _lanes().values() if lane.get("network")}


def _lane_service_names() -> set[str]:
    return {
        service["name"]
        for lane in _lanes().values()
        for service in lane.get("services") or []
        if service.get("name")
    }


def _lane_compose_files() -> list[Path]:
    paths = [_DEV_LANE_BASE_COMPOSE]
    for lane in _lanes().values():
        compose_file = lane.get("compose_file")
        if not compose_file:
            continue
        path = REPO_ROOT / compose_file
        # Generated lane files are rendered at deploy time and may be absent.
        if path.exists() and path != E2E_COMPOSE_PATH:
            paths.append(path)
    return sorted(set(paths))


_CONTAINER_NAME_RE = re.compile(r"^\s+container_name:\s*(\S+)\s*$", re.MULTILINE)


def _top_level_block_names(text: str, key: str) -> set[str]:
    """Resolved ``name:`` values inside a top-level ``volumes:``/``networks:`` block."""
    block_re = re.compile(rf"(?ms)^{key}:\n(.*?)(?=^\S|\Z)")
    names: set[str] = set()
    for block in block_re.finditer(text):
        names.update(re.findall(r"^\s+name:\s*(\S+)\s*$", block.group(1), re.MULTILINE))
    return names


def _lane_owned_names() -> set[str]:
    """Every container/volume/network identifier a live lane owns on the daemon."""
    owned = _lane_service_names() | _lane_networks()
    for path in _lane_compose_files():
        text = _resolve_env(path.read_text(encoding="utf-8"))
        owned.update(_CONTAINER_NAME_RE.findall(text))
        owned.update(_top_level_block_names(text, "volumes"))
        owned.update(_top_level_block_names(text, "networks"))
    return {name for name in owned if name}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_e2e_project_default_is_not_a_censused_lane() -> None:
    """A bare e2e invocation must not resolve into a live lane's compose project."""
    project = _e2e_project()
    lane_projects = _lane_projects()
    assert project not in lane_projects, (
        f"docker/docker-compose.e2e.yml resolves to compose project {project!r}, "
        f"which is a live lane in deploy/lane-census/lane-manifest.yaml "
        f"({sorted(lane_projects)}). A bare `docker compose -f "
        f"docker/docker-compose.e2e.yml down -v --remove-orphans` would then "
        f"delete that lane's containers and data volumes (OMN-15565)."
    )


def test_e2e_container_names_do_not_collide_with_lane_containers() -> None:
    """`container_name` is a global daemon namespace — it is not scoped by project."""
    lane_owned = _lane_owned_names()
    collisions = {
        service: name
        for service, name in _e2e_container_names().items()
        if name in lane_owned
    }
    assert not collisions, (
        f"e2e services {sorted(collisions)} default to container names owned by a "
        f"live lane: {sorted(collisions.values())}. Container names are global to "
        f"the Docker daemon, so `up` recreates the lane's container even from a "
        f"different compose project (OMN-15565)."
    )


def test_e2e_volume_and_network_names_do_not_collide_with_lanes() -> None:
    """Volume/network names are global too — a shared volume means nightly data loss."""
    lane_owned = _lane_owned_names()
    e2e_named = {**_e2e_named_resources("volumes"), **_e2e_named_resources("networks")}
    collisions = {
        alias: name for alias, name in e2e_named.items() if name in lane_owned
    }
    assert not collisions, (
        f"e2e volume/network aliases {sorted(collisions)} default to names owned by "
        f"a live lane: {sorted(collisions.values())}. `down -v` on the e2e stack "
        f"then deletes the lane's data (OMN-15565)."
    )


def test_e2e_resource_names_stay_inside_the_e2e_namespace() -> None:
    """Ratchet: every e2e-owned name is prefixed by the e2e project name.

    Disjointness from *today's* lanes is necessary but not sufficient — a lane
    added tomorrow could take a name the e2e stack already squats. Containment
    makes the e2e namespace self-describing so the lane census can never grow
    into it accidentally.
    """
    project = _e2e_project()
    named = {
        **_e2e_container_names(),
        **_e2e_named_resources("volumes"),
        **_e2e_named_resources("networks"),
    }
    escapees = {
        alias: name
        for alias, name in named.items()
        if not name.startswith(f"{project}-")
    }
    assert not escapees, (
        f"e2e resource defaults are outside the {project!r} namespace: {escapees}. "
        f"Every container/volume/network default in docker-compose.e2e.yml must be "
        f"prefixed with the compose project name."
    )


def test_no_lane_lives_inside_the_e2e_namespace() -> None:
    """The inverse containment check: no censused lane may sit under the e2e prefix."""
    project = _e2e_project()
    intruders = sorted(
        name
        for name in _lane_projects() | _lane_owned_names()
        if name == project or name.startswith(f"{project}-")
    )
    assert not intruders, (
        f"lane-owned names {intruders} fall inside the e2e namespace {project!r}. "
        f"Rename the e2e project rather than allowing the two namespaces to overlap."
    )


# ---------------------------------------------------------------------------
# Workflow invocation scan
# ---------------------------------------------------------------------------

_COMPOSE_RE = re.compile(r"docker[\s-]compose\b", re.IGNORECASE)
_PROJECT_FLAG_RE = re.compile(r"(?:-p|--project-name)[=\s]+\S+")
# The e2e file passed as an actual compose FILE argument. Requiring the flag is
# what separates a real invocation from prose that merely names the file --
# `docker-compose.e2e.yml` itself contains the substring "docker-compose", so a
# looser match flags every input description that mentions it.
_COMPOSE_FILE_FLAG_RE = re.compile(r"(?:-f|--file)[=\s]+\S*docker-compose\.e2e\.yml\b")
_CONTINUATION_RE = re.compile(r"\\\n\s*")


def _invocation_lines(text: str) -> list[str]:
    """Comment-stripped, continuation-joined lines that invoke the e2e compose file."""
    joined = _CONTINUATION_RE.sub(" ", text)
    lines = []
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if _COMPOSE_RE.search(stripped) and _COMPOSE_FILE_FLAG_RE.search(stripped):
            lines.append(stripped)
    return lines


def test_every_workflow_e2e_compose_invocation_names_its_project() -> None:
    """No workflow may let the e2e stack fall back to the compose file's default."""
    violations: list[str] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(
        WORKFLOWS_DIR.glob("*.yaml")
    ):
        for line in _invocation_lines(path.read_text(encoding="utf-8")):
            if not _PROJECT_FLAG_RE.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {line}")
    assert not violations, (
        "workflow invocations of docker/docker-compose.e2e.yml without an explicit "
        "`-p <project>`:\n  " + "\n  ".join(violations) + "\n"
        "An unnamed project resolves to the compose file default. That is how "
        "nightly-integration.yml deleted the lab lane every night (OMN-15565)."
    )


def _nightly_step(name: str) -> dict[str, Any]:
    workflow = yaml.safe_load(
        (WORKFLOWS_DIR / "nightly-integration.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["integration-tests"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def _run_teardown_with_stubs(
    project: str, tmp_path: Path
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Execute the workflow teardown shell with non-mutating uv/docker stubs."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "docker-calls.log"

    uv_stub = fake_bin / "uv"
    uv_stub.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" != run ] || [ "${2:-}" != python ]; then exit 97; fi\n'
        "shift 2\n"
        'exec "$TEST_PYTHON" "$@"\n',
        encoding="utf-8",
    )
    uv_stub.chmod(0o700)

    docker_stub = fake_bin / "docker"
    docker_stub.write_text(
        '#!/bin/sh\nprintf \'docker %s\\n\' "$*" >> "$DOCKER_CALL_LOG"\n',
        encoding="utf-8",
    )
    docker_stub.chmod(0o700)

    env = os.environ | {
        "DOCKER_CALL_LOG": str(call_log),
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_RUN_ID": "314159",
        "OMNIBASE_INFRA_COMPOSE_PROJECT": project,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "TEST_PYTHON": sys.executable,
    }
    result = subprocess.run(
        ["bash", "-c", _nightly_step("Tear down e2e stack")["run"]],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        call_log.read_text(encoding="utf-8").splitlines() if call_log.exists() else []
    )
    return result, calls


@pytest.mark.parametrize(
    "project",
    ["", *sorted(_lane_projects()), "some-other-uncensused-project"],
)
def test_nightly_teardown_rejects_every_non_run_project_without_docker(
    project: str, tmp_path: Path
) -> None:
    """Empty, protected, and arbitrary projects must all fail before Docker."""
    result, calls = _run_teardown_with_stubs(project, tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert calls == []


def test_nightly_teardown_downs_exact_run_project_once(tmp_path: Path) -> None:
    """Only the immutable project for this run may reach the exact scoped down."""
    # The canonical derive recipe uses `echo | tr -cs`, so echo's newline is
    # normalized to the trailing hyphen. Teardown must reproduce it byte-for-byte.
    expected = "omnibase-infra-e2e-314159-2-"
    result, calls = _run_teardown_with_stubs(expected, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [
        f"docker compose -p {expected} -f docker/docker-compose.e2e.yml "
        "down -v --remove-orphans"
    ]


def test_nightly_teardown_rederives_run_identity_before_manifest_and_down() -> None:
    """Ratchet the always-run teardown's immutable identity and lane guards."""
    derive_run = _nightly_step("Derive isolated e2e namespace")["run"]
    teardown = _nightly_step("Tear down e2e stack")
    run = teardown["run"]

    assert teardown["if"] == "always()"

    protected_projects = _lane_projects()
    assert protected_projects, "lane manifest must provide the protected project set"
    assert all(
        f'"{project}"' not in run and f"'{project}'" not in run
        for project in protected_projects
    ), (
        "teardown must derive protected projects from lane-manifest.yaml, not hardcode "
        "today's lane names"
    )

    id_recipe = 'e2e_id="e2e-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"'
    normalization = (
        "e2e_id=\"$(echo \"$e2e_id\" | tr '[:upper:]' '[:lower:]' "
        "| tr -cs 'a-z0-9_-' '-')\""
    )
    assert id_recipe in derive_run and id_recipe in run
    assert normalization in derive_run and normalization in run
    assert 'project="omnibase-infra-${e2e_id}"' in derive_run

    expected_project = 'expected_project="omnibase-infra-${e2e_id}"'
    identity_guard = 'if [[ "$project" != "$expected_project" ]]; then'
    manifest_load = 'open("deploy/lane-census/lane-manifest.yaml", encoding="utf-8")'
    membership_guard = "if project in protected:"
    rejection = "sys.exit("
    destructive_down = (
        'docker compose -p "$project" -f docker/docker-compose.e2e.yml '
        "down -v --remove-orphans"
    )
    for fragment in (
        expected_project,
        identity_guard,
        manifest_load,
        membership_guard,
        rejection,
        destructive_down,
    ):
        assert fragment in run, f"nightly teardown is missing: {fragment}"
    assert (
        run.index(expected_project)
        < run.index(identity_guard)
        < run.index(manifest_load)
        < run.index(membership_guard)
        < run.index(rejection, run.index(membership_guard))
        < run.index(destructive_down)
    ), "identity and manifest rejection must execute before destructive compose down"


# ---------------------------------------------------------------------------
# Scanner self-tests — prove the checks are RED against the pre-fix shape
# ---------------------------------------------------------------------------


def test_resolver_reproduces_the_pre_fix_project_name() -> None:
    """The pre-fix line really did resolve to the lab lane's project."""
    assert _resolve_env("name: ${OMNIBASE_INFRA_COMPOSE_PROJECT:-omnibase-infra}") == (
        "name: omnibase-infra"
    )
    assert _resolve_env("${A:?required}") == ""
    assert _resolve_env("${OUTER:-${INNER:-fallback}}") == "fallback"


def test_pre_fix_project_name_would_be_rejected() -> None:
    """`omnibase-infra` is a censused lane, so the pre-fix default must be a failure."""
    assert "omnibase-infra" in _lane_projects(), (
        "the dev/lab lane must be censused for this guard to have teeth"
    )


def test_workflow_scanner_flags_a_project_less_invocation() -> None:
    """Self-test: the scanner catches the exact pre-fix invocation shapes."""
    bare_up = "docker compose -f docker/docker-compose.e2e.yml up -d \\\n  postgres"
    assert _invocation_lines(bare_up)
    assert not _PROJECT_FLAG_RE.search(_invocation_lines(bare_up)[0])

    bare_down = (
        "run: docker compose -f docker/docker-compose.e2e.yml down -v --remove-orphans"
    )
    assert _invocation_lines(bare_down)
    assert not _PROJECT_FLAG_RE.search(_invocation_lines(bare_down)[0])

    scoped = 'docker compose -p "$PROJECT" -f docker/docker-compose.e2e.yml down -v'
    assert _PROJECT_FLAG_RE.search(_invocation_lines(scoped)[0])

    commented = "# docker compose -f docker/docker-compose.e2e.yml down -v"
    assert not _invocation_lines(commented)

    prose = (
        'description: "External postgres port published by '
        'docker/docker-compose.e2e.yml. Keep in sync with POSTGRES_PORT."'
    )
    assert not _invocation_lines(prose), (
        "prose mentioning the file is not an invocation"
    )
