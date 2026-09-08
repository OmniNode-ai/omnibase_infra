# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Required dev-lane compose variables are DECLARED, and PREFLIGHTED as a set.

OMN-17530. The dev lane brings up two compose files, not one --
``docker/docker-compose.infra.yml`` plus ``docker/docker-compose.dev-lane.yml``
(``resolve_compose_file_args`` in ``scripts/deploy-runtime.sh``, and
``_LANE_CONFIGS[DEV].compose_files`` in the deploy agent) -- and a ``${VAR:?}``
in either file is a hard requirement of the deploy.

Only the base had a declared manifest. The overlay had none, so:

1. Nothing in the repo could be red when a PR added a required variable to it.
   omnibase_infra#3332 added two. Neither was declared anywhere.
2. ``docker compose config`` reports the FIRST unset required variable and
   stops. So the two arrived one deploy at a time: agent command ece3ec11 died
   on ``ONEX_API_IMAGE`` at 2026-09-08T20:14Z, and command 87d10b56 -- published
   after that one was supplied -- died ~14 minutes later on
   ``ONEX_CLOUD_MIGRATE_IMAGE``. A mechanical enumeration afterwards found the
   real missing count was TEN, not two: eight more walls were queued behind
   those, each worth one more failed command and one more lane window.

Two properties close the class, and this file asserts both:

  (a) DECLARATION -- every hard-required interpolation in the compose files the
      dev lane loads is declared in a committed manifest, exactly. A PR that
      adds a ``${VAR:?}`` without declaring it is red.
  (b) ONE MESSAGE -- the preflight reports ALL missing required variables in a
      single invocation, before compose validation, naming the store file that
      should carry each. An operator sees the whole list at once.

Every assertion here has a positive control: a fixture compose carrying an
undeclared required variable must FAIL the declaration check, and a fixture
whose variables are all set must PASS the preflight. A guard that cannot be made
to fire is not evidence.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_BASE_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.infra.yml"
_DEV_LANE_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
_BASE_MANIFEST = _REPO_ROOT / "docker" / "required-env-vars.manifest.txt"
_DEV_LANE_MANIFEST = _REPO_ROOT / "docker" / "dev-lane-required-env.manifest.txt"

_CHECK_SCRIPT = _REPO_ROOT / "scripts" / "check_required_env_vars.py"
_PREFLIGHT_SCRIPT = _REPO_ROOT / "scripts" / "preflight_required_compose_env.py"
_DEPLOY_RUNTIME = _REPO_ROOT / "scripts" / "deploy-runtime.sh"
_REFRESH_DEV_LANE = _REPO_ROOT / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
_EXECUTOR = _REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "executor.py"

# docker-compose required-var syntax. Deliberately re-declared here rather than
# imported from either script, so a change to a script cannot silently weaken
# the test that governs it.
_REQUIRED_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?")

# The two variables omnibase_infra#3332 added to the dev-lane overlay -- the two
# that surfaced one deploy at a time. Named explicitly: this is the concrete
# regression, and it must pass now that they are declared.
_OMN_3332_VARS = ("ONEX_API_IMAGE", "ONEX_CLOUD_MIGRATE_IMAGE")


def _required_names(path: Path) -> set[str]:
    return set(_REQUIRED_VAR_PATTERN.findall(path.read_text(encoding="utf-8")))


def _declared_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line)
    return names


def _run_check(compose: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_CHECK_SCRIPT),
            "--compose-file",
            str(compose),
            "--manifest-file",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )


def _run_preflight(
    composes: list[Path], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_PREFLIGHT_SCRIPT), "--lane", "dev"]
    for compose in composes:
        cmd.extend(["--compose-file", str(compose)])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
    )


# --------------------------------------------------------------------------
# (a) DECLARATION
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_dev_lane_manifest_exists() -> None:
    """The overlay the dev lane loads has a declared-name manifest at all."""
    assert _DEV_LANE_MANIFEST.is_file(), (
        f"{_DEV_LANE_MANIFEST.relative_to(_REPO_ROOT)} is missing. The dev lane "
        "loads docker-compose.dev-lane.yml alongside the base compose; without "
        "this manifest a required variable added to the overlay is declared "
        "nowhere and can only be discovered by a failed deploy (OMN-17530)."
    )


@pytest.mark.unit
def test_every_dev_lane_required_var_is_declared() -> None:
    """Exact agreement: no undeclared name, no stale declaration."""
    required = _required_names(_DEV_LANE_COMPOSE)
    declared = _declared_names(_DEV_LANE_MANIFEST)

    undeclared = sorted(required - declared)
    stale = sorted(declared - required)

    assert not undeclared, (
        f"{len(undeclared)} required var(s) in docker/docker-compose.dev-lane.yml "
        f"are not declared in {_DEV_LANE_MANIFEST.name}: {undeclared}. Declare "
        "each name in the SAME commit that adds its ${VAR:?} — otherwise the "
        "next deploy is the thing that discovers it (OMN-17530)."
    )
    assert not stale, (
        f"{len(stale)} name(s) declared in {_DEV_LANE_MANIFEST.name} are no longer "
        f"required by the overlay: {stale}. Remove them — an inexact manifest "
        "cannot be trusted as the complete set."
    )
    assert required, "the overlay declares no required vars at all — parse failure?"


@pytest.mark.unit
def test_union_of_dev_lane_compose_files_is_fully_declared() -> None:
    """Both files the dev lane loads are covered, across both manifests."""
    required = _required_names(_BASE_COMPOSE) | _required_names(_DEV_LANE_COMPOSE)
    declared = _declared_names(_BASE_MANIFEST) | _declared_names(_DEV_LANE_MANIFEST)

    undeclared = sorted(required - declared)
    assert not undeclared, (
        "the dev lane loads docker-compose.infra.yml + docker-compose.dev-lane.yml, "
        f"and {len(undeclared)} of their required vars are declared in neither "
        f"manifest: {undeclared}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("var_name", _OMN_3332_VARS)
def test_omn_3332_image_vars_are_declared(var_name: str) -> None:
    """The GREEN half: the two variables that cost two deploys are declared."""
    assert var_name in _required_names(_DEV_LANE_COMPOSE), (
        f"{var_name} is no longer a ${{VAR:?}} in the dev-lane overlay; if it was "
        "deliberately removed, remove it from the manifest and from this test."
    )
    assert var_name in _declared_names(_DEV_LANE_MANIFEST), (
        f"{var_name} is required by the dev-lane overlay but not declared in "
        f"{_DEV_LANE_MANIFEST.name}"
    )


@pytest.mark.unit
def test_image_vars_carry_provenance_comments() -> None:
    """Each image tag name is preceded by a provenance comment, not bare.

    An image tag resolved from an operator store, with no compose default and no
    committed value, is exactly the kind of name a reader will otherwise guess
    at. The previous lane had to derive ONEX_CLOUD_MIGRATE_IMAGE by hashing the
    migration corpus against the deployed clone because the intuitive pick — the
    tag sharing the API image's build stamp — was provably the wrong image.
    """
    lines = _DEV_LANE_MANIFEST.read_text(encoding="utf-8").splitlines()
    for var_name in _OMN_3332_VARS:
        index = lines.index(var_name)
        preceding = [line for line in lines[:index][::-1] if line.strip()]
        assert preceding and preceding[0].lstrip().startswith("#"), (
            f"{var_name} in {_DEV_LANE_MANIFEST.name} has no provenance comment "
            "immediately above it. Image-tag names resolve from a store, not from "
            "a compose default — say where the value comes from (OMN-17530)."
        )


@pytest.mark.unit
def test_declaration_check_fires_on_an_undeclared_var(tmp_path: Path) -> None:
    """RED CONTROL. A fixture overlay with an undeclared required var FAILS."""
    compose = tmp_path / "docker-compose.fixture.yml"
    compose.write_text(
        "services:\n"
        "  fixture:\n"
        "    image: ${FIXTURE_UNDECLARED_IMAGE:?set it}\n"
        "    environment:\n"
        "      DECLARED: ${FIXTURE_DECLARED_NAME:?set it}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "fixture-required-env.manifest.txt"
    manifest.write_text("# fixture\nFIXTURE_DECLARED_NAME\n", encoding="utf-8")

    result = _run_check(compose, manifest)

    assert result.returncode == 1, (
        "the declaration check passed a compose file carrying an undeclared "
        f"required var — it cannot be evidence. stdout={result.stdout!r}"
    )
    assert "FIXTURE_UNDECLARED_IMAGE" in result.stderr
    assert "FIXTURE_DECLARED_NAME" not in result.stderr.split("but not declared")[-1]


@pytest.mark.unit
def test_declaration_check_passes_the_real_dev_lane_pair() -> None:
    """GREEN. The shipped overlay + manifest agree exactly, via the real script."""
    result = _run_check(_DEV_LANE_COMPOSE, _DEV_LANE_MANIFEST)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


@pytest.mark.unit
def test_dev_lane_manifest_declares_names_only() -> None:
    """Names only — never a value, never a secret."""
    for raw_line in _DEV_LANE_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" not in line, (
            f"{_DEV_LANE_MANIFEST.name} carries what looks like a VALUE: {line!r}. "
            "This file declares names only."
        )
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", line), (
            f"unexpected line in {_DEV_LANE_MANIFEST.name}: {line!r}"
        )


# --------------------------------------------------------------------------
# (b) ONE MESSAGE
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_preflight_reports_every_missing_var_in_one_message(
    tmp_path: Path,
) -> None:
    """The property the two dead deploy commands did not have.

    One invocation, an environment carrying none of them, and ALL of the names
    in the single message. If this ever regressed to first-wins, the operator
    would again pay one failed deploy per variable.
    """
    compose_a = tmp_path / "docker-compose.base.yml"
    compose_a.write_text(
        "services:\n"
        "  a:\n"
        "    image: ${FIXTURE_ALPHA:?set it}\n"
        "    environment:\n"
        "      B: ${FIXTURE_BRAVO:?set it}\n",
        encoding="utf-8",
    )
    compose_b = tmp_path / "docker-compose.overlay.yml"
    compose_b.write_text(
        "services:\n"
        "  b:\n"
        "    image: ${FIXTURE_CHARLIE:?set it}\n"
        "    environment:\n"
        "      D: ${FIXTURE_DELTA:?set it}\n",
        encoding="utf-8",
    )

    result = _run_preflight([compose_a, compose_b], env={"PATH": "/usr/bin:/bin"})

    assert result.returncode == 1, (
        f"preflight passed with four unset required vars. stdout={result.stdout!r}"
    )
    message = result.stderr
    for name in ("FIXTURE_ALPHA", "FIXTURE_BRAVO", "FIXTURE_CHARLIE", "FIXTURE_DELTA"):
        assert name in message, (
            f"{name} is missing from the preflight's single message — the whole "
            "point is that the operator sees the SET, not the first wall. "
            f"message={message!r}"
        )
    # The overlay's names must be attributed to the overlay, so a reader knows
    # which file to declare them in.
    assert compose_b.name in message
    assert compose_a.name in message


@pytest.mark.unit
def test_preflight_names_the_store_file(tmp_path: Path) -> None:
    """A missing name is useless without saying where the value should live."""
    compose = tmp_path / "docker-compose.fixture.yml"
    compose.write_text(
        "services:\n  a:\n    image: ${FIXTURE_ECHO:?set it}\n", encoding="utf-8"
    )
    env = {"PATH": "/usr/bin:/bin", "OMNIBASE_OPERATOR_ENV_FILE": "/fixture/store.env"}

    result = _run_preflight([compose], env=env)

    assert result.returncode == 1
    assert "store file" in result.stderr
    assert "/fixture/store.env" in result.stderr, (
        "the preflight did not name the operator env store the value should come "
        f"from. message={result.stderr!r}"
    )


@pytest.mark.unit
def test_preflight_routes_contract_rendered_names_to_the_policy_file(
    tmp_path: Path,
) -> None:
    """A contract-rendered name must not be sent to the operator store.

    Telling an operator to hand-add a name that a contract render owns is how a
    hand-edit diverges from its contract.
    """
    compose = tmp_path / "docker-compose.fixture.yml"
    compose.write_text(
        "services:\n"
        "  a:\n"
        "    environment:\n"
        "      P: ${FIXTURE_POLICY_NAME:?set it}\n"
        "      O: ${FIXTURE_OPERATOR_NAME:?set it}\n",
        encoding="utf-8",
    )
    policy = tmp_path / "runtime-policy.env"
    policy.write_text("FIXTURE_POLICY_NAME=rendered\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_PREFLIGHT_SCRIPT),
            "--lane",
            "dev",
            "--runtime-policy-env",
            str(policy),
            "--compose-file",
            str(compose),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "OMNIBASE_OPERATOR_ENV_FILE": "/fixture/store.env",
        },
        check=False,
    )

    assert result.returncode == 1
    blocks = result.stderr.split("FIXTURE_POLICY_NAME")[1].split("FIXTURE_OPERATOR")[0]
    assert "runtime-policy.env" in blocks, (
        f"a contract-rendered name was not routed to the policy file: {result.stderr!r}"
    )


@pytest.mark.unit
def test_preflight_passes_when_every_var_is_set(tmp_path: Path) -> None:
    """POSITIVE CONTROL for the zero: a satisfied fixture exits 0."""
    compose = tmp_path / "docker-compose.fixture.yml"
    compose.write_text(
        "services:\n  a:\n    image: ${FIXTURE_FOXTROT:?set it}\n", encoding="utf-8"
    )
    result = _run_preflight(
        [compose], env={"PATH": "/usr/bin:/bin", "FIXTURE_FOXTROT": "value"}
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "FIXTURE_FOXTROT" not in result.stderr


@pytest.mark.unit
def test_preflight_treats_an_empty_value_as_missing(tmp_path: Path) -> None:
    """``${VAR:?}`` aborts on empty as well as unset; so must the preflight."""
    compose = tmp_path / "docker-compose.fixture.yml"
    compose.write_text(
        "services:\n  a:\n    image: ${FIXTURE_GOLF:?set it}\n", encoding="utf-8"
    )
    result = _run_preflight(
        [compose], env={"PATH": "/usr/bin:/bin", "FIXTURE_GOLF": "   "}
    )
    assert result.returncode == 1, (
        "an empty value passed the preflight but would abort compose — the two "
        "must agree on what 'set' means."
    )
    assert "FIXTURE_GOLF" in result.stderr


@pytest.mark.unit
def test_preflight_never_prints_a_value(tmp_path: Path) -> None:
    """Names and store paths only. A preflight that echoes values is a leak."""
    compose = tmp_path / "docker-compose.fixture.yml"
    compose.write_text(
        "services:\n"
        "  a:\n"
        "    environment:\n"
        "      S: ${FIXTURE_SET_NAME:?set it}\n"
        "      M: ${FIXTURE_MISSING_NAME:?set it}\n",
        encoding="utf-8",
    )
    secret = "fixture-value-that-must-not-be-echoed"
    result = _run_preflight(
        [compose], env={"PATH": "/usr/bin:/bin", "FIXTURE_SET_NAME": secret}
    )
    assert result.returncode == 1
    assert secret not in result.stderr
    assert secret not in result.stdout


# --------------------------------------------------------------------------
# WIRING — the guard must be ON the paths that died
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_preflight_script_exists_and_is_stdlib_only() -> None:
    """No third-party import: a missing venv must not silence the list."""
    assert _PREFLIGHT_SCRIPT.is_file()
    source = _PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    forbidden = ("import yaml", "import pydantic", "from pydantic", "import click")
    for token in forbidden:
        assert token not in source, (
            f"{_PREFLIGHT_SCRIPT.name} imports {token!r}. It runs on a deploy host "
            "before any project venv is guaranteed; keep it stdlib-only."
        )


@pytest.mark.unit
def test_refresh_dev_lane_runs_the_preflight() -> None:
    """The sanctioned operator entry point calls it before it probes anything."""
    source = _REFRESH_DEV_LANE.read_text(encoding="utf-8")
    assert "preflight_required_compose_env.py" in source, (
        "scripts/runtime_build/refresh_dev_lane.sh no longer runs the "
        "required-compose-env preflight. Removing it restores the "
        "one-wall-per-deploy failure OMN-17530 closed."
    )
    assert "docker-compose.dev-lane.yml" in source, (
        "the preflight must be handed the OVERLAY as well as the base — the "
        "overlay is where both variables that killed a deploy live."
    )
    call_index = source.index("if ! run_preflight_required_env; then")
    lock_index = source.index("lane_lock_acquire")
    assert call_index < lock_index, (
        "the preflight must run BEFORE the lane lock is acquired. A refresh that "
        "cannot possibly validate must not hold the dev-lane window while it "
        "finds that out."
    )


@pytest.mark.unit
def test_deploy_agent_preflights_before_lane_validation() -> None:
    """The agent's compose_gen calls it AHEAD of `docker compose config`.

    Ordering is the whole property: after validation it would report nothing new,
    because validation would already have died on the first name.
    """
    source = _EXECUTOR.read_text(encoding="utf-8")
    assert "_preflight_required_compose_env" in source
    call_index = source.index("self._preflight_required_compose_env(")
    validate_index = source.index("validate_cmd = [")
    assert call_index < validate_index, (
        "the deploy agent runs the required-env preflight AFTER compose "
        "validation, which makes it useless — validation dies on the first "
        "missing name first (OMN-17530)."
    )


@pytest.mark.unit
def test_dev_lane_compose_file_set_matches_both_deploy_paths() -> None:
    """The two files this test governs are the two files the lane really loads.

    If a deploy path started loading a third compose file, this test's coverage
    would silently be a subset of the real requirement set.
    """
    resolver = _DEPLOY_RUNTIME.read_text(encoding="utf-8")
    assert "docker-compose.infra.yml" in resolver
    assert "docker-compose.dev-lane.yml" in resolver

    executor = _EXECUTOR.read_text(encoding="utf-8")
    assert "_DEV_LANE_OVERLAY = " in executor
    assert "docker-compose.dev-lane.yml" in executor
    dev_config = executor.split("EnumRuntimeLane.DEV: ModelLaneConfig(", 1)[1][:400]
    assert "compose_files=(COMPOSE_FILE, _DEV_LANE_OVERLAY)" in dev_config, (
        "the deploy agent's DEV lane no longer loads exactly "
        "(base, dev-lane overlay). Update this test and the manifest set together."
    )
