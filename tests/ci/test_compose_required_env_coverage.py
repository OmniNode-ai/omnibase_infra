# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI guard: every `:?`-required env var in docker-compose.infra.yml must be
supplied by EVERY compose-render test fixture that layers that file.

Catches: PRs that add a new service with a required `:?` env var to compose
without updating the render fixtures, which previously caused cascading
failures in #886, #890, #895 (OMN-5240 root cause analysis).

OMN-15263 — why this gate was generalized. It used to check ONE fixture
(`tests/integration/docker/test_docker_integration.py`). OMN-15173 (#2495)
added a `:?` fail-fast for `DEV_REDPANDA_ADVERTISE_HOST` and updated exactly
that fixture, so this gate stayed green while the other three render fixtures —
which build their env hermetically (`env=` replaced, not inherited) — started
failing `docker compose config` on every hosted CI runner: 12 failures in
`Tests (Split 2/15)`, red `CI Summary`, on every PR whose selector escalated to
the full suite. A gate that covers one of four call sites is why that landed
green.

Enforced here:

1. **Coverage** — for each registered render fixture, every `:?`-required var in
   the base infra compose file is supplied by that fixture's env dict, by one of
   the `--env-file` files it passes to compose, or is explicitly declared as
   intentionally unset with a reason.
2. **Fail-closed discovery** — a `*compose_render*.py` module under
   `tests/integration/` that is not registered below fails this gate, so the
   next render fixture cannot silently escape coverage the way three just did.

This module is deliberately static (AST + text parsing, no `docker` invocation)
so it fires on hosts without Docker — the exact hosts where the 12 render tests
`skip` and where this regression was invisible locally.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.infra.yml"

# Any fixture whose source references this file layers the base infra compose
# file and therefore inherits its `:?` vars — compose interpolates every input
# file before merge, regardless of `--profile`.
BASE_COMPOSE_FILENAME = "docker-compose.infra.yml"

# Fail-closed discovery: every module matching this glob must be registered in
# RENDER_FIXTURES below.
RENDER_FIXTURE_GLOB = "tests/integration/**/*compose_render*.py"


class RenderFixture(NamedTuple):
    """A test module that renders compose and must supply its required env."""

    path: str
    # Module-level dict constants holding the render env. Extracted by AST, so
    # a fixture must expose them at module scope, not inside a test body.
    env_dicts: tuple[str, ...]
    # (var, reason) pairs. Escape hatch for a fixture that must NOT set a var
    # because proving the unset behaviour is the point of that fixture.
    intentionally_unset: tuple[tuple[str, str], ...] = ()


RENDER_FIXTURES: tuple[RenderFixture, ...] = (
    RenderFixture(
        path="tests/integration/docker/test_docker_integration.py",
        env_dicts=("COMPOSE_CONFIG_RENDER_ENV",),
    ),
    RenderFixture(
        path="tests/integration/infra/test_dev_runtime_compose_render.py",
        env_dicts=("BASE_REQUIRED_ENV",),
        intentionally_unset=(
            (
                "DEV_REDPANDA_ADVERTISE_HOST",
                (
                    "OMN-15173 counter-test: this module's "
                    "test_dev_redpanda_advertise_host_fails_fast_when_unset proves "
                    "the render FAILS when the var is unset. Adding it to "
                    "BASE_REQUIRED_ENV would make that test vacuous and resurrect "
                    "the silent localhost-advertise regression."
                ),
            ),
        ),
    ),
    RenderFixture(
        path="tests/integration/infra/test_prod_runtime_compose_render.py",
        env_dicts=("COMPOSE_RENDER_ENV",),
    ),
    RenderFixture(
        path="tests/integration/infra/test_judge_compose_render.py",
        env_dicts=("LAYERED_RENDER_DUMMY_ENV",),
    ),
    RenderFixture(
        path="tests/integration/infra/test_stability_test_runtime_compose_render.py",
        env_dicts=("COMPOSE_RENDER_ENV",),
    ),
    # OMN-17150 collaborator lane. `env_dicts=()` is deliberate, not an
    # omission: this fixture supplies NOTHING from a module-level Python dict.
    # Its render env comes entirely from the two committed `--env-file` files it
    # passes to compose (docker/runtime-policy.env + docker/lakshman.env.example),
    # which `extract_env_file_vars` already reads out of the fixture source. That
    # is the point of the fixture — it proves a FRESH CLONE renders, so a dummy
    # dict here would weaken it by supplying values the lane owner will not have.
    #
    # It also renders a STANDALONE lane file (docker/docker-compose.lakshman.yml,
    # never layered on docker-compose.infra.yml), so
    # test_all_required_compose_vars_in_fixture skips it by design. Registration
    # is still mandatory: the OMN-15263 fail-closed discovery check exists
    # precisely so a new render module cannot sit outside this gate's field of
    # view, standalone or not.
    RenderFixture(
        path="tests/integration/infra/test_lakshman_compose_render.py",
        env_dicts=(),
    ),
)


def extract_required_compose_vars(compose_path: Path) -> set[str]:
    """Return all variable names that use `:?` fail-fast syntax in a compose file."""
    text = compose_path.read_text(encoding="utf-8")
    return set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):\?", text))


def extract_module_env_vars(source: str, dict_names: tuple[str, ...]) -> set[str]:
    """Return the string keys of the named module-level dict constants.

    AST-based: only literal `str` keys are collected, and only from assignments
    at module scope, so a dict nested inside a test body is never silently
    counted as coverage.
    """
    tree = ast.parse(source)
    wanted = set(dict_names)
    found: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = {
                    key.value
                    for key in value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
    missing_dicts = wanted - found.keys()
    assert not missing_dicts, (
        "Registered env dict(s) not found as module-level dict literals: "
        + ", ".join(sorted(missing_dicts))
        + ". Either the constant was renamed or moved into a function body, or "
        "RENDER_FIXTURES in this file is stale."
    )
    return set().union(*found.values()) if found else set()


def extract_env_file_vars(source: str) -> set[str]:
    """Return vars supplied by every `--env-file` the fixture passes to compose.

    A fixture may legitimately omit a var from its Python dict when it loads an
    env file that defines it (judge loads docker/judge.env.example; every lane
    loads docker/runtime-policy.env).
    """
    provided: set[str] = set()
    for rel_path in re.findall(r'"--env-file",\s*"([^"]+)"', source):
        env_path = REPO_ROOT / rel_path
        assert env_path.is_file(), (
            f"A render fixture passes --env-file {rel_path!r}, which does not "
            "exist relative to the repo root."
        )
        provided |= set(
            re.findall(
                r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=",
                env_path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )
    return provided


def _fixture_source(fixture: RenderFixture) -> str:
    path = REPO_ROOT / fixture.path
    assert path.is_file(), (
        f"Registered render fixture {fixture.path} does not exist. If it was "
        "renamed or deleted, update RENDER_FIXTURES in this file."
    )
    return path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_every_compose_render_fixture_is_registered() -> None:
    """A new compose-render module must be registered, or this gate goes RED.

    Fail-closed: the OMN-15263 breakage landed green precisely because three
    render fixtures were outside this gate's field of view.
    """
    discovered = {
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.glob(RENDER_FIXTURE_GLOB)
        if path.name.startswith("test_")
    }
    registered = {fixture.path for fixture in RENDER_FIXTURES}
    unregistered = discovered - registered
    assert not unregistered, (
        "These compose-render test modules are not registered in "
        "RENDER_FIXTURES in tests/ci/test_compose_required_env_coverage.py:\n"
        + "\n".join(f"  - {path}" for path in sorted(unregistered))
        + "\n\nFix: add a RenderFixture entry naming the module-level env dict "
        "constant(s) it renders with, so this coverage gate can see it."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "fixture", RENDER_FIXTURES, ids=[fixture.path for fixture in RENDER_FIXTURES]
)
def test_all_required_compose_vars_in_fixture(fixture: RenderFixture) -> None:
    """Every `:?`-required var in compose must be supplied by each render fixture.

    This is the CI twin for the contract: 'if you add a `:?` var to compose, you
    must also add it to every compose-render fixture'. Fails on the PR that
    introduces the gap, not on every unrelated PR afterwards.
    """
    source = _fixture_source(fixture)
    if BASE_COMPOSE_FILENAME not in source:
        pytest.skip(
            f"{fixture.path} does not layer {BASE_COMPOSE_FILENAME}; its `:?` "
            "vars come from another compose file."
        )

    required = extract_required_compose_vars(COMPOSE_FILE)
    provided = (
        extract_module_env_vars(source, fixture.env_dicts)
        | extract_env_file_vars(source)
        | {var for var, _reason in fixture.intentionally_unset}
    )
    missing = required - provided
    assert not missing, (
        "These `:?`-required env vars are in docker/docker-compose.infra.yml but "
        f"NOT supplied by the render fixture {fixture.path}:\n"
        + "\n".join(f"  - {var}" for var in sorted(missing))
        + "\n\nFix: add each missing var to "
        + " / ".join(fixture.env_dicts)
        + f" in {fixture.path} (a render-only dummy value is fine). Every "
        "compose-render fixture layering the base infra file interpolates every "
        "`:?` var in it, regardless of --profile."
    )


@pytest.mark.unit
def test_intentionally_unset_vars_are_justified() -> None:
    """An omission escape hatch must name a real var and carry a reason.

    Keeps the hatch from decaying into a silent allowlist: the var must still be
    `:?`-required in compose (stale exemptions go RED) and must actually appear
    in the fixture that omits it, so the deliberate omission is visible there.
    """
    required = extract_required_compose_vars(COMPOSE_FILE)
    for fixture in RENDER_FIXTURES:
        source = _fixture_source(fixture)
        for var, reason in fixture.intentionally_unset:
            assert var in required, (
                f"{fixture.path} declares {var} intentionally unset, but {var} is "
                "no longer `:?`-required in docker/docker-compose.infra.yml. "
                "Remove the stale exemption."
            )
            assert len(reason.strip()) >= 40, (
                f"{fixture.path}: exemption for {var} needs a real reason, not "
                f"{reason!r}."
            )
            assert var in source, (
                f"{fixture.path} is exempted from supplying {var} but never "
                "mentions it. An exemption is only valid for a fixture that "
                "deliberately exercises the unset case."
            )


@pytest.mark.unit
def test_dev_advertise_host_keeps_fail_fast_form() -> None:
    """OMN-15173 counter-test: the dev advertise host must never regain a default.

    OMN-15263's fix supplies the var in the render fixtures. The wrong fix —
    giving compose a `:-localhost` default again — would also turn every render
    green, while silently restoring the off-host regression OMN-15173 removed.
    This test makes that shortcut RED, on every host, with or without Docker.
    """
    text = COMPOSE_FILE.read_text(encoding="utf-8")

    defaulted = re.findall(r"\$\{DEV_REDPANDA_ADVERTISE_HOST:-[^}]*\}", text)
    assert not defaulted, (
        "DEV_REDPANDA_ADVERTISE_HOST regained a silent default in "
        "docker/docker-compose.infra.yml: "
        + ", ".join(defaulted)
        + ". OMN-15173: an unset advertise host must fail the render loudly, not "
        "render an address no off-host client can reach. Supply the var in the "
        "render fixtures instead."
    )
    assert "DEV_REDPANDA_ADVERTISE_HOST" in extract_required_compose_vars(
        COMPOSE_FILE
    ), (
        "DEV_REDPANDA_ADVERTISE_HOST is no longer `:?`-required in "
        "docker/docker-compose.infra.yml (OMN-15173)."
    )
