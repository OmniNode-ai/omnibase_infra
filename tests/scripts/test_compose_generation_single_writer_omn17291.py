# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The tracked compose file has exactly one writer, and the render's env is declared.

OMN-17291. Two defects, one root cause -- ``docker/docker-compose.infra.yml`` was
written by two authorities that did not agree on its contents.

1. ``scripts/deploy-agent/deploy_agent/executor.py`` ran, on every deploy,
   ``... catalog.cli generate core runtime --output docker/docker-compose.infra.yml``
   (OMN-8430), overwriting the TRACKED file in place. The catalog CLI's own
   declared output is ``docker/docker-compose.generated.yml`` -- a build-artifact
   path that ``.gitignore`` already ignores -- so the ``--output`` override was
   the deviation, not the design. The render carries 12 services the tracked file
   does not and 31 required-var names against the tracked file's 50; a lab clone
   that had just been reset to origin/dev therefore showed a ~2900-line
   uncommitted delta six seconds later, on every deploy, forever.

2. The render requires nine ``${VAR:?}`` names that resolve from no committed
   source. The deploy agent supplied four of them from an inline sentinel dict
   inside ``_compose_env`` and the rest from whatever the lab host happened to
   hold; ``scripts/deploy-runtime.sh`` and
   ``scripts/runtime_build/refresh_stability_lane.sh`` supply none of them. So
   once the render had replaced the tracked file, every deploy-runtime.sh run --
   the dev warm refresh and the stability refresh both -- failed compose
   validation on the first of them and auto-restored. The two lanes that are
   the proof surface for beta work were re-broken by every deploy-agent run.

The four sentinels are deleted rather than relocated. Measured 2026-09-08 at
00a821b1: those names appear in ZERO committed compose files, so after the
single-writer fix no compose command the agent issues references them; and they
were never sufficient anyway -- with all four supplied, the render still fails
interpolation on ONEX_TENANT_DB_URL both on the lab host and in the live agent's
own process environment. A sentinel that fixes nothing and is needed by nothing
is the silent default this ticket exists to remove.

The properties asserted here are the two halves of the fix: the tracked file is
written only by git, and the render's env contract is a committed, exact
declaration rather than the union of one host's environment and one Python
literal.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TRACKED_COMPOSE = "docker/docker-compose.infra.yml"
_GENERATED_COMPOSE = "docker/docker-compose.generated.yml"
_RUNTIME_POLICY_ENV = _REPO_ROOT / "docker" / "runtime-policy.env"
_TRACKED_MANIFEST = _REPO_ROOT / "docker" / "required-env-vars.manifest.txt"
_GENERATED_MANIFEST = (
    _REPO_ROOT / "docker" / "generated-compose-required-env.manifest.txt"
)

# docker-compose required-var syntax: ${VARNAME:?message}. Same pattern as
# scripts/check_required_env_vars.py -- deliberately not imported, so a change
# to that script cannot silently weaken this test.
_REQUIRED_VAR_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]+):\?")

# The four names the deploy agent used to inject a sentinel value for.
_SENTINEL_NAMES = frozenset(
    {
        "CI_CALLBACK_TOKEN",
        "LINEAR_WEBHOOK_SECRET",
        "WAITLIST_NOTIFIER_SLACK_BOT_TOKEN",
        "WAITLIST_NOTIFIER_SLACK_CHANNEL_ID",
    }
)

# The sentinel value they carried. Named here because a test asserting a literal
# is absent has to say which literal; this file is a test, not a deploy path, so
# the scan below does not read it.
_SENTINEL_VALUE = "deploy-agent-compose-parse-only"


def _declared_names(path: Path) -> set[str]:
    """Return the NAMES declared in a names-only manifest or a dotenv file."""
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split("=", 1)[0].strip() if "=" in line else line)
    return names


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _tracked_files() -> list[str]:
    """Every path git tracks in this repo."""
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


# The catalog names whose VALUE the render is allowed to read from the render
# host, and which therefore have to be neutralised before this test measures the
# ${VAR:?} NAME set. Today that is the source_env of every optional directory
# bind mount declared under docker/catalog/services/. Set to a value the
# generator accepts, so the test exercises the configured branch rather than the
# unconfigured one -- the point of OMN-17291 is that both branches render the
# same names, and the companion unit tests in
# tests/unit/infra/test_catalog_generator.py assert exactly that.
_HOST_SENSITIVE_RENDER_INPUTS = ("CODING_AGENT_CLAUDE_CREDS_HOST_DIR",)


def _generate_catalog_compose(output: Path) -> None:
    """Run the catalog CLI generator into *output*, in a subprocess.

    The subprocess gets a CONTROLLED environment for every name the render is
    allowed to read. Inheriting them unfiltered is what made this test's verdict
    a property of the machine: a workstation carrying ambient coding-agent
    credentials rendered one required-var name that a lab host without them did
    not, so the test passed on one and failed on the other for the same commit
    (OMN-17291; measured on lab host h105 at 2026-09-08). The generator no
    longer branches on those values, and pinning them here keeps the test
    measuring the DECLARATION even if a future optional input reintroduces a
    host-sensitive branch.
    """
    env = dict(os.environ)
    for name in _HOST_SENSITIVE_RENDER_INPUTS:
        env[name] = str(_REPO_ROOT)
    # The generator reads os.environ for image tags and similar render-time
    # values, and the CLI also loads ~/.omnibase/.env and the repo .env. Those
    # supply VALUES; with the names above pinned, none of them changes the
    # ${VAR:?} NAME set, which is what this test measures.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "generate",
            "core",
            "runtime",
            "--output",
            str(output),
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        "catalog CLI generate failed:\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )


@pytest.mark.unit
def test_no_code_path_generates_into_the_tracked_compose_file() -> None:
    """The tracked compose file is written by git and by nothing else.

    A generator whose --output is a tracked path makes that file have two
    writers, and the loser is whichever one ran first. Scanned across every
    tracked file rather than only the deploy agent, because the next caller to
    reach for `--output` will be a different script.
    """
    offenders: list[str] = []
    for rel_path in _tracked_files():
        if rel_path == str(Path(__file__).relative_to(_REPO_ROOT)):
            continue
        path = _REPO_ROOT / rel_path
        if path.suffix not in {".py", ".sh", ".yml", ".yaml", ".toml"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "catalog.cli" not in content and "catalog import cli" not in content:
            continue
        if "/tests/" in rel_path or rel_path.startswith("tests/"):
            continue
        lines = content.splitlines()
        for lineno, line in enumerate(lines, start=1):
            if "--output" not in line:
                continue
            # The path may sit on the same line (shell) or on one of the next
            # few (a Python argv list), so judge a small window, not one line.
            window = "\n".join(lines[lineno - 1 : lineno + 3])
            if "docker-compose.infra.yml" in window or "COMPOSE_FILE," in window:
                offenders.append(f"{rel_path}:{lineno}: {line.strip()}")

    assert not offenders, (
        "A catalog-generate call targets the TRACKED compose file with --output. "
        "The generator's output is a build artifact "
        f"({_GENERATED_COMPOSE}, already gitignored); {_TRACKED_COMPOSE} is "
        "changed by PR only:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_generated_compose_path_is_gitignored_and_untracked() -> None:
    """The build artifact must never become a tracked file by accident."""
    assert _GENERATED_COMPOSE not in _tracked_files(), (
        f"{_GENERATED_COMPOSE} is a build artifact and must not be tracked"
    )
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "check-ignore", "-q", _GENERATED_COMPOSE],
        check=False,
    )
    assert result.returncode == 0, (
        f"{_GENERATED_COMPOSE} is not gitignored, so a deploy that renders it "
        "leaves the deploy-source clone dirty -- the OMN-17291 symptom."
    )


@pytest.mark.unit
def test_no_deploy_path_injects_a_parse_only_sentinel() -> None:
    """No deploy path invents a value to make a compose file parse.

    The four sentinels this replaces were visible to exactly one deploy path,
    which is how the deploy agent came to be able to validate a compose render
    that scripts/deploy-runtime.sh could not. Test fixtures are out of scope: a
    fixture supplying render-only env cannot desynchronise one deploy path from
    another.
    """
    offenders: list[str] = []
    for rel_path in _tracked_files():
        path = _REPO_ROOT / rel_path
        if path.suffix not in {".py", ".sh"}:
            continue
        if not rel_path.startswith(("scripts/", "src/")):
            continue
        if "/tests/" in rel_path or rel_path.startswith("tests/"):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if _SENTINEL_VALUE in line and not line.lstrip().startswith("#"):
                offenders.append(f"{rel_path}:{lineno}")

    assert not offenders, (
        "A deploy path injects a parse-only sentinel into a compose environment. "
        "The render that needed it is no longer written to a path any deploy "
        "command reads, and no committed compose file references those names at "
        "all:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_sentinel_names_appear_in_no_committed_compose_file() -> None:
    """The evidence the sentinels were safe to delete, kept as a live check.

    If one of these names ever becomes required by a committed compose file it
    needs a real source, and this test failing is the signal to find one --
    never to reintroduce a placeholder.
    """
    compose_files = sorted((_REPO_ROOT / "docker").glob("docker-compose*.yml"))
    assert compose_files, "positive control: found no compose files to scan"

    found: list[str] = []
    for compose_file in compose_files:
        referenced = set(
            _REQUIRED_VAR_PATTERN.findall(compose_file.read_text(encoding="utf-8"))
        )
        found.extend(
            f"{compose_file.name}: {name}"
            for name in sorted(_SENTINEL_NAMES & referenced)
        )

    assert not found, (
        "A committed compose file now requires a name OMN-17291 deleted a "
        "sentinel for. It needs a real source, not a placeholder:\n  "
        + "\n  ".join(found)
    )


@pytest.mark.unit
def test_tracked_compose_manifest_is_exact() -> None:
    """Regression guard: the tracked file's manifest stays a pure, exact diff.

    OMN-17291 must not be 'fixed' by widening docker/required-env-vars.manifest.txt
    to also cover the render's names -- that would make the tracked file's own
    manifest inexact, which is the property OMN-15537 built it for.
    """
    tracked_required = set(
        _REQUIRED_VAR_PATTERN.findall(
            (_REPO_ROOT / _TRACKED_COMPOSE).read_text(encoding="utf-8")
        )
    )
    declared = _declared_names(_TRACKED_MANIFEST)
    assert tracked_required == declared, (
        "docker/required-env-vars.manifest.txt drifted from "
        f"{_TRACKED_COMPOSE}:\n"
        f"  required but undeclared: {sorted(tracked_required - declared)}\n"
        f"  declared but not required: {sorted(declared - tracked_required)}"
    )


@pytest.mark.unit
def test_generated_compose_required_env_is_exactly_declared(tmp_path: Path) -> None:
    """Every ${VAR:?} the catalog render needs is declared in a committed manifest.

    Runs the real generator. Before OMN-17291 nine of these names were declared
    nowhere in the repo, so the only way to learn that a render could not be
    validated by deploy-runtime.sh was to watch a lane deploy fail.
    """
    assert _GENERATED_MANIFEST.is_file(), f"missing {_GENERATED_MANIFEST}"

    output = tmp_path / "docker-compose.generated.yml"
    _generate_catalog_compose(output)

    required = set(_REQUIRED_VAR_PATTERN.findall(output.read_text(encoding="utf-8")))
    assert required, "positive control: the render declared no required vars at all"

    declared = _declared_names(_GENERATED_MANIFEST)
    assert required == declared, (
        f"{_GENERATED_MANIFEST.name} drifted from the live catalog render:\n"
        f"  required but undeclared: {sorted(required - declared)}\n"
        f"  declared but not required: {sorted(declared - required)}\n"
        "Regenerate with: uv run python -m omnibase_infra.docker.catalog.cli "
        "generate core runtime --output docker/docker-compose.generated.yml"
    )
