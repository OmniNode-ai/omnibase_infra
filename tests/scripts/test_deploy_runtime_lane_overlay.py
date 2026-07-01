# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""deploy-runtime.sh must layer the lane overlay on every compose invocation.

OMN-13581: deploy-runtime.sh historically passed ONLY ``-f docker-compose.infra.yml``
on every ``docker compose`` call -- including ``warm_broker_topic_provisioning``'s
``up redpanda`` step. The base infra compose hardcodes
``container_name: omnibase-infra-redpanda`` (the DEV name) and the dev network.
Running the warmup against a non-dev compose project (e.g.
``omnibase-infra-stability-test``) therefore made compose try to recreate redpanda
as the DEV-named container, which collided with the live dev broker, got a Docker
hash prefix, and landed in ``created`` -- DESTROYING the lane's own correctly-named
broker. That left the stability lane broker-less for ~3 days.

The fix introduces ``resolve_lane_overlay_filename`` / ``resolve_compose_file_args``
which derive the lane from the compose project suffix and LAYER the matching overlay
(``docker-compose.<lane>.yml``) onto ``docker-compose.infra.yml`` for every non-dev
project, mirroring the authoritative lane->compose-file mapping in
``scripts/deploy-agent/deploy_agent/executor.py`` (_LANE_CONFIGS).

These tests assert:
  - the helper functions exist,
  - the dev / stability-test / prod / judge lane mapping is correct (behavioral,
    by extracting + executing the pure helper functions -- no docker required),
  - an unknown non-dev project fails CLOSED (refuses to run on the bare dev config),
  - NO compose-invoking function passes a bare ``-f .../docker-compose.infra.yml``
    string anymore; every call site routes through ``resolve_compose_file_args``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy-runtime.sh"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """Return the source text of a single top-level bash function ``name()``.

    The helper functions under test are pure (no top-level deps), so they can be
    extracted and executed in isolation -- giving a real behavioral assertion
    without sourcing the whole script (which runs main and needs docker).
    """
    text = _script_text()
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


def _run_overlay_resolver(compose_project: str) -> subprocess.CompletedProcess[str]:
    """Execute the extracted resolver functions for one compose project.

    Stubs ``log_error`` so the fail-closed path does not depend on the script's
    logging helpers, then echoes the resolved ``-f`` token sequence.
    """
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            _extract_function("resolve_lane_overlay_filename"),
            _extract_function("resolve_compose_file_args"),
            "declare -a out",
            f'resolve_compose_file_args out "/DEPLOY" "{compose_project}"',
            'printf "%s\\n" "${out[*]}"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_defines_lane_overlay_resolver_functions() -> None:
    text = _script_text()
    assert re.search(r"^resolve_lane_overlay_filename\s*\(\)", text, re.MULTILINE), (
        "deploy-runtime.sh must define resolve_lane_overlay_filename()"
    )
    assert re.search(r"^resolve_compose_file_args\s*\(\)", text, re.MULTILINE), (
        "deploy-runtime.sh must define resolve_compose_file_args()"
    )


@pytest.mark.unit
def test_dev_project_gets_no_overlay() -> None:
    """The bare dev project runs from infra.yml alone (fixed dev names are correct)."""
    result = _run_overlay_resolver("omnibase-infra")
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert out == "-f /DEPLOY/docker/docker-compose.infra.yml", out
    assert "stability-test" not in out
    assert "prod" not in out


@pytest.mark.unit
@pytest.mark.parametrize(
    ("compose_project", "lane"),
    [
        ("omnibase-infra-stability-test", "stability-test"),
        ("omnibase-infra-prod", "prod"),
        ("omnibase-infra-judge", "judge"),
    ],
)
def test_non_dev_project_layers_matching_overlay(
    compose_project: str, lane: str
) -> None:
    """Every non-dev lane LAYERS its overlay onto infra.yml, in order."""
    result = _run_overlay_resolver(compose_project)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    expected = (
        "-f /DEPLOY/docker/docker-compose.infra.yml "
        f"-f /DEPLOY/docker/docker-compose.{lane}.yml"
    )
    assert out == expected, out
    # infra.yml must come FIRST so the overlay's container_name/project/network win.
    assert out.index("docker-compose.infra.yml") < out.index(
        f"docker-compose.{lane}.yml"
    )


@pytest.mark.unit
def test_unknown_non_dev_project_fails_closed() -> None:
    """An unrecognized non-dev project must abort, not silently run on dev config.

    This is the whole point of the fix: running a non-dev lane on the bare
    infra.yml config recreates the DEV-named redpanda and displaces the lane's
    broker (OMN-13581). An unknown lane must therefore fail CLOSED.
    """
    result = _run_overlay_resolver("omnibase-infra-mystery-lane")
    assert result.returncode != 0, (
        "unknown non-dev compose project must fail closed, got: "
        f"stdout={result.stdout!r} rc={result.returncode}"
    )
    assert "mystery-lane" in result.stderr or "Unknown lane" in result.stderr


@pytest.mark.unit
def test_no_compose_invocation_uses_bare_infra_only() -> None:
    """No active code path may pass a single bare ``-f infra.yml`` to docker compose.

    Every compose call site must route its ``-f`` flags through
    ``resolve_compose_file_args`` so the lane overlay is always layered. A bare
    ``compose_file="${...}/docker-compose.infra.yml"`` local re-introduces the
    broker-displacement bug.
    """
    lines = [
        line
        for line in _script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    # The old anti-pattern: a per-function local pinning the single infra compose
    # file, then passing it as the sole -f flag.
    offenders = [
        (i + 1, line)
        for i, line in enumerate(lines)
        if re.search(r"compose_file=.*docker-compose\.infra\.yml", line)
    ]
    assert offenders == [], (
        "Found bare single-compose-file locals in deploy-runtime.sh; every compose "
        f"call must layer the lane overlay via resolve_compose_file_args: {offenders}"
    )


@pytest.mark.unit
def test_compose_call_sites_route_through_resolver() -> None:
    """Each compose-invoking function must call resolve_compose_file_args.

    Guards against a future function being added that re-pins a bare infra.yml.
    """
    text = _script_text()
    # Functions that issue `docker compose` with `-f`.
    for func in (
        "sanity_check",
        "build_images",
        "warm_broker_topic_provisioning",
        "run_runtime_migration_preflight",
        "restart_services",
    ):
        match = re.search(
            rf"^{func}\s*\(\)\s*\{{.*?\n\}}",
            text,
            re.DOTALL | re.MULTILINE,
        )
        assert match is not None, f"could not find function {func}()"
        body = match.group(0)
        assert "resolve_compose_file_args" in body, (
            f"{func}() must resolve its -f flags via resolve_compose_file_args "
            "(so the lane overlay is layered)"
        )
