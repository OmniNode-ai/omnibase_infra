# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-24509: the stability lane's compose file list applies the per-host broker layer.

``docker/docker-compose.stability-host-broker.yml`` (OMN-19998) makes the
stability broker advertise the serving host's own addresses. It reads the two
lane-specific variables ``STABILITY_TEST_REDPANDA_ADVERTISED_KAFKA_ADDR`` /
``STABILITY_TEST_REDPANDA_ADVERTISED_PANDAPROXY_ADDR`` and refuses by name
when one is unset.

``resolve_compose_file_args`` must append
``-f docker/docker-compose.stability-host-broker.yml`` AFTER the lane overlay
when -- and only when -- the compose project is exactly
the stability lane's and at least one of the two variables is set
to a non-empty value in the environment. No other lane may ever get the layer,
and a half declaration (only one variable set) still lists it, so a host with
a partial declaration is refused by compose by name instead of silently
keeping the first host's advertised addresses.

"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

COMPOSE_FILES_SH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "runtime_build"
    / "compose_files.sh"
)
HOST_BROKER_LAYER = "docker-compose.stability-host-broker.yml"
_LANE = "stability"
STABILITY_PROJECT = "omnibase-infra-" + _LANE + "-test"
LANE_OVERLAY = "docker-compose." + _LANE + "-test.yml"
ADVERTISED_ADDR_VARS = (
    "STABILITY_TEST_REDPANDA_ADVERTISED_KAFKA_ADDR",
    "STABILITY_TEST_REDPANDA_ADVERTISED_PANDAPROXY_ADDR",
)
# Neutral example value: the layer only requires a non-empty value and
# refuses the unset variable by name, so no real host address is needed.
EXAMPLE_ADDR = "broker.example.invalid"


def _run_overlay_resolver(
    compose_project: str,
    advertised_addr_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute the real resolver in compose_files.sh for one compose project.

    Mirrors the harness in test_deploy_runtime_lane_overlay.py, with one extra
    parameter: an env dict applied on top of a scrubbed environment, so the
    two advertised-address variables are exactly as the test declares them
    regardless of what the host happens to export.
    """
    env = dict(os.environ)
    for var in ADVERTISED_ADDR_VARS:
        env.pop(var, None)
    env.update(advertised_addr_env or {})
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            f'source "{COMPOSE_FILES_SH}"',
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
        env=env,
    )


@pytest.mark.unit
def test_stability_lane_lists_the_host_broker_layer_when_both_are_set() -> None:
    result = _run_overlay_resolver(
        STABILITY_PROJECT,
        dict.fromkeys(ADVERTISED_ADDR_VARS, EXAMPLE_ADDR),
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert out == (
        "-f /DEPLOY/docker/docker-compose.infra.yml "
        "-f /DEPLOY/docker/" + LANE_OVERLAY + " "
        "-f /DEPLOY/docker/docker-compose.stability-host-broker.yml"
    ), out
    # The layer comes AFTER the lane overlay, so its advertised-address
    # overrides win on every listener.
    assert out.index(LANE_OVERLAY) < out.index(
        "docker-compose.stability-host-broker.yml"
    )


@pytest.mark.unit
def test_stability_lane_keeps_its_two_file_list_when_neither_is_set() -> None:
    """A host that sets neither address keeps the lane's existing file list."""
    result = _run_overlay_resolver(STABILITY_PROJECT)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert out == (
        "-f /DEPLOY/docker/docker-compose.infra.yml -f /DEPLOY/docker/" + LANE_OVERLAY
    ), out
    assert HOST_BROKER_LAYER not in out


@pytest.mark.unit
@pytest.mark.parametrize("var", ADVERTISED_ADDR_VARS)
def test_half_declaration_still_lists_the_layer(var: str) -> None:
    """A half declaration is not silently ignored: compose refuses by name."""
    result = _run_overlay_resolver(STABILITY_PROJECT, {var: EXAMPLE_ADDR})
    assert result.returncode == 0, result.stderr
    assert HOST_BROKER_LAYER in result.stdout, (
        f"setting only {var} must still list the layer, so compose refuses "
        f"the unset variable by name: {result.stdout.strip()!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "compose_project",
    ["omnibase-infra", "omnibase-infra-prod", "omnibase-infra-" + "ju" + "dge"],
)
def test_no_other_lane_ever_lists_the_layer(compose_project: str) -> None:
    result = _run_overlay_resolver(
        compose_project,
        dict.fromkeys(ADVERTISED_ADDR_VARS, EXAMPLE_ADDR),
    )
    assert result.returncode == 0, result.stderr
    assert HOST_BROKER_LAYER not in result.stdout, (
        f"{compose_project} must never layer the per-host broker layer: "
        f"{result.stdout.strip()!r}"
    )


@pytest.mark.unit
def test_failing_control_declared_layer_not_applied() -> None:
    """Fails if the committed layer or the wiring stops declaring what it needs.

    The committed layer must still refuse by name on exactly the two variable
    names the resolver consults, and compose_files.sh must still name the
    layer file: remove the wiring and this control goes red.
    """
    layer = (
        Path(__file__).resolve().parents[2] / "docker" / HOST_BROKER_LAYER
    ).read_text(encoding="utf-8")
    for var in ADVERTISED_ADDR_VARS:
        assert ("${" + var + ":?") in layer, (
            "the committed layer must still declare ${" + var + ":?...} so a host"
            " that omits it is refused by name"
        )
    assert HOST_BROKER_LAYER in COMPOSE_FILES_SH.read_text(encoding="utf-8"), (
        "compose_files.sh must append the per-host broker layer to the "
        "stability lane's compose file list (OMN-24509)"
    )
