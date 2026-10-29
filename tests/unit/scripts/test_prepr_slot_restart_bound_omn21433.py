# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21433: a pre-PR slot service that cannot start stops instead of looping.

Measured on the lab host on 2026-10-28. The slot's projection writers inherit
``restart: unless-stopped`` from the dev-lane overlay. Three of them exit on
start in a slot because the standalone binding they open expects the dev
principal ``omninode_runtime`` and the slot's DSN names ``omninode_runtime_<slot>``.
The slot's savings writer had restarted 9047 times, and the log of that one
container held 172978 refusals ``too many connections for role`` on the slot's
intelligence role: every restart opens its pools and its schema stamp on the dev
lane's shared Postgres, against a slot role that carries a budget of 6. The budget did its job (the refusals were the slot's own), but a
loop with no end is a standing load on the shared server and a slot that never
reports its failure.

A bounded policy turns the loop into a finding: three restarts, then the container
stays exited, ``docker ps`` shows it, and the slot's health probe fails on it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY = REPO_ROOT / "docker" / "docker-compose.prepr.yml"

SLOT_SERVICES = (
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
    "projection-api",
    "tenant-projection-writer",
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
    "onex-api",
)


# Use the same chain and profiles as prepr_verify_lane.sh. Compose itself must
# apply interpolation, !override, !reset, inheritance and normalization.
COMPOSE_CHAIN = (
    REPO_ROOT / "docker" / "docker-compose.infra.yml",
    REPO_ROOT / "docker" / "docker-compose.dev-lane.yml",
    OVERLAY,
)


@pytest.fixture(scope="module")
def rendered_services(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    scratch = tmp_path_factory.mktemp("prepr-restart-render")
    runtime_policy = REPO_ROOT / "docker" / "runtime-policy.env"
    policy_keys = set(
        re.findall(r"^([A-Z_][A-Z_0-9]*)=", runtime_policy.read_text(), re.M)
    )
    # Render only: no daemon, credentials, private lane manifest or running
    # containers are needed. Supply synthetic values for required inputs outside
    # the checked-in runtime policy; leave the actual Compose YAML untouched.
    required = {
        match.group(1)
        for path in (
            *COMPOSE_CHAIN,
            REPO_ROOT / "docker" / "docker-compose.broker-profile.yml",
        )
        for match in re.finditer(r"\$\{([A-Z_][A-Z_0-9]*):?\?", path.read_text())
    }
    project = os.environ.get("COMPOSE_PROJECT_NAME", "prepr-restart-render")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(scratch),
        "COMPOSE_PROJECT_NAME": project,
        **dict.fromkeys(required - policy_keys, "render-only"),
        **{key: "48080" for key in required - policy_keys if key.endswith("_PORT")},
        "ONEX_PREPR_SLOT": "1",
        "ONEX_DB_SLOT": "prepr1",
        "PREPR_VALKEY_DB_INDEX": "11",
        "ONEX_LANE_OVERLAY_DIR": str(scratch),
        "ONEX_PREPR_TENANT_STATE_DIR": str(scratch),
        "ONEX_LAB_TENANT_STATE_DIR": str(scratch),
    }
    command = [
        "docker",
        "compose",
        "--env-file",
        str(runtime_policy),
        "-p",
        project,
    ]
    for path in COMPOSE_CHAIN:
        command.extend(("-f", str(path)))
    command.extend(
        (
            "--profile",
            "prepr",
            "--profile",
            "prepr-gateway",
            "config",
            "--format",
            "json",
        )
    )
    done = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    services: dict[str, Any] = json.loads(done.stdout)["services"]
    return services


@pytest.mark.parametrize("name", SLOT_SERVICES)
def test_every_slot_service_has_a_bounded_restart_policy(
    name: str, rendered_services: dict[str, Any]
) -> None:
    policy = rendered_services[name].get("restart")
    assert policy == "on-failure:3", (
        f"{name} renders restart policy {policy!r} from the infra, dev-lane and "
        f"prepr chain. The slot must stop after three failed restarts instead of "
        f"looping on the shared Postgres (OMN-21433)."
    )
