# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prepare isolated containers to read back the prod recipe's overlay binds.

Run from the repository root with ``--output /tmp/probe/compose.json``, then
``docker compose -p <isolated-project> -f /tmp/probe/compose.json run
--no-deps <service>`` for omninode-runtime, runtime-effects and runtime-worker.
Only the rendered overlay mount and its path are copied into the probe recipe;
no production container names, ports, networks, data volumes or credentials are
used. A public Python image reads the file and checks that the bind is read-only.
This exercises Docker's actual bind behavior, not full runtime startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path

import pytest

LOGGER = logging.getLogger(__name__)
SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")
PROJECT = "prod-overlay-probe"
COMPOSE_FILES = (
    "docker/docker-compose.infra.yml",
    "docker/docker-compose.prod.yml",
)
REQUIRED_VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]*):?\?[^}]*\}")
READBACK = """
import hashlib
import os
from pathlib import Path

path = Path(os.environ['BIFROST_LANE_OVERLAY_PATH'])
assert hashlib.sha256(path.read_bytes()).hexdigest() == os.environ['OVERLAY_SHA256']
mounts = [line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
assert any(mount[4] == str(path) and 'ro' in mount[5].split(',') for mount in mounts), 'overlay bind is not read-only'
print(os.environ['PROBE_SERVICE'] + ': overlay bytes match; bind is read-only')
"""


def prepare_probe(root: Path, output: Path) -> None:
    """Render the real recipe and retain its overlay binds in isolated probes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay = output.parent / f"{PROJECT}.bifrost.yaml"
    overlay.write_bytes(
        (root / "docker/lane-overlays/local.bifrost.example.yaml").read_bytes()
    )
    policy = root / "docker/runtime-policy.env"
    policy_keys = {
        line.partition("=")[0]
        for line in policy.read_text().splitlines()
        if line and not line.startswith("#")
    }
    # Compose still interpolates services the probe never starts. Supply neutral
    # values for required inputs, preserving the checked-in runtime policy.
    required = {
        key
        for name in COMPOSE_FILES
        for key in REQUIRED_VARIABLE.findall((root / name).read_text())
    }
    render_env = {
        key: "10000" if key.endswith("_PORT") else "probe-only"
        for key in required - policy_keys
    }
    render_env.update(
        PATH=os.environ["PATH"],
        HOME=os.environ.get("HOME", ""),
        COMPOSE_PROJECT_NAME=PROJECT,
        ONEX_LANE_OVERLAY_DIR=str(output.parent),
    )
    command = ["docker", "compose", "--env-file", str(policy)]
    for name in COMPOSE_FILES:
        command.extend(("-f", str(root / name)))
    command.extend(("--profile", "runtime", "config", "--format", "json"))
    rendered = subprocess.run(
        command,
        cwd=root,
        env=render_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    services = json.loads(rendered.stdout)["services"]
    probes = {}
    for name in SERVICES:
        service = services[name]
        target = service["environment"]["BIFROST_LANE_OVERLAY_PATH"]
        mounts = [
            mount
            for mount in service.get("volumes", [])
            if mount.get("target") == target
        ]
        # Keep the actual mounts, including an empty list on the broken recipe:
        # the container read then fails with FileNotFoundError.
        probes[name] = {
            "image": "python:3.13-alpine",
            "network_mode": "none",
            "read_only": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "environment": {
                "BIFROST_LANE_OVERLAY_PATH": target,
                "OVERLAY_SHA256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                "PROBE_SERVICE": name,
            },
            "volumes": mounts,
            "entrypoint": ["python", "-c", READBACK],
        }
    output.write_text(json.dumps({"services": probes}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    prepare_probe(args.root.resolve(), args.output.resolve())


@pytest.mark.integration
def test_prod_overlay_binds_are_readable_and_read_only_in_live_containers(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "compose.json"
    prepare_probe(root, output)
    project = os.environ.get("COMPOSE_PROJECT_NAME", PROJECT)
    prefix = f"{project}-{uuid.uuid4().hex}"
    names = []
    try:
        for service in SERVICES:
            name = f"{prefix}-{service}"
            names.append(name)
            readback = subprocess.run(
                [
                    "docker",
                    "compose",
                    "--project-name",
                    project,
                    "--file",
                    str(output),
                    "run",
                    "--name",
                    name,
                    "--no-deps",
                    service,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert readback.returncode == 0, readback.stdout + readback.stderr
            assert (
                f"{service}: overlay bytes match; bind is read-only" in readback.stdout
            )
    finally:
        if names:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", *names],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                LOGGER.warning("Docker probe cleanup timed out for %s", names)


if __name__ == "__main__":
    main()
