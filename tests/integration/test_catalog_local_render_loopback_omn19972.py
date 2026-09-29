# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19972: the laptop profile's written render binds loopback and names no lab host.

The unit tests pin ``generate_compose`` directly. This drives the real
``omnibase_infra.docker.catalog.cli generate`` entry point in a subprocess, the
way ``make up-local`` does, with an env file built from the laptop template, and
reads the compose file it wrote: that file is what ``docker compose`` runs.

The ``core`` render is the control. The lab lanes render the same shared
manifests and must keep publishing on every interface with the lab's
advertised broker address, so a render that bound loopback everywhere, or
that stripped the lab default everywhere, fails here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_TEMPLATE = REPO_ROOT / "docker" / "local.env.example"
_OVERLAY_TEMPLATE = (
    REPO_ROOT / "docker" / "lane-overlays" / "local.bifrost.example.yaml"
)
_LAB_HOST_MARKERS = ("192.168.86.", "tail75df5e", "omninode-pc")


def _render(bundle: str, tmp_path: Path) -> tuple[dict[str, object], str]:
    env_file = tmp_path / f"{bundle}.env"
    env_text = re.sub(
        r"__REPLACE_WITH_[A-Z_]+__",
        "integration-test-value",
        _ENV_TEMPLATE.read_text(encoding="utf-8"),
    )
    env_text = re.sub(
        r"^ONEX_LOCAL_BIFROST_OVERLAY=.*$",
        f"ONEX_LOCAL_BIFROST_OVERLAY={_OVERLAY_TEMPLATE}",
        env_text,
        flags=re.MULTILINE,
    )
    env_file.write_text(env_text, encoding="utf-8")
    output = tmp_path / f"{bundle}.compose.yml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "generate",
            bundle,
            "--env-file",
            str(env_file),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rendered = output.read_text(encoding="utf-8")
    return yaml.safe_load(rendered), rendered


def _published_ports(compose: dict[str, object]) -> list[str]:
    services = compose["services"]
    assert isinstance(services, dict)
    return [str(p) for svc in services.values() for p in svc.get("ports") or []]


@pytest.mark.integration
def test_cli_local_render_binds_loopback_and_names_no_lab_host(tmp_path: Path) -> None:
    compose, rendered = _render("local", tmp_path)

    ports = _published_ports(compose)
    # The five host ports the laptop guide names. A render that published
    # nothing would pass the loopback check below vacuously.
    assert len(ports) >= 5, ports
    assert all(p.startswith("127.0.0.1:") and p.count(":") == 2 for p in ports), ports

    assert [m for m in _LAB_HOST_MARKERS if m in rendered] == []
    services = compose["services"]
    assert isinstance(services, dict)
    advertise = " ".join(services["redpanda"]["command"])
    assert "${REDPANDA_ADVERTISE_HOST:-localhost}" in advertise


@pytest.mark.integration
def test_cli_core_render_keeps_all_interface_ports_and_the_lab_default(
    tmp_path: Path,
) -> None:
    compose, rendered = _render("core", tmp_path)

    ports = _published_ports(compose)
    assert ports
    assert all(p.count(":") == 1 for p in ports), ports
    # Positive control for the lab-host check above: the shared manifest's
    # lab default is still in a lab-facing render.
    assert "${REDPANDA_ADVERTISE_HOST:-192.168.86.201}" in rendered
