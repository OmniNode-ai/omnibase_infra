# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19972 demo half: the laptop render carries the services the six pages read.

The unit tests pin ``generate_compose`` directly. This drives the real
``omnibase_infra.docker.catalog.cli generate`` entry point in a subprocess, the
way ``make up-local`` does, and reads the compose file it wrote: that file is
what ``docker compose`` runs, so a service, port or dependency lost between the
resolver and the written file fails here.

The ``runtime-observability-projections`` render is the control. It uses the
same shared manifests, and the laptop-only port and dependency overrides must
not reach it.
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
_PAGE_SERVICES = (
    "projection-api",
    "omnimarket-projection-llm-cost",
    "consumer-health-projection",
    # OMN-19972 (T3.6): the usage-by-model writer's carrier (RUNTIME_PROFILE
    # tenant-projection); the page's usage rows come from it.
    "tenant-projection-writer",
)


def _render(bundle: str, tmp_path: Path) -> dict[str, dict[str, object]]:
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
    compose = yaml.safe_load(output.read_text(encoding="utf-8"))
    services = compose["services"]
    assert isinstance(services, dict)
    return services


@pytest.mark.integration
def test_cli_local_render_writes_the_page_services_healthy_and_ordered(
    tmp_path: Path,
) -> None:
    services = _render("local", tmp_path)

    assert [name for name in _PAGE_SERVICES if name not in services] == []
    # The laptop's projection API is off the lab lanes' 3002 and on loopback.
    assert services["projection-api"]["ports"] == ["127.0.0.1:3102:3002"]
    # It starts only after the kernel that provisions its exposure topics is
    # healthy; without that it waits 300 s for topic metadata and exits.
    depends_on = services["projection-api"]["depends_on"]
    assert isinstance(depends_on, dict)
    assert depends_on.get("omninode-runtime") == {"condition": "service_healthy"}
    # It is told where the broker is; without it the API exits at startup.
    env = services["projection-api"]["environment"]
    assert isinstance(env, dict)
    assert env.get("KAFKA_BROKERS") == "redpanda:9092"
    # Every long-running service reports health, so the CI boot step can tell
    # a dead one from a live one.
    unwatched = sorted(
        name
        for name, svc in services.items()
        if svc.get("restart") != "no" and "healthcheck" not in svc
    )
    assert unwatched == []


@pytest.mark.integration
def test_cli_other_bundle_render_keeps_the_shared_projection_api(
    tmp_path: Path,
) -> None:
    services = _render("runtime-observability-projections", tmp_path)

    # Control: the laptop-only overrides stay in the laptop bundle.
    assert services["projection-api"]["ports"] == ["3002:3002"]
    depends_on = services["projection-api"]["depends_on"]
    assert isinstance(depends_on, dict)
    assert "omninode-runtime" not in depends_on
