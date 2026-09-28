# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The laptop profile renders from one env file, through the real CLI (OMN-19496).

``make up-local`` writes ``~/.omnibase/local.env`` from ``docker/local.env.example``
(two generated passwords and the absolute path of the model overlay copied from
``docker/lane-overlays/local.bifrost.example.yaml``), then runs the catalog CLI
with ``--env-file`` pointing at it. These tests drive that path end to end
across its real parts -- the shipped templates on disk, the CLI as a separate
process, the env loader, the resolver, the host-port clash validator and the
generator -- with a process environment that carries none of the operator
values, so a pass means the one env file is sufficient on its own.

Nothing here starts a container; the render is written to a temporary file.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[3]
_ENV_TEMPLATE = _REPO / "docker" / "local.env.example"
_OVERLAY_TEMPLATE = _REPO / "docker" / "lane-overlays" / "local.bifrost.example.yaml"
_PROJECT = "omnibase-infra-local"
_OVERLAY_PIN = "/app/config/delegation/local.bifrost.yaml"
_OPERATOR_KEYS = ("POSTGRES_PASSWORD", "VALKEY_PASSWORD", "ONEX_LOCAL_BIFROST_OVERLAY")


def _write_laptop_files(tmp_path: Path) -> tuple[Path, Path]:
    """Do what ``make local-env`` does: copy the overlay, fill the env template."""
    overlay = tmp_path / "omnibase" / "local.bifrost.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(_OVERLAY_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    values = {
        "POSTGRES_PASSWORD": secrets.token_hex(32),
        "VALKEY_PASSWORD": secrets.token_hex(32),
        "ONEX_LOCAL_BIFROST_OVERLAY": str(overlay),
    }
    lines = []
    for line in _ENV_TEMPLATE.read_text(encoding="utf-8").splitlines():
        key = line.partition("=")[0]
        lines.append(f"{key}={values[key]}" if key in values else line)
    env_file = tmp_path / "omnibase" / "local.env"
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file, overlay


def _cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # HOME points at the temporary directory so no operator ~/.omnibase/.env can
    # be read, and no operator value is inherited from this process.
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    for key in _OPERATOR_KEYS:
        assert key not in env

    python_path = os.pathsep.join(
        path
        for path in (
            str(_REPO / "src"),
            str(_REPO),
            os.environ.get("PYTHONPATH", ""),
        )
        if path
    )
    env["PYTHONPATH"] = python_path

    return subprocess.run(
        [sys.executable, "-m", "omnibase_infra.docker.catalog.cli", *args],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_laptop_env_file_alone_validates_and_renders_the_local_stack(
    tmp_path: Path,
) -> None:
    env_file, overlay = _write_laptop_files(tmp_path)

    validated = _cli(tmp_path, "validate", "local", "--env-file", str(env_file))
    assert validated.returncode == 0, validated.stdout + validated.stderr
    assert "All required env vars are set." in validated.stdout

    output = tmp_path / "compose.local.yml"
    generated = _cli(
        tmp_path,
        "generate",
        "local",
        "--env-file",
        str(env_file),
        "--output",
        str(output),
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr

    compose: dict[str, Any] = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert compose["name"] == _PROJECT
    services = compose["services"]
    assert {
        "postgres",
        "redpanda",
        "valkey",
        "forward-migration",
        "migration-gate",
        "omninode-runtime",
        "runtime-effects",
        "omnimarket-projection-delegation",
    } <= set(services)

    published: list[str] = []
    for svc in services.values():
        for port in svc.get("ports", []):
            published.append(str(port).rsplit(":", 1)[0])
    assert published, "the laptop stack publishes host ports; none were read"
    assert len(published) == len(set(published)), published

    for name in ("omninode-runtime", "runtime-effects"):
        mounts = [
            v for v in services[name]["volumes"] if v.endswith(f":{_OVERLAY_PIN}:ro")
        ]
        assert len(mounts) == 1, services[name]["volumes"]
        assert mounts[0].startswith("${ONEX_LOCAL_BIFROST_OVERLAY:?")
    # The path the env file hands that mount is a typed overlay for this lane.
    ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(overlay.read_text(encoding="utf-8"))
    )


def test_unfilled_template_is_refused_by_the_cli(tmp_path: Path) -> None:
    validated = _cli(tmp_path, "validate", "local", "--env-file", str(_ENV_TEMPLATE))
    assert validated.returncode == 1, validated.stdout + validated.stderr
    assert "template placeholders" in validated.stderr
    for key in _OPERATOR_KEYS:
        assert key in validated.stderr


def test_env_file_missing_a_required_value_is_refused_by_the_cli(
    tmp_path: Path,
) -> None:
    env_file, _ = _write_laptop_files(tmp_path)
    kept = [
        line
        for line in env_file.read_text(encoding="utf-8").splitlines()
        if not line.startswith("VALKEY_PASSWORD=")
    ]
    env_file.write_text("\n".join(kept) + "\n", encoding="utf-8")

    validated = _cli(tmp_path, "validate", "local", "--env-file", str(env_file))
    assert validated.returncode == 1, validated.stdout + validated.stderr
    assert "VALKEY_PASSWORD" in validated.stderr
