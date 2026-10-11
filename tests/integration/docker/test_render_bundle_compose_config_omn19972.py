# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The rendered ``local`` bundle is a compose project docker accepts (OMN-19972).

``render_bundle`` hands the local provision node a compose file and the env
files to interpolate it from. This runs the real ``docker compose config`` over
exactly that pair, the same flags the node passes, so a render that compose
cannot interpolate fails here rather than at ``onex local up``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import render_bundle
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision import (
    compose_base_command,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not on PATH"),
]


def test_local_bundle_render_passes_compose_config(tmp_path: Path) -> None:
    required = (
        CatalogResolver(catalog_dir=str(_REPO_ROOT / "docker" / "catalog"))
        .resolve(["local"])
        .required_env
    )
    # An absolute path satisfies every required variable's shape for
    # ``config``: some are bind-mount sources (the model overlay file).
    value = tmp_path / "value"
    value.touch()
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "".join(f"{var}={value}\n" for var in sorted(required)), encoding="utf-8"
    )
    rendered = render_bundle(
        ["local"],
        infra_root=_REPO_ROOT,
        env_file=env_file,
        output_path=tmp_path / "compose.yml",
        process_environment={},
    )
    cmd = compose_base_command(Path(rendered.compose_file_path), rendered.env_files) + [
        "config",
        "--services",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if "is not a docker command" in proc.stderr:
        pytest.skip("docker compose plugin not installed")
    assert proc.returncode == 0, proc.stderr
    assert set(proc.stdout.split()) == set(rendered.services)
