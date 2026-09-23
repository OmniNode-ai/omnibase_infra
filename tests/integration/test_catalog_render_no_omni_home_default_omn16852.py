# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16852 AC3: the catalog CLI's written render hands the build no empty OMNI_HOME.

The unit test pins ``generate_compose`` directly. This drives the real
``omnibase_infra.docker.catalog.cli generate`` entry point in a subprocess with
``OMNI_HOME`` removed from its environment, then reads the compose file it
wrote, because the file on disk is what ``docker compose`` and the deploy
agent's compose_gen actually consume.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.integration
def test_cli_render_carries_no_omni_home_build_arg(tmp_path: Path) -> None:
    output = tmp_path / "docker-compose.generated.yml"
    env = {key: value for key, value in os.environ.items() if key != "OMNI_HOME"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "generate",
            "runtime",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rendered = output.read_text(encoding="utf-8")
    compose = yaml.safe_load(rendered)

    build_args = compose["services"]["omninode-runtime"]["build"]["args"]
    # Positive control: the render really carries the runtime build stanza, so
    # the absence below is a statement about the stanza, not an empty read.
    assert build_args["BUILD_SOURCE"] == "${BUILD_SOURCE:-release}"
    assert "OMNI_HOME" not in build_args
    assert "${OMNI_HOME:-" not in rendered
