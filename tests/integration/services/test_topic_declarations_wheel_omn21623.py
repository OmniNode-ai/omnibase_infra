# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise topic declarations from the built distribution, outside the source tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

pytestmark = pytest.mark.integration


def test_wheel_loads_topic_declarations_and_serves_snapshots(tmp_path: Path) -> None:
    """The shipped YAML must load and drive the cache in an isolated interpreter."""
    root = Path(__file__).resolve().parents[3]
    wheel_dir = tmp_path / "dist"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "-t",
            "wheel",
            "-d",
            str(wheel_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    (wheel,) = wheel_dir.glob("*.whl")
    installed = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        archive.extractall(installed)

    probe = r"""
import sys
from pathlib import Path

from omnibase_infra.models.cost_api import load_cost_api_topics
from omnibase_infra.models.developer_principal.topics import SCOPED_WRITE_TOPICS
from omnibase_infra.nodes.node_board_probe_effect.handlers import _consumer_flow_constants
from omnibase_infra.services.cost_api import snapshot_cache

installed = Path(sys.argv[1]).resolve()
assert Path(snapshot_cache.__file__).resolve().is_relative_to(installed)
assert Path(_consumer_flow_constants.__file__).resolve().is_relative_to(installed)
topics = load_cost_api_topics()
assert SCOPED_WRITE_TOPICS
assert _consumer_flow_constants.EXPOSURE_TOPIC.startswith("onex.snapshot.")
assert _consumer_flow_constants.SEAM_DLQ.startswith("onex.dlq.")
for name, topic in topics.model_dump().items():
    payload = {"snapshot": name, "total": 7}
    snapshot_cache.store_latest_snapshot(topic, "1h", payload)
    readback = snapshot_cache.get_latest_snapshot(topic, "1h")
    assert readback == payload
    readback["total"] = 99
    assert snapshot_cache.get_latest_snapshot(topic, "1h") == payload
print("wheel-runtime-readback: 3 snapshot round trips; cost, board and plan declarations loaded")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(installed)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(installed),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
