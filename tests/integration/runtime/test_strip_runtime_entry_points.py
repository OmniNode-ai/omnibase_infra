# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for runtime image entry-point stripping."""

from __future__ import annotations

import subprocess
import sys
from os import environ
from pathlib import Path


def _write_dist(
    site_packages: Path,
    name: str,
    entry_points: str,
) -> Path:
    dist_info = site_packages / f"{name.replace('-', '_')}-1.0.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Name: {name}\n", encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    return dist_info


def test_runtime_entry_point_stripper_cli_preserves_only_market_runtime_surface(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    legacy_dist = _write_dist(
        site_packages,
        "omninode-memory",
        """
[onex.node_package]
omnimemory = omnimemory.nodes

[onex.nodes]
node_memory = omnimemory.nodes.node_memory
""".lstrip(),
    )
    market_dist = _write_dist(
        site_packages,
        "omnimarket",
        """
[onex.node_package]
omnimarket = omnimarket.nodes

[onex.nodes]
node_runtime_sweep = omnimarket.nodes.node_runtime_sweep
""".lstrip(),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.runtime.strip_runtime_entry_points",
            "--site-packages",
            str(site_packages),
        ],
        check=True,
        capture_output=True,
        env={**environ, "PYTHONPATH": str(Path.cwd() / "src")},
        text=True,
    )

    assert "omninode-memory" in completed.stdout
    assert not (legacy_dist / "entry_points.txt").exists()
    market_entry_points = (market_dist / "entry_points.txt").read_text(encoding="utf-8")
    assert "[onex.node_package]" in market_entry_points
    assert "[onex.nodes]" in market_entry_points
