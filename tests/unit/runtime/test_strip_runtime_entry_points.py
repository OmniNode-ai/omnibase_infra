# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import pytest

from omnibase_infra.runtime.strip_runtime_entry_points import (
    DEFAULT_STRIPPED_GROUPS,
    normalize_distribution_name,
    strip_runtime_entry_points,
)


def _dist_info(
    site_packages: Path,
    name: str,
    version: str = "1.0.0",
    *,
    entry_points: str,
) -> Path:
    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Name: {name}\n", encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    return dist_info


@pytest.mark.unit
def test_strips_onex_runtime_groups_from_non_market_distribution(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    non_market = _dist_info(
        site_packages,
        "omninode-memory",
        entry_points="""
[onex.node_package]
omnimemory = omnimemory.nodes

[onex.nodes]
node_memory_storage_effect = omnimemory.nodes.node_memory_storage_effect

[onex.domain_plugins]
memory = omnimemory.plugin:PluginMemory

[console_scripts]
omni-memory = omnimemory.cli:main
""".lstrip(),
    )
    allowed = _dist_info(
        site_packages,
        "omnimarket",
        entry_points="""
[onex.node_package]
omnimarket = omnimarket.nodes

[onex.nodes]
node_market = omnimarket.nodes.node_market
""".lstrip(),
    )

    report = strip_runtime_entry_points([site_packages])

    non_market_entry_points = (non_market / "entry_points.txt").read_text(
        encoding="utf-8"
    )
    assert "[console_scripts]" in non_market_entry_points
    for group in DEFAULT_STRIPPED_GROUPS:
        assert f"[{group}]" not in non_market_entry_points

    allowed_entry_points = (allowed / "entry_points.txt").read_text(encoding="utf-8")
    assert "[onex.node_package]" in allowed_entry_points
    assert "[onex.nodes]" in allowed_entry_points

    assert [dist.distribution for dist in report.stripped_distributions] == [
        "omninode-memory"
    ]
    assert report.stripped_distributions[0].groups == tuple(
        sorted(DEFAULT_STRIPPED_GROUPS)
    )
    assert report.stripped_distributions[0].removed_file is False


@pytest.mark.unit
def test_removes_entry_points_file_when_only_runtime_groups_remain(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    dist_info = _dist_info(
        site_packages,
        "omninode-intelligence",
        entry_points="""
[onex.domain_plugins]
intelligence = omniintelligence.plugin:PluginIntelligence
""".lstrip(),
    )

    report = strip_runtime_entry_points([site_packages])

    assert not (dist_info / "entry_points.txt").exists()
    assert report.stripped_distributions[0].distribution == "omninode-intelligence"
    assert report.stripped_distributions[0].removed_file is True


@pytest.mark.unit
def test_distribution_name_normalization_handles_underscores_and_dots() -> None:
    assert normalize_distribution_name("omninode_intelligence") == (
        "omninode-intelligence"
    )
    assert normalize_distribution_name("OmniNode.Memory") == "omninode-memory"


_INTELLIGENCE_ENTRY_POINTS = """
[onex.domain_plugins]
intelligence = omniintelligence.runtime.plugin:PluginIntelligence

[onex.node_package]
omniintelligence = omniintelligence.nodes

[onex.nodes]
node_pattern_storage_effect = omniintelligence.nodes.node_pattern_storage_effect
""".lstrip()


@pytest.mark.unit
def test_intelligence_keeps_onex_nodes_so_its_http_mount_is_discovered(
    tmp_path: Path,
) -> None:
    """OMN-22391: the image's omninode-intelligence still lists its onex.nodes."""
    site_packages = tmp_path / "site-packages"
    dist_info = _dist_info(
        site_packages,
        "omninode-intelligence",
        version="0.25.3",
        entry_points=_INTELLIGENCE_ENTRY_POINTS,
    )

    report = strip_runtime_entry_points([site_packages])

    entry_points = (dist_info / "entry_points.txt").read_text(encoding="utf-8")
    assert "[onex.nodes]" in entry_points
    assert "node_pattern_storage_effect" in entry_points
    assert "[onex.domain_plugins]" not in entry_points
    assert "[onex.node_package]" not in entry_points
    assert report.stripped_distributions[0].groups == (
        "onex.domain_plugins",
        "onex.node_package",
    )
    assert report.stripped_distributions[0].removed_file is False


@pytest.mark.unit
def test_runtime_dockerfile_fails_the_build_without_intelligence_nodes() -> None:
    """The image build asserts the installed dist lists onex.nodes after the strip."""
    dockerfile = Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.runtime"
    lines = [
        line
        for line in dockerfile.read_text(encoding="utf-8").splitlines()
        if line.startswith("RUN ")
    ]
    strip = [i for i, line in enumerate(lines) if "strip_runtime_entry_points" in line]
    assert len(strip) == 1
    after = "\n".join(lines[strip[0] + 1 :])
    assert re.search(
        r'entry_points\(group="onex\.nodes"\).*assert "omninode-intelligence" in',
        after,
    )
