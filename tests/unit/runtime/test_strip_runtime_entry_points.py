# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

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


def test_distribution_name_normalization_handles_underscores_and_dots() -> None:
    assert normalize_distribution_name("omninode_intelligence") == (
        "omninode-intelligence"
    )
    assert normalize_distribution_name("OmniNode.Memory") == "omninode-memory"
