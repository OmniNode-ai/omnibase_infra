# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the Plugin* lifecycle class validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validators.no_plugin_daemon_classes import main, validate_paths


@pytest.mark.integration
def test_validator_scans_tree_and_reports_nested_plugin_lifecycle_class(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_root = tmp_path / "src" / "omnibase_infra" / "fake_plugins"
    package_root.mkdir(parents=True)
    (package_root / "plugin_runtime.py").write_text(
        """
class RuntimeBase:
    pass


class PluginRuntimeBridge(RuntimeBase):
    pass
""",
        encoding="utf-8",
    )
    (package_root / "plugin_compute.py").write_text(
        """
class PluginComputeBase:
    pass


class PluginRuntimeMetrics(PluginComputeBase):
    pass
""",
        encoding="utf-8",
    )

    findings = validate_paths([tmp_path / "src"])
    exit_code = main([str(tmp_path / "src")])
    captured = capsys.readouterr()

    assert [finding.class_name for finding in findings] == ["PluginRuntimeBridge"]
    assert exit_code == 1
    assert "PluginRuntimeBridge" in captured.err
