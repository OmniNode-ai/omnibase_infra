# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17648: the real ``onex skill`` dispatch leaves node boolean defaults in control."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_skill

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("flags", [[], ["--flag-only", "--dry-run"]])
def test_linear_housekeeping_dispatch_preserves_node_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str]
) -> None:
    """Inspect the actual CLI dispatch payload without executing Linear writes."""
    captured: list[dict[str, object]] = []

    def capture_dispatch(**kwargs: object) -> int:
        assert kwargs["node_name"] == "node_linear_triage"
        input_path = kwargs["input_path"]
        assert isinstance(input_path, Path)
        captured.append(json.loads(input_path.read_text(encoding="utf-8")))
        return 0

    monkeypatch.setattr(cli_skill, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_skill, "_resolve_packaged_contract", lambda _: tmp_path / "contract.yaml"
    )
    monkeypatch.setattr(cli_skill, "run_receipt_mode", capture_dispatch)
    cli_skill.load_skill_registry.cache_clear()
    try:
        result = CliRunner().invoke(
            cli_skill.run_skill_by_name,
            ["linear_housekeeping", "--state-root", str(tmp_path), *flags],
        )
        assert result.exit_code == 0, result.output
        assert captured == ([{"flag_only": True, "dry_run": True}] if flags else [{}])
    finally:
        cli_skill.load_skill_registry.cache_clear()
