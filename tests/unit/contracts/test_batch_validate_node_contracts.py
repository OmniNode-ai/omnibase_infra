# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for batch_validate_node_contracts.py — OMN-9752.

Validates that the batch validator:
- Passes all well-formed contracts through model_validate()
- Fails on invalid node_type values
- Fails on missing required fields
- Reports correct counts
- Returns non-zero exit on any failure
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "batch_validate_node_contracts.py"
)


def _write_contract(directory: Path, name: str, node_type: str) -> Path:
    """Write a minimal valid contract.yaml into directory/name/contract.yaml."""
    node_dir = directory / name
    node_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "name": name,
        "node_type": node_type,
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "node_version": "1.0.0",
        "description": "Test contract",
        "input_model": {
            "name": "ModelInput",
            "module": "omnibase_core.models.model_input",
            "description": "Input",
        },
        "output_model": {
            "name": "ModelOutput",
            "module": "omnibase_core.models.model_output",
            "description": "Output",
        },
        "handler_routing": {
            "routing_strategy": "payload_type_match",
            "handlers": [],
        },
    }
    path = node_dir / "contract.yaml"
    path.write_text(yaml.dump(contract, default_flow_style=False), encoding="utf-8")
    return path


def _run(directory: Path, *, verbose: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), "--directory", str(directory)]
    if verbose:
        cmd.append("--verbose")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)


class TestBatchValidateAllPass:
    """All valid contracts → exit 0, correct pass count."""

    def test_all_valid_contracts_pass(self, tmp_path: Path) -> None:
        for name, node_type in [
            ("node_foo_effect", "EFFECT_GENERIC"),
            ("node_bar_compute", "COMPUTE_GENERIC"),
            ("node_baz_reducer", "REDUCER_GENERIC"),
            ("node_qux_orchestrator", "ORCHESTRATOR_GENERIC"),
        ]:
            _write_contract(tmp_path, name, node_type)

        result = _run(tmp_path)
        assert result.returncode == 0, f"Expected exit 0; stderr: {result.stderr}"
        assert "4/4 passed" in result.stdout

    def test_lowercase_node_type_aliases_pass(self, tmp_path: Path) -> None:
        """Lowercase aliases like 'effect' must be accepted (MixinNodeTypeValidator)."""
        node_dir = tmp_path / "node_alias_effect"
        node_dir.mkdir()
        contract = {
            "name": "node_alias_effect",
            "node_type": "effect",
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "handlers": [],
            },
        }
        (node_dir / "contract.yaml").write_text(
            yaml.dump(contract, default_flow_style=False), encoding="utf-8"
        )
        result = _run(tmp_path)
        assert result.returncode == 0, f"Expected exit 0; stderr: {result.stderr}"

    def test_empty_directory_exits_zero(self, tmp_path: Path) -> None:
        """No contracts found → exit 0 (nothing to fail)."""
        result = _run(tmp_path)
        assert result.returncode == 0


class TestBatchValidateFailures:
    """Invalid contracts → exit 1, failure listed in output."""

    def test_invalid_node_type_fails(self, tmp_path: Path) -> None:
        node_dir = tmp_path / "node_bad"
        node_dir.mkdir()
        bad = {
            "name": "node_bad",
            "node_type": "TOTALLY_WRONG",
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "handlers": [],
            },
        }
        (node_dir / "contract.yaml").write_text(
            yaml.dump(bad, default_flow_style=False), encoding="utf-8"
        )
        result = _run(tmp_path)
        assert result.returncode == 1, f"Expected exit 1; stdout: {result.stdout}"
        assert (
            "0/1 passed, 1 failed" in result.stdout
            or "Failed contracts:" in result.stdout
        )

    def test_missing_node_type_fails(self, tmp_path: Path) -> None:
        node_dir = tmp_path / "node_missing_type"
        node_dir.mkdir()
        bad = {
            "name": "node_missing_type",
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "handlers": [],
            },
        }
        (node_dir / "contract.yaml").write_text(
            yaml.dump(bad, default_flow_style=False), encoding="utf-8"
        )
        result = _run(tmp_path)
        assert result.returncode == 1

    def test_invalid_yaml_fails(self, tmp_path: Path) -> None:
        node_dir = tmp_path / "node_bad_yaml"
        node_dir.mkdir()
        (node_dir / "contract.yaml").write_text(
            "key: [unclosed bracket\n", encoding="utf-8"
        )
        result = _run(tmp_path)
        assert result.returncode == 1

    def test_mixed_valid_and_invalid_counts_correctly(self, tmp_path: Path) -> None:
        _write_contract(tmp_path, "node_good", "COMPUTE_GENERIC")
        node_dir = tmp_path / "node_bad"
        node_dir.mkdir()
        bad = {
            "name": "node_bad",
            "node_type": "NOT_A_TYPE",
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "handlers": [],
            },
        }
        (node_dir / "contract.yaml").write_text(
            yaml.dump(bad, default_flow_style=False), encoding="utf-8"
        )
        result = _run(tmp_path)
        assert result.returncode == 1
        assert "1/2 passed, 1 failed" in result.stdout


class TestBatchValidateCLI:
    """CLI argument tests."""

    def test_missing_directory_exits_2(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        result = _run(nonexistent)
        assert result.returncode == 2

    def test_verbose_flag_shows_paths(self, tmp_path: Path) -> None:
        _write_contract(tmp_path, "node_verbose_test", "EFFECT_GENERIC")
        result = _run(tmp_path, verbose=True)
        assert result.returncode == 0
        assert "node_verbose_test" in result.stdout
        assert "PASS" in result.stdout


class TestBatchValidateRealContracts:
    """Smoke test: all real node contracts in the repo must pass model_validate()."""

    def test_all_infra_node_contracts_pass(self) -> None:
        """Every contract.yaml under src/omnibase_infra/nodes/ must pass model_validate().

        This is the CI gate: if any contract has an invalid node_type, this test fails.
        """
        nodes_dir = (
            Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"
        )
        assert nodes_dir.is_dir(), f"Nodes directory not found: {nodes_dir}"

        result = _run(nodes_dir, verbose=True)
        assert result.returncode == 0, (
            f"Batch contract validation failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
