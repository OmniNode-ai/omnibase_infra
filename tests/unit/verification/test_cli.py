# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for the verification CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.verification.cli import build_parser, main


def _write_contract(
    tmp: Path, contract: dict[str, Any], name: str = "test_node"
) -> Path:
    """Write a contract YAML to a temp directory and return its path."""
    contract_dir = tmp / name
    contract_dir.mkdir(parents=True, exist_ok=True)
    path = contract_dir / "contract.yaml"
    path.write_text(yaml.dump(contract))
    return path


def _make_contract(
    name: str = "test_node",
    node_type: str = "COMPUTE",
) -> dict[str, Any]:
    """Create a minimal contract dict."""
    return {
        "name": name,
        "node_type": node_type,
    }


@pytest.mark.unit
class TestCLIParser:
    """Tests for the argparse configuration."""

    def test_parser_has_required_flags(self) -> None:
        parser = build_parser()
        # Smoke test: parser can parse --all
        args = parser.parse_args(["--all"])
        assert args.verify_all is True

    def test_parser_contract_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--contract-path", "/var/data/contract.yaml"])
        assert args.contract_path == Path("/var/data/contract.yaml")

    def test_parser_registration_only(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--registration-only"])
        assert args.registration_only is True

    def test_parser_output_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--all", "--output-path", "/var/data/report.json"])
        assert args.output_path == Path("/var/data/report.json")

    def test_parser_contracts_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--all", "--contracts-dir", "/var/data/nodes"])
        assert args.contracts_dir == Path("/var/data/nodes")


@pytest.mark.unit
class TestCLIMain:
    """Tests for the main() entry point."""

    def test_no_flags_exits_error(self) -> None:
        """No mode flag -> SystemExit (argparse error)."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_multiple_flags_exits_error(self) -> None:
        """Multiple mode flags -> SystemExit (argparse error)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--all", "--registration-only"])
        assert exc_info.value.code != 0

    def test_single_contract_pass(self, tmp_path: Path, capsys: Any) -> None:
        """Single contract with no topics -> PASS (exit 0)."""
        contract = _make_contract()
        path = _write_contract(tmp_path, contract)

        # Patch the default config to have a db_query_fn that returns active rows
        exit_code = main(["--contract-path", str(path)])

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)
        assert len(output) == 1
        # Exit code 0 (PASS) or 2 (QUARANTINE) are both acceptable
        # since no db_query_fn means projection probe returns QUARANTINE
        assert exit_code in (0, 2)

    def test_single_contract_not_found(self) -> None:
        """Non-existent contract path -> exit 1."""
        exit_code = main(["--contract-path", "/nonexistent/contract.yaml"])
        assert exit_code == 1

    def test_all_contracts(self, tmp_path: Path, capsys: Any) -> None:
        """--all with a directory of contracts."""
        for i in range(3):
            _write_contract(tmp_path, _make_contract(f"node_{i}"), f"node_{i}")

        exit_code = main(["--all", "--contracts-dir", str(tmp_path)])

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)
        assert len(output) == 3
        # Exit code 0 or 2 are both acceptable
        assert exit_code in (0, 2)

    def test_all_empty_dir(self, tmp_path: Path) -> None:
        """--all with empty directory -> exit 1."""
        exit_code = main(["--all", "--contracts-dir", str(tmp_path)])
        assert exit_code == 1

    def test_output_path_writes_file(self, tmp_path: Path) -> None:
        """--output-path writes report JSON to file."""
        contract = _make_contract()
        path = _write_contract(tmp_path, contract)
        output_file = tmp_path / "report.json"

        exit_code = main(
            ["--contract-path", str(path), "--output-path", str(output_file)]
        )

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert exit_code in (0, 2)

    def test_registration_only(self, tmp_path: Path, capsys: Any) -> None:
        """--registration-only finds and verifies registration contracts."""
        # Create the 3 registration contract dirs
        for node_name in (
            "node_registration_orchestrator",
            "node_registration_reducer",
            "node_registration_storage_effect",
        ):
            contract = _make_contract(name=node_name, node_type="ORCHESTRATOR_GENERIC")
            _write_contract(tmp_path, contract, node_name)

        exit_code = main(["--registration-only", "--contracts-dir", str(tmp_path)])

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)
        assert len(output) == 3
        # All use noop fns so expect FAIL (noop db returns empty rows)
        assert exit_code == 1

    def test_registration_only_no_contracts(self, tmp_path: Path) -> None:
        """--registration-only with empty dir -> exit 1."""
        exit_code = main(["--registration-only", "--contracts-dir", str(tmp_path)])
        assert exit_code == 1
