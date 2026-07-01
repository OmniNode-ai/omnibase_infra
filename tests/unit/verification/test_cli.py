# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for the verification CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification import cli
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

    def test_registration_only(
        self,
        tmp_path: Path,
        capsys: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--registration-only finds and verifies registration contracts."""
        # Hermeticity: clear OMNIBASE_INFRA_DB_URL so _make_runtime_db_query_fn()
        # returns None. When the var leaks in from a developer shell or a CI
        # runner pointed at a live Postgres, the DB-backed checks PASS against
        # real data and main() returns 0 instead of the asserted (1, 2). The
        # "No runtime DB is configured in this unit test" precondition below is
        # only true once this var is removed.
        monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
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
        # No runtime DB is configured in this unit test, so DB-backed checks
        # quarantine instead of fabricating empty projection rows.
        assert exit_code in (1, 2)

    def test_registration_only_uses_runtime_db_for_projection_state(
        self,
        tmp_path: Path,
        capsys: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OMN-9539/9540/9541: configured DB rows drive projection_state PASS."""
        for node_name, node_type in (
            ("node_registration_orchestrator", "ORCHESTRATOR_GENERIC"),
            ("node_registration_reducer", "REDUCER_GENERIC"),
            ("node_registration_storage_effect", "EFFECT_GENERIC"),
        ):
            contract = _make_contract(name=node_name, node_type=node_type)
            _write_contract(tmp_path, contract, node_name)

        def db_query_fn(sql: str) -> list[dict[str, str]]:
            if "information_schema.columns" in sql:
                return [
                    {"column_name": "entity_id"},
                    {"column_name": "current_state"},
                    {"column_name": "node_type"},
                    {"column_name": "data_provenance"},
                ]
            if "WHERE node_type = 'orchestrator'" in sql:
                return [
                    {
                        "entity_id": "orchestrator-1",
                        "current_state": "active",
                        "node_type": "orchestrator",
                    }
                ]
            return [
                {
                    "entity_id": "orchestrator-1",
                    "current_state": "active",
                    "node_type": "orchestrator",
                },
                {
                    "entity_id": "reducer-1",
                    "current_state": "active",
                    "node_type": "reducer",
                },
                {
                    "entity_id": "effect-1",
                    "current_state": "active",
                    "node_type": "effect",
                },
            ]

        monkeypatch.setattr(cli, "_make_runtime_db_query_fn", lambda: db_query_fn)

        exit_code = cli.main(["--registration-only", "--contracts-dir", str(tmp_path)])

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)
        assert len(output) == 3
        assert exit_code == 0

        for report in output:
            projection_check = next(
                check
                for check in report["checks"]
                if check["check_type"] == "projection_state"
            )
            assert projection_check["verdict"] == "pass"
            assert "Found 3/3 rows in terminal states" in projection_check["evidence"]
            assert (
                "No rows found in registration_projections"
                not in projection_check["evidence"]
            )

    def test_registration_only_no_contracts(self, tmp_path: Path) -> None:
        """--registration-only with empty dir -> exit 1."""
        exit_code = main(["--registration-only", "--contracts-dir", str(tmp_path)])
        assert exit_code == 1


@pytest.mark.unit
class TestCLIWiresDbQueryFn:
    """Regression for OMN-13555 Defect 2.

    ``--all`` and ``--contract-path`` must build a real ``db_query_fn`` from
    ``OMNIBASE_INFRA_DB_URL`` (mirroring ``--registration-only``) so projection
    probes run instead of auto-QUARANTINE.
    """

    def _capture_configs(self, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        captured: list[Any] = []
        real_report = MagicMock()
        real_report.model_dump.return_value = {"overall_verdict": "pass", "checks": []}
        real_report.overall_verdict = EnumValidationVerdict.PASS

        def fake_run(path: Path, config: Any) -> Any:
            captured.append(config)
            return real_report

        monkeypatch.setattr(cli, "run_contract_verification", fake_run)
        # Force a real (non-None) db_query_fn regardless of host DB config.
        sentinel_fn = lambda sql: []  # noqa: E731
        monkeypatch.setattr(cli, "_make_runtime_db_query_fn", lambda: sentinel_fn)
        return captured

    def test_all_builds_non_none_db_query_fn(
        self, tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_configs(monkeypatch)
        _write_contract(tmp_path, _make_contract("node_a"), "node_a")

        cli.main(["--all", "--contracts-dir", str(tmp_path)])
        capsys.readouterr()

        assert captured, "run_contract_verification was never invoked"
        assert all(cfg.db_query_fn is not None for cfg in captured)

    def test_contract_path_builds_non_none_db_query_fn(
        self, tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_configs(monkeypatch)
        path = _write_contract(tmp_path, _make_contract("node_b"), "node_b")

        cli.main(["--contract-path", str(path)])
        capsys.readouterr()

        assert len(captured) == 1
        assert captured[0].db_query_fn is not None

    def test_make_runtime_config_wires_db_query_fn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel_fn = lambda sql: []  # noqa: E731
        monkeypatch.setattr(cli, "_make_runtime_db_query_fn", lambda: sentinel_fn)
        config = cli._make_runtime_config()
        assert config.db_query_fn is sentinel_fn
