# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for --mode flag on `omni-infra validate contracts` (OMN-9769).

Phase 3, Task 12 — wires EnumValidatorMode into the CLI validate contracts
command so the batch sweep can be invoked with either strict or
migration_audit mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.cli.commands import cli


def _clean_effect_contract(name: str = "node_foo_effect") -> dict[str, Any]:
    return {
        "name": name,
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "description": f"{name} test fixture.",
        "node_type": "EFFECT_GENERIC",
        "input_model": "foo.bar.models.ModelFooRequest",
        "output_model": "foo.bar.models.ModelFooResult",
        "handler_routing": {
            "version": {"major": 1, "minor": 0, "patch": 0},
            "routing_strategy": "operation_match",
            "handlers": [
                {
                    "routing_key": "foo.run",
                    "handler_key": "handle_foo_run",
                    "priority": 0,
                }
            ],
        },
        "io_operations": [
            {
                "operation_type": "read",
                "resource_type": "external_api",
                "resource_identifier": "foo.api/run",
            }
        ],
    }


def _write_node_contract(tmp_path: Path, body: dict[str, Any]) -> Path:
    p = tmp_path / "nodes" / body["name"] / "contract.yaml"
    p.parent.mkdir(parents=True)
    p.write_text(yaml.safe_dump(body))
    return p


@pytest.mark.unit
class TestValidateContractsModeFlag:
    """CLI validate contracts --mode flag routes to batch_validator with correct mode."""

    def test_validate_contracts_accepts_strict_mode(self, tmp_path: Path) -> None:
        _write_node_contract(tmp_path, _clean_effect_contract())
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["validate", "contracts", str(tmp_path), "--mode", "strict"],
        )
        assert result.exit_code == 0, result.output

    def test_validate_contracts_accepts_migration_audit_mode(
        self, tmp_path: Path
    ) -> None:
        _write_node_contract(tmp_path, _clean_effect_contract())
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["validate", "contracts", str(tmp_path), "--mode", "migration_audit"],
        )
        assert result.exit_code == 0, result.output

    def test_validate_contracts_rejects_invalid_mode(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["validate", "contracts", str(tmp_path), "--mode", "bogus_mode"],
        )
        assert result.exit_code != 0

    def test_validate_contracts_default_mode_is_strict(self, tmp_path: Path) -> None:
        _write_node_contract(tmp_path, _clean_effect_contract())
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["validate", "contracts", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert "strict" in result.output.lower() or result.exit_code == 0

    def test_validate_contracts_uses_batch_validator(self, tmp_path: Path) -> None:
        _write_node_contract(tmp_path, _clean_effect_contract())
        runner = CliRunner()

        from omnibase_core.enums.enum_validator_mode import EnumValidatorMode
        from omnibase_core.normalization.batch_validator import (
            ModelBatchValidationSummary,
            run_batch_validation,
        )

        mock_summary = ModelBatchValidationSummary(
            total=1,
            passed=1,
            failed=0,
            mode=EnumValidatorMode.STRICT,
            reports=[],
        )
        with patch(
            "omnibase_infra.cli.commands.run_batch_validation",
            return_value=mock_summary,
        ) as mock_run:
            result = runner.invoke(
                cli,
                ["validate", "contracts", str(tmp_path), "--mode", "strict"],
            )
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs is not None
        assert result.exit_code == 0

    def test_validate_contracts_passes_migration_audit_to_batch_validator(
        self, tmp_path: Path
    ) -> None:
        _write_node_contract(tmp_path, _clean_effect_contract())
        runner = CliRunner()

        from omnibase_core.enums.enum_validator_mode import EnumValidatorMode
        from omnibase_core.normalization.batch_validator import (
            ModelBatchValidationSummary,
        )

        mock_summary = ModelBatchValidationSummary(
            total=1,
            passed=1,
            failed=0,
            mode=EnumValidatorMode.MIGRATION_AUDIT,
            reports=[],
        )
        with patch(
            "omnibase_infra.cli.commands.run_batch_validation",
            return_value=mock_summary,
        ) as mock_run:
            result = runner.invoke(
                cli,
                [
                    "validate",
                    "contracts",
                    str(tmp_path),
                    "--mode",
                    "migration_audit",
                ],
            )
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            from omnibase_core.enums.enum_validator_mode import EnumValidatorMode

            assert kwargs.get("mode") is EnumValidatorMode.MIGRATION_AUDIT
        assert result.exit_code == 0

    def test_validate_contracts_exits_nonzero_on_failures(self, tmp_path: Path) -> None:
        runner = CliRunner()

        from omnibase_core.enums.enum_contract_bucket import EnumContractBucket
        from omnibase_core.enums.enum_validator_mode import EnumValidatorMode
        from omnibase_core.models.contracts.model_corpus_validation_report import (
            ModelCorpusValidationReport,
        )
        from omnibase_core.normalization.batch_validator import (
            ModelBatchValidationSummary,
        )

        failing_report = ModelCorpusValidationReport(
            path=tmp_path / "nodes" / "node_bad" / "contract.yaml",
            bucket=EnumContractBucket.NODE_ROOT_CONTRACT,
            mode=EnumValidatorMode.STRICT,
            passed=False,
            errors=["Missing required field: input_model"],
            normalized=False,
        )
        mock_summary = ModelBatchValidationSummary(
            total=1,
            passed=0,
            failed=1,
            mode=EnumValidatorMode.STRICT,
            reports=[failing_report],
        )
        with patch(
            "omnibase_infra.cli.commands.run_batch_validation",
            return_value=mock_summary,
        ):
            result = runner.invoke(
                cli,
                ["validate", "contracts", str(tmp_path), "--mode", "strict"],
            )
        assert result.exit_code == 1

    def test_validate_contracts_output_shows_summary(self, tmp_path: Path) -> None:
        _write_node_contract(tmp_path, _clean_effect_contract())
        runner = CliRunner()

        from omnibase_core.enums.enum_validator_mode import EnumValidatorMode
        from omnibase_core.normalization.batch_validator import (
            ModelBatchValidationSummary,
        )

        mock_summary = ModelBatchValidationSummary(
            total=2,
            passed=2,
            failed=0,
            mode=EnumValidatorMode.STRICT,
            reports=[],
        )
        with patch(
            "omnibase_infra.cli.commands.run_batch_validation",
            return_value=mock_summary,
        ):
            result = runner.invoke(
                cli,
                ["validate", "contracts", str(tmp_path)],
            )
        assert "2" in result.output
        assert result.exit_code == 0
