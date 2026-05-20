# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for OmniGate check execution."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from omnibase_core.enums.ticket.enum_receipt_status import EnumReceiptStatus
from omnibase_core.models.common.model_validation_result import ModelValidationResult
from omnibase_core.models.gate.model_omnigate_check import ModelOmniGateCheck
from omnibase_core.models.gate.model_omnigate_config import ModelOmniGateConfig
from omnibase_core.models.gate.model_omnigate_validator_ref import (
    ModelOmniGateValidatorRef,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.gate.executor import SecretBearingOutputError, execute_checks
from omnibase_infra.gate.validator_registry import (
    clear_builtin_validators,
    register_builtin_validator,
)


def _config(
    *,
    checks: tuple[ModelOmniGateCheck, ...] = (),
    validators: tuple[ModelOmniGateValidatorRef, ...] = (),
    denied_command_patterns: tuple[str, ...] = (),
) -> ModelOmniGateConfig:
    return ModelOmniGateConfig(
        version=ModelSemVer(major=1, minor=0, patch=0),
        project_name="test",
        project_url="https://github.com/a/b",
        denied_command_patterns=denied_command_patterns,
        checks=checks,
        validators=validators,
    )


@pytest.fixture(autouse=True)
def _clear_validators() -> None:
    clear_builtin_validators()


@pytest.mark.unit
class TestExecuteChecks:
    def test_passing_check(self, tmp_path: Path) -> None:
        config = _config(
            checks=(ModelOmniGateCheck(name="echo", run="echo ok"),),
        )

        results = execute_checks(config, tmp_path)

        assert len(results) == 1
        assert results[0].status == EnumReceiptStatus.PASS
        assert results[0].name == "echo"

    def test_invokes_bash_without_shell_true(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run(
            args: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        config = _config(
            checks=(ModelOmniGateCheck(name="echo", run="echo ok"),),
        )

        execute_checks(config, tmp_path)

        assert captured["args"] == ["bash", "-eo", "pipefail", "-c", "echo ok"]
        assert "shell" not in captured["kwargs"]

    def test_failing_check(self, tmp_path: Path) -> None:
        config = _config(
            checks=(ModelOmniGateCheck(name="fail", run="exit 1"),),
        )

        results = execute_checks(config, tmp_path)

        assert results[0].status == EnumReceiptStatus.FAIL

    def test_captures_stdout_and_stderr_preview_and_hash(
        self,
        tmp_path: Path,
    ) -> None:
        command = "printf hello_world; printf err_world >&2"
        output = "hello_worlderr_world"
        config = _config(
            checks=(ModelOmniGateCheck(name="hello", run=command),),
        )

        results = execute_checks(config, tmp_path)

        assert results[0].stdout_preview == output
        expected_digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
        assert results[0].stdout_hash == f"sha256:{expected_digest}"

    def test_caps_stdout_preview_but_hashes_full_output(self, tmp_path: Path) -> None:
        output = "x" * 5000
        config = _config(
            checks=(ModelOmniGateCheck(name="long", run=f"printf {output!r}"),),
        )

        results = execute_checks(config, tmp_path)

        assert results[0].stdout_preview == output[:4096]
        expected_digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
        assert results[0].stdout_hash == f"sha256:{expected_digest}"

    def test_uses_sanitized_environment_per_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OMNIGATE_ALLOWED", "visible")
        monkeypatch.setenv("OMNIGATE_BLOCKED", "hidden")
        config = _config(
            checks=(
                ModelOmniGateCheck(
                    name="env",
                    run=(
                        'test "$OMNIGATE_ALLOWED" = visible && '
                        'test -z "${OMNIGATE_BLOCKED:-}"'
                    ),
                    allowed_env=("OMNIGATE_ALLOWED",),
                ),
            ),
        )

        results = execute_checks(config, tmp_path)

        assert results[0].status == EnumReceiptStatus.PASS

    def test_refuses_secret_bearing_output(self, tmp_path: Path) -> None:
        config = _config(
            checks=(ModelOmniGateCheck(name="secret", run="printf 'token=abc123'"),),
        )

        with pytest.raises(SecretBearingOutputError):
            execute_checks(config, tmp_path)

    def test_timeout_returns_failure(self, tmp_path: Path) -> None:
        config = _config(
            checks=(
                ModelOmniGateCheck(
                    name="timeout",
                    run="sleep 2",
                    timeout_seconds=1,
                ),
            ),
        )

        results = execute_checks(config, tmp_path)

        assert results[0].status == EnumReceiptStatus.FAIL
        assert "timed out after 1s" in (results[0].stdout_preview or "")

    def test_denied_command_pattern_blocks_execution(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker"
        config = _config(
            checks=(ModelOmniGateCheck(name="denied", run=f"touch {marker}"),),
            denied_command_patterns=(r"\btouch\b",),
        )

        results = execute_checks(config, tmp_path)

        assert results[0].status == EnumReceiptStatus.FAIL
        assert "Command denied" in (results[0].stdout_preview or "")
        assert not marker.exists()

    def test_executes_validators_after_shell_checks_in_config_order(
        self,
        tmp_path: Path,
    ) -> None:
        def passing_validator(
            repo_path: Path,
            config: dict[str, object],
        ) -> ModelValidationResult[None]:
            assert repo_path == tmp_path
            assert config == {"strict": True}
            return ModelValidationResult.create_valid(summary="validator ok")

        register_builtin_validator("local-validator", passing_validator)
        config = _config(
            checks=(ModelOmniGateCheck(name="echo", run="echo shell"),),
            validators=(
                ModelOmniGateValidatorRef(
                    id="local-validator",
                    config={"strict": True},
                ),
            ),
        )

        results = execute_checks(config, tmp_path)

        assert [result.name for result in results] == ["echo", "local-validator"]
        assert [result.status for result in results] == [
            EnumReceiptStatus.PASS,
            EnumReceiptStatus.PASS,
        ]

    def test_unknown_validator_fails_closed(self, tmp_path: Path) -> None:
        config = _config(
            validators=(ModelOmniGateValidatorRef(id="missing-validator"),),
        )

        results = execute_checks(config, tmp_path)

        assert results[0].status == EnumReceiptStatus.FAIL
        assert "Unknown OmniGate validator id" in (results[0].stdout_preview or "")
