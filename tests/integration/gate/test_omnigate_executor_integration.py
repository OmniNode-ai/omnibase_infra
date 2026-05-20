# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the OmniGate check executor (OMN-11142).

Exercises `execute_checks` against the real `subprocess.run` path with real
shell commands inside a tmp_path repo. Unlike the unit suite (which patches
subprocess to capture argv), this test exercises the full bash-pipefail
invocation, the real stdout/stderr capture, secret-pattern scanning over
genuine output, and the validator-registry composition path with a
registered builtin validator.
"""

from __future__ import annotations

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

pytestmark = pytest.mark.integration


def _config(
    *,
    checks: tuple[ModelOmniGateCheck, ...] = (),
    validators: tuple[ModelOmniGateValidatorRef, ...] = (),
    denied_command_patterns: tuple[str, ...] = (),
) -> ModelOmniGateConfig:
    return ModelOmniGateConfig(
        version=ModelSemVer(major=1, minor=0, patch=0),
        project_name="integration",
        project_url="https://github.com/a/b",
        denied_command_patterns=denied_command_patterns,
        checks=checks,
        validators=validators,
    )


@pytest.fixture(autouse=True)
def _clear_validators() -> None:
    clear_builtin_validators()
    yield
    clear_builtin_validators()


def test_real_bash_pipefail_invocation_passes(tmp_path: Path) -> None:
    """Real bash -eo pipefail execution: success path produces PASS receipt."""
    config = _config(checks=(ModelOmniGateCheck(name="ok", run="echo ok"),))
    results = execute_checks(config, tmp_path)
    assert len(results) == 1
    assert results[0].status == EnumReceiptStatus.PASS
    assert results[0].name == "ok"


def test_real_bash_pipefail_invocation_fails_on_nonzero_exit(tmp_path: Path) -> None:
    """Real shell failure produces FAIL receipt."""
    config = _config(checks=(ModelOmniGateCheck(name="boom", run="exit 7"),))
    results = execute_checks(config, tmp_path)
    assert results[0].status == EnumReceiptStatus.FAIL


def test_secret_scanner_blocks_token_in_real_output(tmp_path: Path) -> None:
    """A command that emits a GitHub PAT-style token must be rejected."""
    config = _config(
        checks=(
            ModelOmniGateCheck(
                name="leak",
                run="echo ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA0123456789",
            ),
        ),
    )
    with pytest.raises(SecretBearingOutputError):
        execute_checks(config, tmp_path)


def test_validator_registry_composes_with_shell_checks(tmp_path: Path) -> None:
    """Shell checks and registered validators both appear in the result tuple."""

    def _always_pass(
        repo_path: Path,
        config: dict[str, object],
    ) -> ModelValidationResult[None]:
        assert repo_path == tmp_path
        return ModelValidationResult.create_valid(summary="integration ok")

    register_builtin_validator("integration-validator", _always_pass)

    config = _config(
        checks=(ModelOmniGateCheck(name="shell-ok", run="echo hi"),),
        validators=(ModelOmniGateValidatorRef(id="integration-validator"),),
    )

    results = execute_checks(config, tmp_path)
    names = {r.name for r in results}
    assert names == {"shell-ok", "integration-validator"}
    assert all(r.status == EnumReceiptStatus.PASS for r in results)
