# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OmniGate validator registry and result normalization."""

from __future__ import annotations

import time
from collections.abc import Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

from omnibase_core.enums.ticket.enum_receipt_status import EnumReceiptStatus
from omnibase_core.models.common.model_validation_result import ModelValidationResult
from omnibase_core.models.gate.model_omnigate_check_result import (
    ModelOmniGateCheckResult,
)
from omnibase_core.models.gate.model_omnigate_validator_ref import (
    ModelOmniGateValidatorRef,
)
from omnibase_core.validation.validator_base import ValidatorBase

ENTRY_POINT_GROUP = "omnigate.validators"
type ValidatorConfig = Mapping[str, object]


@runtime_checkable
class OmniGateValidatorCallable(Protocol):
    """Callable validator surface for OmniGate registry adapters."""

    def __call__(
        self,
        repo_path: Path,
        config: ValidatorConfig,
    ) -> ModelOmniGateCheckResult:
        """Run a validator for a repository path."""


_BUILTIN_VALIDATORS: dict[str, object] = {}


def register_builtin_validator(
    validator_id: str,
    validator: object,
) -> None:
    """Register an infra-owned validator for tests or local adapters."""
    _BUILTIN_VALIDATORS[validator_id] = validator


def clear_builtin_validators() -> None:
    """Clear infra-owned validators registered in-process."""
    _BUILTIN_VALIDATORS.clear()


def resolve_validator(validator_id: str) -> object | None:
    """Resolve a validator from entry points first, then built-ins."""
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name == validator_id:
            loaded_validator: object = entry_point.load()
            return loaded_validator
    return _BUILTIN_VALIDATORS.get(validator_id)


def execute_validator(
    validator_ref: ModelOmniGateValidatorRef,
    repo_path: Path,
) -> ModelOmniGateCheckResult:
    """Execute one declared validator and fail closed when it is unavailable."""
    start = time.monotonic()
    validator = resolve_validator(validator_ref.id)
    if validator is None:
        return _validator_result(
            validator_ref.id,
            EnumReceiptStatus.FAIL,
            int((time.monotonic() - start) * 1000),
            f"Unknown OmniGate validator id: {validator_ref.id}",
        )

    try:
        raw_result = _invoke_validator(
            validator,
            repo_path,
            validator_ref.config or {},
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return _normalize_validator_result(validator_ref.id, raw_result, duration_ms)
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.monotonic() - start) * 1000)
        return _validator_result(
            validator_ref.id,
            EnumReceiptStatus.FAIL,
            duration_ms,
            f"Validator {validator_ref.id} raised {type(exc).__name__}",
        )


def _invoke_validator(
    validator: object,
    repo_path: Path,
    config: ValidatorConfig,
) -> object:
    if isinstance(validator, type) and issubclass(validator, ValidatorBase):
        return validator().validate(repo_path)
    if isinstance(validator, ValidatorBase):
        return validator.validate(repo_path)
    if callable(validator):
        return validator(repo_path, config)
    msg = f"Registered validator is not callable: {type(validator).__name__}"
    raise TypeError(msg)


def _normalize_validator_result(
    validator_id: str,
    raw_result: object,
    duration_ms: int,
) -> ModelOmniGateCheckResult:
    if isinstance(raw_result, ModelOmniGateCheckResult):
        return raw_result
    if isinstance(raw_result, bool):
        return _validator_result(
            validator_id,
            EnumReceiptStatus.PASS if raw_result else EnumReceiptStatus.FAIL,
            duration_ms,
            "Validator passed" if raw_result else "Validator failed",
        )

    if isinstance(raw_result, ModelValidationResult):
        status = (
            EnumReceiptStatus.PASS if raw_result.is_valid else EnumReceiptStatus.FAIL
        )
        preview = raw_result.summary
        if raw_result.issues:
            messages = [issue.message for issue in raw_result.issues[:20]]
            preview = "\n".join((raw_result.summary, *messages))
        return _validator_result(validator_id, status, duration_ms, preview)

    return _validator_result(
        validator_id,
        EnumReceiptStatus.FAIL,
        duration_ms,
        f"Validator returned unsupported result type: {type(raw_result).__name__}",
    )


def _validator_result(
    validator_id: str,
    status: EnumReceiptStatus,
    duration_ms: int,
    stdout_preview: str,
) -> ModelOmniGateCheckResult:
    from omnibase_infra.gate.executor import redact_if_secret_bearing

    redacted = redact_if_secret_bearing(stdout_preview)
    return ModelOmniGateCheckResult(
        name=validator_id,
        command=f"validator:{validator_id}",
        status=status,
        duration_ms=duration_ms,
        stdout_preview=redacted[:4096] if redacted else None,
        stdout_hash=None,
    )


__all__ = [
    "ENTRY_POINT_GROUP",
    "OmniGateValidatorCallable",
    "clear_builtin_validators",
    "execute_validator",
    "register_builtin_validator",
    "resolve_validator",
]
