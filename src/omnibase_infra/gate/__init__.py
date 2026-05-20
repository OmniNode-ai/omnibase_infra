# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OmniGate infra execution surfaces."""

from __future__ import annotations

from omnibase_infra.gate.executor import SecretBearingOutputError, execute_checks
from omnibase_infra.gate.signer import OmniGateSigner
from omnibase_infra.gate.validator_registry import (
    OmniGateValidatorCallable,
    clear_builtin_validators,
    execute_validator,
    register_builtin_validator,
    resolve_validator,
)

__all__ = [
    "OmniGateSigner",
    "OmniGateValidatorCallable",
    "SecretBearingOutputError",
    "clear_builtin_validators",
    "execute_checks",
    "execute_validator",
    "register_builtin_validator",
    "resolve_validator",
]
