# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OmniGate infra execution surfaces."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Load OmniGate surfaces lazily across stacked core branches."""
    if name in {"SecretBearingOutputError", "execute_checks"}:
        from omnibase_infra.gate.executor import (
            SecretBearingOutputError,
            execute_checks,
        )

        return {
            "SecretBearingOutputError": SecretBearingOutputError,
            "execute_checks": execute_checks,
        }[name]
    if name == "OmniGateSigner":
        from omnibase_infra.gate.signer import OmniGateSigner

        return OmniGateSigner
    if name in {
        "OmniGateValidatorCallable",
        "clear_builtin_validators",
        "execute_validator",
        "register_builtin_validator",
        "resolve_validator",
    }:
        from omnibase_infra.gate.validator_registry import (
            OmniGateValidatorCallable,
            clear_builtin_validators,
            execute_validator,
            register_builtin_validator,
            resolve_validator,
        )

        return {
            "OmniGateValidatorCallable": OmniGateValidatorCallable,
            "clear_builtin_validators": clear_builtin_validators,
            "execute_validator": execute_validator,
            "register_builtin_validator": register_builtin_validator,
            "resolve_validator": resolve_validator,
        }[name]
    raise AttributeError(name)


__all__ = [
    "OmniGateValidatorCallable",
    "OmniGateSigner",
    "SecretBearingOutputError",
    "clear_builtin_validators",
    "execute_checks",
    "execute_validator",
    "register_builtin_validator",
    "resolve_validator",
]
