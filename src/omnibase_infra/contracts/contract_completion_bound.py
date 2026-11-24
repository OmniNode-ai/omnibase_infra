# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``completion_bound`` block of a contract, read through the contract cache."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.contracts.contract_yaml import read_contract_yaml
from omnibase_infra.runtime.state_io.model_completion_bound import (
    ModelCompletionBound,
)

__all__ = ["read_completion_bound"]


def read_completion_bound(contract_path: Path) -> ModelCompletionBound | None:
    """Read and validate the top-level ``completion_bound`` block (OMN-18296).

    Returns ``None`` when the block is absent — a ``state_io`` contract that
    declares no bound keeps the pre-OMN-18296 behaviour (the environment-variable
    give-up TTL, row-only), so this is additive for every contract that has not
    opted in. A block that IS present but malformed raises: a bound the runtime
    cannot read is worse than no bound at all, because a reader would believe one
    was being enforced.

    Shape and rationale live in
    :mod:`omnibase_infra.runtime.state_io.model_completion_bound`.
    """
    try:
        raw = read_contract_yaml(contract_path)
    except FileNotFoundError:
        return None
    if not isinstance(raw, dict):
        return None
    block = raw.get("completion_bound")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ModelOnexError(
            f"handler_wiring: completion_bound in {contract_path} must be a "
            f"mapping — got {type(block).__name__}."
        )
    try:
        return ModelCompletionBound.model_validate(block)
    except ValidationError as exc:
        raise ModelOnexError(
            f"handler_wiring: completion_bound in {contract_path} is invalid: {exc}"
        ) from exc
