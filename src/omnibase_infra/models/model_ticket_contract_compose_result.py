# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Composed YAML or a refusal; composition never executes evidence checks."""

from pydantic import BaseModel, ConfigDict


class ModelTicketContractComposeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    contract_yaml: str = ""
    error_message: str | None = None
