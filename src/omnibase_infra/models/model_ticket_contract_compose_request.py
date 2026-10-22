# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Filesystem inputs for composing one ticket's additive contract fragments."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelTicketContractComposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["ticket_contract.compose"] = "ticket_contract.compose"
    contracts_dir: str = "contracts"
    ticket_ref: str
