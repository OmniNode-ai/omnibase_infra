# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The C28 producer's recorded observation shape, plus read status."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_board_probe_effect.models.typed_dict_consumer_flow import (
    TypedDictConsumerFlowBoot,
    TypedDictConsumerFlowCursor,
    TypedDictConsumerFlowIdentity,
    TypedDictConsumerFlowKinds,
    TypedDictConsumerFlowNegative,
)


class ModelConsumerFlowObservation(BaseModel):
    """Keep recorded wire evidence intact, including unknown exposure metadata.

    The nested mappings deliberately preserve the script's replay format; the
    pure grader checks counter types without coercing booleans or strings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    read_ok: bool
    read_error: str = ""
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    boot_identity: dict[str, TypedDictConsumerFlowIdentity] = Field(
        default_factory=dict
    )
    kinds: TypedDictConsumerFlowKinds = Field(
        default_factory=TypedDictConsumerFlowKinds
    )
    negative: TypedDictConsumerFlowNegative = Field(
        default_factory=TypedDictConsumerFlowNegative
    )
    cursor: TypedDictConsumerFlowCursor = Field(
        default_factory=TypedDictConsumerFlowCursor
    )
    boot: TypedDictConsumerFlowBoot = Field(default_factory=TypedDictConsumerFlowBoot)
