# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The staging input whose opaque JSON remains owned by workflow state."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    Sha256Digest,
)
from omnibase_infra.runtime.first_effect_ledger.verified_first_effect_grant_types import (
    CanonicalModelEventClass,
    VerifiedGrantExpectedOutputTopic,
)


class ModelFirstEffectStageRequest(BaseModel):
    """One opaque pending emission, written only to workflow pending_emissions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    authorization_digest: Sha256Digest
    expected_ledger_version: int = Field(ge=0)
    expected_workflow_version: int = Field(ge=0)
    outbox_body_sha256: Sha256Digest
    emitted_output_topic: VerifiedGrantExpectedOutputTopic
    emitted_output_event_class: CanonicalModelEventClass
    emitted_output_event_index: Literal[0]
    pending_emissions_json: str = Field(min_length=2)

    @model_validator(mode="after")
    def _single_pending_emission(self) -> ModelFirstEffectStageRequest:
        try:
            decoded = json.loads(self.pending_emissions_json)
        except json.JSONDecodeError as exc:
            raise ValueError("pending_emissions_json must be valid JSON") from exc
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise ValueError("pending_emissions_json must be a single-entry array")
        entry = decoded[0]
        if not isinstance(entry, dict):
            raise ValueError("pending_emissions_json entry must be an object")
        if entry.get("class_name") != self.emitted_output_event_class:
            raise ValueError(
                "pending emission class_name must match emitted output class"
            )
        if entry.get("index") != self.emitted_output_event_index:
            raise ValueError("pending emission index must match emitted output index")
        return self


__all__ = ["ModelFirstEffectStageRequest"]
