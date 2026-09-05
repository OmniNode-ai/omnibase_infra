# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Primary metadata evidence for the inert RSD artifact binding."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelRsdArtifactSourceRelation(BaseModel):
    """Immutable primary-metadata evidence with an explicit non-approval state."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_metadata_authority: Literal["huggingface-model-revision-api-v1"]
    source_model_card_sha256: Literal[
        "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
    ]
    source_config_sha256: Literal[
        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
    ]
    artifact_metadata_authority: Literal["huggingface-model-revision-api-v1"]
    artifact_model_card_sha256: Literal[
        "3704987ff0e2206ab934af6d71cd0a9b5140536ee8d305aa7ba6e7665f135058"
    ]
    artifact_embedded_source_model_card_sha256: Literal[
        "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
    ]
    artifact_declared_base_model_id: Literal["Qwen/Qwen3.8-27B"]
    relation_status: Literal["publisher-declared-unverified"]
    approval_status: Literal["not-approved"]


__all__ = ["ModelRsdArtifactSourceRelation"]
