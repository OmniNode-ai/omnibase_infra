# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Strict provenance facts binding one model artifact to a served identity."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_rsd_artifact_source_relation import (
    ModelRsdArtifactSourceRelation,
)
from omnibase_infra.runtime.models.rsd_live_delegation_schema import (
    _RFC3339_UTC,
    CanonicalCapabilityRef,
    CanonicalEd25519Signature,
    CanonicalSha256,
    CanonicalUuid4,
)


class ModelRsdModelArtifactProvenance(BaseModel):
    """Signed, topology-free provenance for exactly one inert model choice."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["rsd.model-artifact-provenance.v2"]
    execute_enabled: Literal[False]
    approval_status: Literal["unapproved"]
    model_id: Literal["qwen/qwen3.8-27b"]
    base_model_id: Literal["Qwen/Qwen3.8-27B"]
    base_model_revision_sha: Literal["1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
    artifact_id: Literal["gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"]
    artifact_revision_sha: Literal["0cc27958cefbbe231782ec8511de8c4eb5233348"]
    artifact_manifest_digest_sha256: Literal[
        "e46ef4e3895ed0a6db7c237d642121095629c53bd5b3e5ac799b8a8e2ae83e4f"
    ]
    artifact_manifest_algorithm: Literal["sha256-path-size-content-sha256-v1"]
    artifact_source_relation: ModelRsdArtifactSourceRelation
    quantization: Literal["modelopt_nvfp4"]
    weight_activation_precision: Literal["w4a4"]
    kv_cache_dtype: Literal["fp8"]
    architecture: Literal["Qwen3_5ForConditionalGeneration"]
    runtime_implementation: Literal["vllm"]
    runtime_version: Literal["0.27.1"]
    required_hardware_capability: Literal["nvidia.rtx5090_32gb"]
    served_model_id: Literal["Qwen/Qwen3.8-27B"]
    launch_profile_id: Literal["qwen38-nvfp4-rtx5090-v1"]
    launch_profile_digest: Literal[
        "40defad1345d27226916e8946647482bb3eaaeca96c4330968e6a0bcaad074b3"
    ]
    issued_at: str = Field(pattern=_RFC3339_UTC)
    expires_at: str = Field(pattern=_RFC3339_UTC)
    signer_capability_ref: CanonicalCapabilityRef
    signer_key_id: CanonicalUuid4
    signer_public_key_fingerprint_sha256: CanonicalSha256
    signature_domain: Literal["omninode-rsd.model-artifact-provenance.v2"]
    signature_base64: CanonicalEd25519Signature


__all__ = ["ModelRsdModelArtifactProvenance"]
