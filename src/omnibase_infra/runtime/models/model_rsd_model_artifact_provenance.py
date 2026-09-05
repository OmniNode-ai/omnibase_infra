# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Strict provenance facts binding one model artifact to a served identity."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.rsd_live_delegation_schema import (
    _RFC3339_UTC,
    CanonicalCapabilityRef,
    CanonicalEd25519Signature,
    CanonicalSha256,
    CanonicalUuid4,
)

_REGISTRY_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_IDENTIFIER = r"^[A-Za-z][A-Za-z0-9._-]{1,127}$"
_SHA1 = r"^[0-9a-f]{40}$"
_RUNTIME_VERSION = r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$"

ModelRegistryId = Annotated[str, Field(pattern=_REGISTRY_ID)]
RuntimeVersion = Annotated[str, Field(pattern=_RUNTIME_VERSION)]
ArtifactManifestAlgorithm = Literal[
    "sha256-canonical-json-v1", "sha256-path-size-content-sha256-v1"
]


def _reject_topology_like_registry_id(value: str) -> str:
    """Keep registry identifiers distinct from host and endpoint names."""
    owner = value.split("/", 1)[0].lower()
    if owner in {"localhost", "loopback"} or re.fullmatch(
        r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", owner
    ):
        raise ValueError("registry identifiers must not contain host identities")
    return value


class ModelRsdModelArtifactProvenance(BaseModel):
    """Signed, topology-free provenance for exactly one inert model choice."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["rsd.model-artifact-provenance.v1"]
    execute_enabled: Literal[False]
    approval_status: Literal["unapproved"]
    base_model_id: ModelRegistryId
    base_model_revision_sha: str = Field(pattern=_SHA1)
    artifact_id: ModelRegistryId
    artifact_revision_sha: str | None = Field(default=None, pattern=_SHA1)
    artifact_manifest_digest_sha256: CanonicalSha256 | None = None
    artifact_manifest_algorithm: ArtifactManifestAlgorithm | None = None
    quantization: str = Field(pattern=_IDENTIFIER)
    weight_activation_precision: Literal["w4a4"]
    kv_cache_dtype: Literal["fp8"]
    architecture: str = Field(pattern=_IDENTIFIER)
    runtime_implementation: str = Field(pattern=_IDENTIFIER)
    runtime_version: RuntimeVersion | None = None
    required_hardware_capability: str = Field(pattern=_IDENTIFIER)
    served_model_id: ModelRegistryId
    issued_at: str = Field(pattern=_RFC3339_UTC)
    expires_at: str = Field(pattern=_RFC3339_UTC)
    signer_capability_ref: CanonicalCapabilityRef
    signer_key_id: CanonicalUuid4
    signer_public_key_fingerprint_sha256: CanonicalSha256
    signature_domain: Literal["omninode-rsd.model-artifact-provenance.v1"]
    signature_base64: CanonicalEd25519Signature

    @model_validator(mode="after")
    def validate_provenance_binding(self) -> Self:
        """Require one immutable artifact binding and reject topology aliases."""
        if (
            self.artifact_revision_sha is None
            and self.artifact_manifest_digest_sha256 is None
        ):
            raise ValueError(
                "an immutable artifact revision or manifest digest is required"
            )
        if self.artifact_manifest_digest_sha256 is None:
            if self.artifact_manifest_algorithm is not None:
                raise ValueError("manifest algorithm requires a manifest digest")
        elif self.artifact_manifest_algorithm is None:
            raise ValueError("manifest digest requires a declared algorithm")
        for value in (self.base_model_id, self.artifact_id, self.served_model_id):
            _reject_topology_like_registry_id(value)
        return self


__all__ = ["ModelRsdModelArtifactProvenance"]
