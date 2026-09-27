# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Source, image and build identity of the consumer that ran a delegation (OMN-18930)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelDelegationBuildIdentity(BaseModel):
    """Content identities of the consumer's runtime, read at capture time.

    * ``source_revision`` -- the full commit the image was built from (the
      image's ``org.opencontainers.image.revision`` label);
    * ``image_digest`` -- the image content id the consumer container runs;
    * ``build_provenance_sha256`` -- SHA-256 of the image's workspace build
      provenance manifest, which pins every sibling package the build composed.

    Branch names, short shas, tags and ``unknown`` are refused: a label that
    can move is not a build identity.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    source_revision: str = Field(min_length=40, max_length=40)
    image_digest: str = Field(min_length=71, max_length=71)
    build_provenance_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_content_identities(self) -> ModelDelegationBuildIdentity:
        if not _GIT_SHA_PATTERN.fullmatch(self.source_revision):
            raise ValueError("source_revision must be a full lowercase git sha")
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.image_digest):
            raise ValueError("image_digest must be sha256:<64 lowercase hex>")
        if not _SHA256_PATTERN.fullmatch(self.build_provenance_sha256):
            raise ValueError("build_provenance_sha256 must be lowercase SHA-256 hex")
        return self
