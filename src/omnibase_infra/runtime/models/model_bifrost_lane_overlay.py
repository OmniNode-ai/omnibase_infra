# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed model for the Bifrost lane overlay YAML (OMN-12864).

The lane overlay commits the four BIFROST_LOCAL_* endpoint URL bindings that
were previously only available via ephemeral shell exports on .201. Storing
them as a committed typed YAML file under docker/lane-overlays/ makes the
deployment authority auditable, diff-able, and CI-checked.

Each endpoint_url is the COMPLETE final URL including the /v1/chat/completions
path (OMN-12815). The overlay file is rendered to a dotenv sidecar
(docker/lane-overlays/<lane>.bifrost.env) consumed by the compose stack at
runtime. Any mismatch between the YAML source and rendered env sidecar fails
CI via test_bifrost_lane_overlay_env_in_sync.

Validation enforces OMN-12815: every URL must end in /v1/chat/completions or
/chat/completions to catch bare-base misconfiguration early.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelBifrostLaneOverlay(BaseModel):
    """Typed deployment bindings for Bifrost local backend endpoint URLs.

    All four endpoint_url values are COMPLETE final URLs (OMN-12815):
    they are posted VERBATIM at every call site — no in-code construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    lane: str = Field(
        min_length=1,
        description="Lane identifier (e.g. 'dev', 'stability-test', 'prod').",
    )
    coder_endpoint_url: str = Field(
        min_length=1,
        description=(
            "COMPLETE chat-completions URL for the local coder backend "
            "(vLLM on .201:8000). Posted VERBATIM — no path appended."
        ),
    )
    reasoner_endpoint_url: str = Field(
        min_length=1,
        description=(
            "COMPLETE chat-completions URL for the local reasoner backend "
            "(vLLM on .201:8001). Posted VERBATIM — no path appended."
        ),
    )
    embedding_endpoint_url: str = Field(
        min_length=1,
        description=(
            "COMPLETE chat-completions URL for the local embedding backend "
            "(.201:8100). Posted VERBATIM — no path appended."
        ),
    )
    ds4_flash_endpoint_url: str = Field(
        min_length=1,
        description=(
            "COMPLETE chat-completions URL for the DS-V4-Flash backend "
            "(.200:8101). Posted VERBATIM — no path appended."
        ),
    )

    @model_validator(mode="after")
    def _validate_complete_endpoint_urls(self) -> ModelBifrostLaneOverlay:
        """Reject bare-base URLs that lack the /chat/completions path."""
        fields = {
            "coder_endpoint_url": self.coder_endpoint_url,
            "reasoner_endpoint_url": self.reasoner_endpoint_url,
            "embedding_endpoint_url": self.embedding_endpoint_url,
            "ds4_flash_endpoint_url": self.ds4_flash_endpoint_url,
        }
        errors: list[str] = []
        for field_name, url in fields.items():
            stripped = url.rstrip("/")
            if not (stripped.endswith(("/v1/chat/completions", "/chat/completions"))):
                errors.append(
                    f"{field_name}: endpoint_url must end in /v1/chat/completions "
                    f"or /chat/completions (OMN-12815), got {url!r}"
                )
        if errors:
            msg = "Bifrost lane overlay endpoint_url completeness errors: " + "; ".join(
                errors
            )
            raise ValueError(msg)
        return self

    def as_env_dict(self) -> dict[str, str]:
        """Return the env-var mapping consumed by the compose stack."""
        return {
            "BIFROST_LOCAL_CODER_ENDPOINT_URL": self.coder_endpoint_url,
            "BIFROST_LOCAL_REASONER_ENDPOINT_URL": self.reasoner_endpoint_url,
            "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL": self.embedding_endpoint_url,
            "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL": self.ds4_flash_endpoint_url,
        }


__all__: list[str] = ["ModelBifrostLaneOverlay"]
