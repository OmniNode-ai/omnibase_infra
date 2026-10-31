# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model identity pinned by the RSD contract, read once from the contract file."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Any, cast

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "contracts" / "rsd_pinned_model_identity.yaml"
)


class ModelRsdPinnedModelIdentity(BaseModel):
    """Model ids the inert RSD binding accepts, declared in the contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    served_model: str
    base_model: str
    artifact_model: str
    artifact_ref: str
    architecture: str
    launch_profile: str
    observed_model: str

    @classmethod
    @functools.cache
    def from_contract(cls) -> ModelRsdPinnedModelIdentity:
        """Load the pinned identity from the packaged contract."""
        document = cast(
            "dict[str, Any]",
            yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8")),
        )
        return cls.model_validate(document["pinned_model_identity"])

    def require(self, field_name: str, value: str) -> str:
        """Return ``value`` when it equals the pinned ``field_name``, else raise."""
        if field_name not in type(self).model_fields:
            raise ValueError(f"{field_name} is not a pinned model identity field")
        if value != getattr(self, field_name):
            raise ValueError(
                f"model identity differs from the contract pin {field_name}"
            )
        return value


def _pinned(field_name: str) -> AfterValidator:
    """Build a validator that holds a field to the contract-declared pin."""
    return AfterValidator(
        lambda value: ModelRsdPinnedModelIdentity.from_contract().require(
            field_name, value
        )
    )


ServedModelId = Annotated[str, _pinned("served_model")]
BaseModelId = Annotated[str, _pinned("base_model")]
ArtifactModelId = Annotated[str, _pinned("artifact_model")]
ArtifactId = Annotated[str, _pinned("artifact_ref")]
Architecture = Annotated[str, _pinned("architecture")]
LaunchProfileId = Annotated[str, _pinned("launch_profile")]
ObservedModelId = Annotated[str, _pinned("observed_model")]

__all__ = [
    "Architecture",
    "ArtifactId",
    "ArtifactModelId",
    "BaseModelId",
    "LaunchProfileId",
    "ModelRsdPinnedModelIdentity",
    "ObservedModelId",
    "ServedModelId",
]
