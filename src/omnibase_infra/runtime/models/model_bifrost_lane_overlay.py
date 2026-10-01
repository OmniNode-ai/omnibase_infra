# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Strict contract overlay for Bifrost delegation bindings (OMN-15807).

OMN-17502 adds the execution-locale axis. See
:class:`~omnibase_infra.runtime.models.enum_bifrost_lane_locale.EnumBifrostLaneLocale`
for why a lane that runs off the lab network must reach no local backend.
Cloud overlays may be empty, rebind base cloud backends, or add fully declared
non-local backends. The model rejects local-tier additions; the renderer also
rejects bindings targeting local-tier backends in the base contract.

OMN-17099 removed the set-equality rule: a lab lane used to have to declare
EXACTLY a backend set hardcoded in the product. Which backends a lab lane must
bind is now derived from the base contract by the renderer — every local
backend the base contract routes to — and a lane may add backends the base
does not declare, fully specified.
The overlay chooses the served model independently of the base contract.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    LOCAL_TIER,
    ModelBifrostLaneBackendBinding,
)

# v2 -> v3 (OMN-17502): ``locale`` is a required field, so a v2 file is not a
# v3 file. The version is bumped rather than made permissive on purpose — an
# image/overlay skew in either direction then fails naming the schema version,
# instead of failing on a missing field whose absence used to be legal.
_SCHEMA_VERSION = "bifrost_lane_overlay.v3"


class ModelBifrostLaneOverlay(BaseModel):
    """The typed authority for a lane's Bifrost endpoint and model bindings.

    Lab lanes bind every routed local backend (enforced by the renderer).
    Cloud lanes may bind only non-local backends, or declare no bindings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    schema_version: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    #: Required, no default. A defaulted locale would make ``lab`` the silent
    #: answer for any overlay that forgot to declare one — the same fallthrough
    #: class OMN-17150 removed from the overlay PATH, one level down in the
    #: overlay CONTENT.
    locale: EnumBifrostLaneLocale
    backends: tuple[ModelBifrostLaneBackendBinding, ...]

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {_SCHEMA_VERSION!r}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _validate_bindings_against_locale(self) -> Self:
        backend_keys = [binding.backend_key for binding in self.backends]
        if len(backend_keys) != len(set(backend_keys)):
            raise ValueError(
                f"lane {self.lane!r}: backends must not contain duplicate "
                f"backend_id values, got {backend_keys}"
            )

        if self.locale is EnumBifrostLaneLocale.CLOUD:
            for binding in self.backends:
                if binding.declares_new_backend and binding.tier == LOCAL_TIER:
                    raise ValueError(
                        f"lane {self.lane!r} declares locale "
                        f"{EnumBifrostLaneLocale.CLOUD.value!r} but adds local-tier "
                        f"backend {binding.backend_key!r}: a cloud lane must "
                        "reach no lab endpoint (OMN-17502)."
                    )
            return self

        if not backend_keys:
            raise ValueError(
                f"lane {self.lane!r} declares locale "
                f"{EnumBifrostLaneLocale.LAB.value!r} and must bind at least one "
                "backend. A lab lane with no local backends silently degrades "
                "to the metered ceiling (OMN-16833); a lane that runs off the lab "
                f"network declares locale {EnumBifrostLaneLocale.CLOUD.value!r} "
                "instead (OMN-17502)."
            )
        return self


__all__ = [
    "ModelBifrostLaneOverlay",
]
