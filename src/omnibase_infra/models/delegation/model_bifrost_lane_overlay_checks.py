# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-binding checks of a Bifrost lane overlay (OMN-15807, OMN-17502, OMN-22756).

Moved out of ``ModelBifrostLaneOverlay`` with the check that a lane has at most
one serving backend per typed media capability (OMN-22756), so the overlay model
stays declarative. Every ``ValueError`` raised here surfaces from the overlay as
a pydantic ``ValidationError``, exactly as it did from the model's own validator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from omnibase_infra.models.delegation.model_bifrost_lane_media_capability_resolver import (
    check_one_serving_backend_per_capability,
)
from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    LOCAL_TIER,
)

if TYPE_CHECKING:
    from omnibase_infra.models.delegation.model_bifrost_lane_endpoint_capacity import (
        ModelBifrostLaneEndpointCapacity,
    )
    from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
        ModelBifrostLaneBackendBinding,
    )


def validate_overlay_bindings(
    lane: str,
    locale: EnumBifrostLaneLocale,
    backends: Sequence[ModelBifrostLaneBackendBinding],
    endpoint_capacity: Sequence[ModelBifrostLaneEndpointCapacity],
) -> None:
    """Refuse an overlay whose bindings disagree with each other or its locale."""
    backend_keys = [binding.backend_key for binding in backends]
    if len(backend_keys) != len(set(backend_keys)):
        raise ValueError(
            f"lane {lane!r}: backends must not contain duplicate "
            f"backend_id values, got {backend_keys}"
        )

    capacity_endpoints = [capacity.endpoint for capacity in endpoint_capacity]
    if len(capacity_endpoints) != len(set(capacity_endpoints)):
        raise ValueError(
            f"lane {lane!r}: endpoint_capacity must not declare an "
            f"endpoint twice, got {capacity_endpoints}"
        )

    check_one_serving_backend_per_capability(lane, backends)

    if locale is EnumBifrostLaneLocale.CLOUD:
        for binding in backends:
            if binding.declares_new_backend and binding.tier == LOCAL_TIER:
                raise ValueError(
                    f"lane {lane!r} declares locale "
                    f"{EnumBifrostLaneLocale.CLOUD.value!r} but adds local-tier "
                    f"backend {binding.backend_key!r}: a cloud lane must "
                    "reach no lab endpoint (OMN-17502)."
                )
        return

    if not backend_keys:
        raise ValueError(
            f"lane {lane!r} declares locale "
            f"{EnumBifrostLaneLocale.LAB.value!r} and must bind at least one "
            "backend. A lab lane with no local backends silently degrades "
            "to the metered ceiling (OMN-16833); a lane that runs off the lab "
            f"network declares locale {EnumBifrostLaneLocale.CLOUD.value!r} "
            "instead (OMN-17502)."
        )


__all__ = ["validate_overlay_bindings"]
