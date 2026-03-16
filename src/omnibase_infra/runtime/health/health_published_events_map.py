# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Health check for the published_events topic router map.

Promotes the startup INFO/WARNING logs (OMN-5153/OMN-5154) to a health-check
component so that an empty or missing map pages and blocks rollouts instead of
silently degrading event routing.

OMN-5159
"""

from __future__ import annotations

from omnibase_infra.runtime.models.model_component_health import (
    ModelComponentHealth,
)


def check_published_events_map_health(
    published_events_map: dict[str, str] | None,
    contract_path: str | None = None,
) -> ModelComponentHealth:
    """Check whether the published_events topic router map is populated.

    Args:
        published_events_map: The topic router dict mapping event class names
            to Kafka topics (built from contract.yaml published_events).
        contract_path: Optional path to the contract.yaml file for diagnostic
            messages.

    Returns:
        ModelComponentHealth indicating healthy (map has entries) or unhealthy
        (map is None or empty).
    """
    if not published_events_map:
        error_msg = "published_events_map is empty or not loaded"
        if contract_path:
            error_msg += f" (contract: {contract_path})"
        return ModelComponentHealth.unhealthy(
            name="published_events_map",
            error=error_msg,
        )

    return ModelComponentHealth.healthy(
        name="published_events_map",
        details={"entry_count": len(published_events_map)},
    )


__all__: list[str] = ["check_published_events_map_health"]
