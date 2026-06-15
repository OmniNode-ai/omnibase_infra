# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Health check for runtime config provenance (OMN-12958).

Promotes config provenance (deployed-volume sha vs packaged-source sha) to a
health-check component so volume/config drift pages an operator and surfaces in
proof packets instead of silently degrading delegation routing.

* deployed contract absent  -> unhealthy (runtime has no config to load)
* deployed sha != source sha -> degraded (drift: re-seed required)
* source absent              -> degraded (cannot prove provenance)
* deployed sha == source sha -> healthy
"""

from __future__ import annotations

from omnibase_core.types import JsonType
from omnibase_infra.runtime.config_provenance import ModelConfigProvenance
from omnibase_infra.runtime.models.model_component_health import (
    ModelComponentHealth,
)

_COMPONENT_NAME = "config_provenance"


def check_config_provenance_health(
    provenance: ModelConfigProvenance,
) -> ModelComponentHealth:
    """Classify config provenance into a component-health status.

    Args:
        provenance: Computed provenance for a deployed contract.

    Returns:
        ModelComponentHealth carrying the path + sha details so the health
        endpoint and proof packets expose provenance, not just a status bit.
    """
    details: dict[str, JsonType] = {
        "config_name": provenance.config_name,
        "deployed_path": provenance.deployed_path,
        "deployed_sha256": provenance.deployed_sha256,
        "source_path": provenance.source_path,
        "source_sha256": provenance.source_sha256,
        "has_drifted": provenance.has_drifted,
    }

    if not provenance.deployed_present:
        return ModelComponentHealth.unhealthy(
            name=_COMPONENT_NAME,
            error=(
                f"deployed config absent at {provenance.deployed_path} "
                f"({provenance.config_name})"
            ),
            details=details,
        )

    if not provenance.source_present:
        return ModelComponentHealth.degraded(
            name=_COMPONENT_NAME,
            error=(
                f"packaged source absent at {provenance.source_path}; "
                "cannot prove config provenance"
            ),
            details=details,
        )

    if provenance.has_drifted:
        return ModelComponentHealth.degraded(
            name=_COMPONENT_NAME,
            error=(
                f"volume config drifted from packaged source for "
                f"{provenance.config_name}: deployed_sha256="
                f"{provenance.deployed_sha256} source_sha256="
                f"{provenance.source_sha256}; re-seed required"
            ),
            details=details,
        )

    return ModelComponentHealth.healthy(
        name=_COMPONENT_NAME,
        details=details,
    )


__all__: list[str] = ["check_config_provenance_health"]
