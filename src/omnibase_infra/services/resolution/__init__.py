# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Resolution event ledger services and contract YAML integration models.

Provides:
- Publisher for resolution decision audit events (Phase 6, OMN-2895)
- Local configuration models for tiered resolution contract YAML (Phase 7, OMN-2896)

Related:
    - OMN-2895: Resolution Event Ledger (Phase 6 of OMN-2897 epic)
    - OMN-2896: Contract YAML Integration (Phase 7 of OMN-2897 epic)
    - ModelResolutionEvent: omnibase_core model (PR #575, not yet released)
    - ModelTieredResolutionConfig / ModelTrustDomainConfig: omnibase_core (PR #576)
"""

from __future__ import annotations

from omnibase_infra.services.resolution.model_resolution_event_local import (
    ModelResolutionEventLocal,
)
from omnibase_infra.services.resolution.model_resolution_proof_local import (
    ModelResolutionProofLocal,
)
from omnibase_infra.services.resolution.model_tier_attempt_local import (
    ModelTierAttemptLocal,
)
from omnibase_infra.services.resolution.model_tiered_resolution_config_local import (
    VALID_CLASSIFICATIONS,
    VALID_PROOF_TYPES,
    VALID_RESOLUTION_TIERS,
    ModelTieredResolutionConfigLocal,
)
from omnibase_infra.services.resolution.model_trust_domain_config_local import (
    ModelTrustDomainConfigLocal,
)
from omnibase_infra.services.resolution.service_resolution_event_publisher import (
    ServiceResolutionEventPublisher,
)

__all__: list[str] = [
    "ModelResolutionEventLocal",
    "ModelResolutionProofLocal",
    "ModelTierAttemptLocal",
    "ModelTieredResolutionConfigLocal",
    "ModelTrustDomainConfigLocal",
    "ServiceResolutionEventPublisher",
    "VALID_CLASSIFICATIONS",
    "VALID_PROOF_TYPES",
    "VALID_RESOLUTION_TIERS",
]
