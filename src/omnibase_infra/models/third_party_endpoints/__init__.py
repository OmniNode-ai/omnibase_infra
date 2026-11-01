# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Third-party endpoints declared in ``third_party_endpoints.yaml`` (OMN-21623)."""

from omnibase_infra.models.third_party_endpoints.model_third_party_endpoints import (
    ModelThirdPartyEndpoints,
    load_third_party_endpoints,
)

__all__ = ["ModelThirdPartyEndpoints", "load_third_party_endpoints"]
