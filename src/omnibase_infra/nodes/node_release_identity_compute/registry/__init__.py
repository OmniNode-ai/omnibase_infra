# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry module for NodeReleaseIdentityCompute.

Provides the RegistryInfraReleaseIdentityCompute class for dependency injection
registration and factory methods.

Ticket: OMN-14471
"""

from omnibase_infra.nodes.node_release_identity_compute.registry.registry_infra_release_identity_compute import (
    RegistryInfraReleaseIdentityCompute,
)

__all__: list[str] = ["RegistryInfraReleaseIdentityCompute"]
