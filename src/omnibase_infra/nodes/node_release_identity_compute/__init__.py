# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Release Identity Compute — fresh-deploy fitness gate node.

Canonical CONTRACT+NODE+HANDLER replacement for the freestanding imperative script
``scripts/check_release_identity.py`` (legacy, OMN-13412). Enforces that packaged
source (``src/**``) may not merge onto an already-published version without a bump.

Architecture:
    - NodeReleaseIdentityCompute is a declarative shell (no custom logic).
    - HandlerReleaseIdentity contains the pure version-ahead / src-change gate.
    - contract.yaml defines behavior via handler_routing.

The git/pyproject I/O is performed by the thin CLI collector/shim at
``scripts/check_release_identity.py`` and handed to the handler as a typed request,
so the handler is pure and deterministic.

Ticket: OMN-14471
"""

from omnibase_infra.nodes.node_release_identity_compute.handlers import (
    HandlerReleaseIdentity,
)
from omnibase_infra.nodes.node_release_identity_compute.models import (
    ModelReleaseIdentityDecision,
    ModelReleaseIdentityRequest,
)
from omnibase_infra.nodes.node_release_identity_compute.node import (
    NodeReleaseIdentityCompute,
)
from omnibase_infra.nodes.node_release_identity_compute.registry import (
    RegistryInfraReleaseIdentityCompute,
)

__all__: list[str] = [
    "HandlerReleaseIdentity",
    "ModelReleaseIdentityDecision",
    "ModelReleaseIdentityRequest",
    "NodeReleaseIdentityCompute",
    "RegistryInfraReleaseIdentityCompute",
]
