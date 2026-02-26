# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Models for the Contract Resolver Bridge node.

I/O models are sourced directly from omnibase_core (OMN-2754).
This package re-exports them for convenient access via the bridge node.
"""

from omnibase_core.models.nodes.contract_resolve import (
    ModelContractResolveInput,
    ModelContractResolveOptions,
    ModelContractResolveOutput,
    ModelOverlayRef,
    ModelResolverBuild,
)

__all__ = [
    "ModelContractResolveInput",
    "ModelContractResolveOptions",
    "ModelContractResolveOutput",
    "ModelOverlayRef",
    "ModelResolverBuild",
]
