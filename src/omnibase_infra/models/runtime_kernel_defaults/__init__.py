# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Models for the contract-declared runtime kernel defaults (OMN-21623)."""

from omnibase_infra.models.runtime_kernel_defaults.model_runtime_kernel_defaults import (
    ModelRuntimeKernelDefaults,
    load_runtime_kernel_defaults,
)

__all__ = [
    "ModelRuntimeKernelDefaults",
    "load_runtime_kernel_defaults",
]
