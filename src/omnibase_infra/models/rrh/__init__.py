# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Release Readiness Handshake (RRH) shared models.

These models are shared across the three RRH nodes:

- ``node_rrh_emit_effect``: Collects environment data
- ``node_rrh_validate_compute``: Validates against rules and profiles
- ``node_rrh_storage_effect``: Writes result artifacts

The RRH result composes existing shared primitives
(``ModelRuleCheckResult``, ``EnumVerdict``) so dashboards can consume
both architecture-validation and RRH results uniformly.
"""

from omnibase_infra.models.rrh.model_rrh_environment_data import (
    ModelRRHEnvironmentData,
)
from omnibase_infra.models.rrh.model_rrh_profile import ModelRRHProfile
from omnibase_infra.models.rrh.model_rrh_repo_state import ModelRRHRepoState
from omnibase_infra.models.rrh.model_rrh_result import ModelRRHResult
from omnibase_infra.models.rrh.model_rrh_runtime_target import ModelRRHRuntimeTarget
from omnibase_infra.models.rrh.model_rrh_toolchain_versions import (
    ModelRRHToolchainVersions,
)

__all__: list[str] = [
    "ModelRRHEnvironmentData",
    "ModelRRHProfile",
    "ModelRRHRepoState",
    "ModelRRHResult",
    "ModelRRHRuntimeTarget",
    "ModelRRHToolchainVersions",
]
