# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Storage Handler Models.

Handler-specific request models for registration storage operations.

Note:
    Core storage models (ModelRegistrationRecord, ModelStorageResult,
    ModelUpsertResult) are defined in the canonical location:
    ``omnibase_infra.nodes.node_registration_storage_effect.models``
"""

from omnibase_infra.handlers.registration_storage.models.model_delete_registration_request import (
    ModelDeleteRegistrationRequest,
)
from omnibase_infra.handlers.registration_storage.models.model_update_registration_request import (
    ModelUpdateRegistrationRequest,
)

__all__: list[str] = [
    "ModelDeleteRegistrationRequest",
    "ModelUpdateRegistrationRequest",
]
