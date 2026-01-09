# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Storage Handler Models.

Models for registration storage handler operations.

Note:
    ModelRegistrationRecord, ModelStorageResult, and ModelUpsertResult are
    defined in ``omnibase_infra.nodes.node_registration_storage_effect.models``
    and re-exported here for convenience.
"""

from omnibase_infra.handlers.registration_storage.models.model_delete_registration_request import (
    ModelDeleteRegistrationRequest,
)
from omnibase_infra.handlers.registration_storage.models.model_update_registration_request import (
    ModelUpdateRegistrationRequest,
)

# Import from canonical location
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelRegistrationRecord,
    ModelStorageResult,
    ModelUpsertResult,
)

__all__: list[str] = [
    "ModelDeleteRegistrationRequest",
    "ModelRegistrationRecord",
    "ModelStorageResult",
    "ModelUpdateRegistrationRequest",
    "ModelUpsertResult",
]
