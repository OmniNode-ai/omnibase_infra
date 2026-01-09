# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Storage Handler Models.

Models for registration storage handler operations.
"""

from omnibase_infra.handlers.registration_storage.models.model_delete_registration_request import (
    ModelDeleteRegistrationRequest,
)
from omnibase_infra.handlers.registration_storage.models.model_registration_record import (
    ModelRegistrationRecord,
)
from omnibase_infra.handlers.registration_storage.models.model_storage_result import (
    ModelStorageResult,
)
from omnibase_infra.handlers.registration_storage.models.model_update_registration_request import (
    ModelUpdateRegistrationRequest,
)
from omnibase_infra.handlers.registration_storage.models.model_upsert_result import (
    ModelUpsertResult,
)

__all__: list[str] = [
    "ModelDeleteRegistrationRequest",
    "ModelRegistrationRecord",
    "ModelStorageResult",
    "ModelUpdateRegistrationRequest",
    "ModelUpsertResult",
]
