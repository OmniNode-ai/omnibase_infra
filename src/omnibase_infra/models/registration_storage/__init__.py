# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Settings of the PostgreSQL registration storage handler (OMN-22131).

Importing this package performs no settings file I/O.
"""

from omnibase_infra.models.registration_storage.model_registration_storage_postgres_config import (
    DECLARED_SETTINGS_PATH,
    ModelRegistrationStoragePostgresConfig,
)

__all__ = ["DECLARED_SETTINGS_PATH", "ModelRegistrationStoragePostgresConfig"]
