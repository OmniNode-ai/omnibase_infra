# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Container binding for HandlerRegistrationStoragePostgres in tests (OMN-22131).

The handler resolves ModelRegistrationStoragePostgresConfig from its container.
``container_binding`` returns a container stub whose ``get_service`` answers
with that config, built from the declared settings file plus the values a test
names.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.models.registration_storage import (
    DECLARED_SETTINGS_PATH,
    ModelRegistrationStoragePostgresConfig,
)


def container_binding(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    auto_create_schema: bool = False,
) -> MagicMock:
    """Return a container stub that resolves the handler's settings."""
    config = ModelRegistrationStoragePostgresConfig.from_file(
        DECLARED_SETTINGS_PATH,
        dsn="",
        host=host,
        port=port,
        password=password,
        auto_create_schema=auto_create_schema,
    ).model_copy(update={"database": database, "user": user})
    container = MagicMock(spec=ModelONEXContainer)
    container.get_service.return_value = config
    return container


__all__ = ["container_binding"]
