# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Public resource port for registration's pre-dispatch effect backend."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from omnibase_infra.models.registration.model_registration_bootstrap_result import Pool


@runtime_checkable
class ProtocolRegistrationBootstrapResources(Protocol):
    """Transport operations supplied by the registered registration backend."""

    async def create_pool(self, postgres_dsn: str) -> Pool: ...

    def read_declarations(self, path: Path) -> dict[str, object]: ...

    def topic_router(self, path: Path) -> dict[str, str]: ...

    def read_schema(self, path: Path) -> str: ...

    def is_duplicate_schema_error(self, error: Exception) -> bool: ...
