# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Secret store implementations of ``ProtocolSecretStore``.

Each module here provides a ``ProtocolSecretStore`` adapter for a specific
backend (env vars, Infisical, etc.). These wrappers are *composition* over
existing adapters/SDKs — they do not mutate or replace the underlying
clients.
"""

from __future__ import annotations

from omnibase_infra.secret_stores.adapter_env_secret_store import (
    AdapterEnvSecretStore,
)

__all__: list[str] = ["AdapterEnvSecretStore"]
