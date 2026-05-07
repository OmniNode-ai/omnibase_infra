# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Secret store implementations of ``ProtocolSecretStore``.

Each module here provides a ``ProtocolSecretStore`` adapter for a specific
backend (env vars, Infisical, etc.). These wrappers are *composition* over
existing adapters/SDKs — they do not mutate or replace the underlying
clients.

Allowed importers of ``omnibase_infra.adapters._internal``:
    handlers, tests, and this ``secret_stores`` package (allowlist enforced
    by ``validator_no_direct_adapter`` — see OMN-10557 for the
    architectural amendment that admitted ``secret_stores`` as a
    cross-cutting wrapper layer).
"""

from __future__ import annotations

from omnibase_infra.secret_stores.adapter_env_secret_store import (
    AdapterEnvSecretStore,
)
from omnibase_infra.secret_stores.infisical_secret_store import InfisicalSecretStore

__all__: list[str] = ["AdapterEnvSecretStore", "InfisicalSecretStore"]
