# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test utilities ``omnibase_infra`` ships for downstream tests (OMN-10559).

These helpers are deliberately *not* under ``tests/`` because consumer
repos (omnimarket, etc.) import them from their own conftests. They are
shipped in the wheel — acceptable for an internal platform package.

Add lightweight fakes / fixtures here. Heavyweight fixtures that depend
on infrastructure (Postgres, Kafka) belong in ``omnibase_infra.testing``
instead.
"""

from __future__ import annotations

from omnibase_infra.test_utils.fake_postgres_adapter import FakePostgresAdapter
from omnibase_infra.test_utils.fake_secret_store import FakeSecretStore
from omnibase_infra.test_utils.fake_valkey_client import FakeValkeyClient

__all__: list[str] = ["FakePostgresAdapter", "FakeSecretStore", "FakeValkeyClient"]
