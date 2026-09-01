# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed first-effect publish composition tests."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.first_effect_ledger import (
    FirstEffectPublishCompositionError,
    PostgresFirstEffectLedger,
    require_transactional_first_effect_outbox,
)


@pytest.mark.unit
def test_observation_ledger_cannot_be_composed_as_publish_authority() -> None:
    async def pool_factory() -> object:
        raise AssertionError("the composition guard must not perform I/O")

    ledger = PostgresFirstEffectLedger(pool_factory=pool_factory)  # type: ignore[arg-type]

    with pytest.raises(FirstEffectPublishCompositionError, match="transactional"):
        require_transactional_first_effect_outbox(ledger)


@pytest.mark.unit
def test_observation_ledger_exposes_no_begin_publishing_permission_method() -> None:
    assert not hasattr(PostgresFirstEffectLedger, "begin_publishing")
