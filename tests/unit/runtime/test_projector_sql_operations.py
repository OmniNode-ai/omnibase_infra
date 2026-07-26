# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.mixins.mixin_projector_sql_operations import (
    MixinProjectorSqlOperations,
)


def test_normalize_value_parses_timestamptz_string() -> None:
    mixin = MixinProjectorSqlOperations()

    result = mixin.normalize_value(
        "2026-07-09T19:25:00Z",
        "emitted_at",
        column_type="TIMESTAMPTZ",
    )

    assert result == datetime(2026, 7, 9, 19, 25, tzinfo=UTC)


def test_normalize_value_wraps_malformed_timestamptz() -> None:
    mixin = MixinProjectorSqlOperations()

    with pytest.raises(ProtocolConfigurationError) as exc_info:
        mixin.normalize_value(
            "not-a-timestamp",
            "emitted_at",
            column_type="TIMESTAMPTZ",
        )

    err = exc_info.value
    assert "Invalid TIMESTAMPTZ value" in str(err)
    assert err.__cause__ is not None
    assert type(err.__cause__).__name__ == "ValueError"
