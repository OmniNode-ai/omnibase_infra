# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reward binder's EFFECT contract permits handler I/O (OMN-21784)."""

from pathlib import Path

import pytest

from omnibase_core.validation.validator_no_io_outside_effects import (
    ValidatorNoIoOutsideEffects,
)


@pytest.mark.integration
def test_reward_binder_contract_allows_effect_io() -> None:
    node_dir = Path("src/omnibase_infra/nodes/node_reward_binder_effect")
    result = ValidatorNoIoOutsideEffects().validate(
        [node_dir / "contract.yaml", node_dir / "handlers/handler_reward_binder.py"]
    )

    assert result.is_valid, result.issues
