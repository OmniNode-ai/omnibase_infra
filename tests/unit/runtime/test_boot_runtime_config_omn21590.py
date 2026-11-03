# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot limits are validated contract data, including explicit overlay values."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.handlers.handler_package_announcer import load_boot_runtime_config


def test_boot_config_declares_a_positive_settle_timeout() -> None:
    config = load_boot_runtime_config()
    assert config.catalog_settle_timeout_seconds > 0


def test_explicit_contract_overlay_changes_the_bound(tmp_path: Path) -> None:
    path = tmp_path / "overlay.yaml"
    values = {
        "catalog_settle_timeout_seconds": 0.125,
    }
    path.write_text(yaml.safe_dump({"runtime_config": values}))
    assert load_boot_runtime_config(path).model_dump() == values


@pytest.mark.parametrize(
    "key",
    [
        "catalog_settle_timeout_seconds",
    ],
)
def test_invalid_contract_bound_is_refused(tmp_path: Path, key: str) -> None:
    path = tmp_path / "overlay.yaml"
    values = load_boot_runtime_config().model_dump()
    values[key] = 0
    path.write_text(yaml.safe_dump({"runtime_config": values}))
    with pytest.raises(ValidationError):
        load_boot_runtime_config(path)
