# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The network binding settings load from the declared file and refuse omissions (OMN-21623)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.network_binding import (
    ModelNetworkBindingSettings,
    load_network_binding_settings,
)
from omnibase_infra.models.network_binding import (
    model_network_binding_settings as settings_module,
)

pytestmark = pytest.mark.unit

_DECLARED_PATH = Path(settings_module.__file__).with_name(
    "network_binding_settings.yaml"
)


def _declared_block() -> dict[str, object]:
    document = yaml.safe_load(_DECLARED_PATH.read_text(encoding="utf-8"))
    block = document["settings"]
    assert isinstance(block, dict)
    return block


def test_loaded_settings_equal_the_declared_file() -> None:
    settings = load_network_binding_settings()

    assert settings == ModelNetworkBindingSettings.model_validate(_declared_block())
    assert settings.loopback_address != settings.bind_all_address
    assert settings.tcp_port_min < settings.tcp_port_max


@pytest.mark.parametrize("key", sorted(_declared_block()))
def test_an_omitted_value_fails_to_load(key: str) -> None:
    block = _declared_block()
    del block[key]

    with pytest.raises(ValidationError):
        ModelNetworkBindingSettings.model_validate(block)


def test_an_unknown_key_fails_to_load() -> None:
    block = {**_declared_block(), "invented_key": "x"}

    with pytest.raises(ValidationError):
        ModelNetworkBindingSettings.model_validate(block)


def test_a_file_without_a_settings_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "network_binding_settings.yaml"
    path.write_text("other: {}\n", encoding="utf-8")

    with pytest.raises(ProtocolConfigurationError):
        ModelNetworkBindingSettings.from_file(path)
