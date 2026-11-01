# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The endpoint defaults load from the declared file and refuse omissions (OMN-21690)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.endpoint_defaults import (
    ModelEndpointDefaults,
    load_endpoint_defaults,
)
from omnibase_infra.models.endpoint_defaults import (
    model_endpoint_defaults as defaults_module,
)

pytestmark = pytest.mark.unit

_DECLARED_PATH = Path(defaults_module.__file__).with_name("endpoint_defaults.yaml")


def _declared_block() -> dict[str, object]:
    document = yaml.safe_load(_DECLARED_PATH.read_text(encoding="utf-8"))
    block = document["settings"]
    assert isinstance(block, dict)
    return block


def test_loaded_defaults_equal_the_declared_file() -> None:
    defaults = load_endpoint_defaults()

    assert defaults == ModelEndpointDefaults.model_validate(_declared_block())
    assert defaults.loopback_address != defaults.bind_all_address
    assert defaults.loopback_address in defaults.loopback_host_names
    assert defaults.local_host in defaults.loopback_host_names


def test_a_server_bound_to_the_bind_all_address_is_reached_on_loopback() -> None:
    defaults = load_endpoint_defaults()

    assert (
        defaults.connect_address(defaults.bind_all_address) == defaults.loopback_address
    )
    assert defaults.connect_address("db.example.test") == "db.example.test"


def test_only_the_declared_loopback_host_names_are_loopback() -> None:
    defaults = load_endpoint_defaults()

    assert all(defaults.is_loopback(name) for name in defaults.loopback_host_names)
    assert not defaults.is_loopback(defaults.bind_all_address)
    assert not defaults.is_loopback("db.example.test")


@pytest.mark.parametrize("key", sorted(_declared_block()))
def test_an_omitted_value_fails_to_load(key: str) -> None:
    block = _declared_block()
    del block[key]

    with pytest.raises(ValidationError):
        ModelEndpointDefaults.model_validate(block)


def test_an_unknown_key_fails_to_load() -> None:
    block = {**_declared_block(), "invented_key": "x"}

    with pytest.raises(ValidationError):
        ModelEndpointDefaults.model_validate(block)


def test_an_out_of_range_port_fails_to_load() -> None:
    block = {**_declared_block(), "runtime_health_port": 70000}

    with pytest.raises(ValidationError):
        ModelEndpointDefaults.model_validate(block)


def test_a_file_without_a_settings_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "endpoint_defaults.yaml"
    path.write_text("other: {}\n", encoding="utf-8")

    with pytest.raises(ProtocolConfigurationError):
        ModelEndpointDefaults.from_file(path)
