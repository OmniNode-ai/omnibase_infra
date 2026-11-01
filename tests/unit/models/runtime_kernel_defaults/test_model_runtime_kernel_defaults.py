# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime kernel defaults load from the declared file and refuse omissions (OMN-21623)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.runtime_kernel_defaults import (
    ModelRuntimeKernelDefaults,
    load_runtime_kernel_defaults,
)
from omnibase_infra.models.runtime_kernel_defaults import (
    model_runtime_kernel_defaults as settings_module,
)
from omnibase_infra.runtime import service_kernel

pytestmark = pytest.mark.unit

_DECLARED_PATH = Path(settings_module.__file__).with_name(
    "runtime_kernel_defaults.yaml"
)


def _declared_block() -> dict[str, object]:
    document = yaml.safe_load(_DECLARED_PATH.read_text(encoding="utf-8"))
    block = document["settings"]
    assert isinstance(block, dict)
    return block


def test_loaded_settings_equal_the_declared_file() -> None:
    assert load_runtime_kernel_defaults() == (
        ModelRuntimeKernelDefaults.model_validate(_declared_block())
    )


@pytest.mark.parametrize("key", sorted(_declared_block()))
def test_an_omitted_value_fails_to_load(key: str) -> None:
    block = _declared_block()
    del block[key]

    with pytest.raises(ValidationError):
        ModelRuntimeKernelDefaults.model_validate(block)


@pytest.mark.parametrize("key", ["default_contracts_dir", "workspace_contracts_dir"])
def test_an_empty_value_fails_to_load(key: str) -> None:
    block = {**_declared_block(), key: ""}

    with pytest.raises(ValidationError):
        ModelRuntimeKernelDefaults.model_validate(block)


def test_an_empty_endpoint_list_fails_to_load() -> None:
    block = {**_declared_block(), "llm_endpoint_env_keys": []}

    with pytest.raises(ValidationError):
        ModelRuntimeKernelDefaults.model_validate(block)


def test_endpoint_names_come_from_the_variable_names() -> None:
    defaults = ModelRuntimeKernelDefaults.model_validate(
        {
            **_declared_block(),
            "llm_endpoint_env_keys": ["LLM_ALPHA_URL", "LLM_BETA_GAMMA_URL"],
        }
    )

    assert defaults.llm_endpoint_names == {
        "LLM_ALPHA_URL": "alpha",
        "LLM_BETA_GAMMA_URL": "beta-gamma",
    }


def test_an_unknown_key_fails_to_load() -> None:
    block = {**_declared_block(), "invented_key": "x"}

    with pytest.raises(ValidationError):
        ModelRuntimeKernelDefaults.model_validate(block)


def test_a_file_without_a_settings_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "runtime_kernel_defaults.yaml"
    path.write_text("other: {}\n", encoding="utf-8")

    with pytest.raises(ProtocolConfigurationError):
        ModelRuntimeKernelDefaults.from_file(path)


def test_the_kernel_reads_its_defaults_from_the_declared_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(service_kernel.ENV_CONTRACTS_DIR, raising=False)
    settings = load_runtime_kernel_defaults()

    assert service_kernel._get_contracts_dir() == Path(settings.default_contracts_dir)
    assert (
        Path(settings.workspace_contracts_dir)
        == service_kernel.WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH
    )
