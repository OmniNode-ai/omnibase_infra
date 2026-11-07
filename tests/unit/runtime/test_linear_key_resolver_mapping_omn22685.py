# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22685: the dev lane resolves the Linear key the ticket nodes ask for.

``node_create_ticket`` (and ``node_linear_ticket_write_effect`` through it) resolves the key with
``contract_secret_ref(_CONTRACT_PATH, "LINEAR_API_KEY")``, the literal name from the node's own
``secrets:`` block. The dev lane runs with ``enable_convention_fallback: false``, so an unmapped
literal name resolved to nothing and every lab ticket write over the bus ended
``secret_unavailable`` while the container environment already held the key.

These tests build the resolver the way the container does, from the config the renderer puts in
``docker/runtime-policy.env``, and resolve the name against a synthetic environment value. The
control builds the same resolver from the same config with the mapping removed and must refuse.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from omnibase_infra.errors import SecretResolutionError
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOGICAL_NAME = "LINEAR_API_KEY"
_SYNTHETIC_VALUE = "synthetic-linear-value-for-the-mapping-test"

# Every dev-lane process variable that carries a resolver config.
_DEV_CONFIG_VARIABLES = (
    "DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON",
    "DEV_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON",
    "DEV_RUNTIME_EFFECTS_STORE_SECRET_RESOLVER_CONFIG_JSON",
    "DEV_RUNTIME_WORKER_SECRET_RESOLVER_CONFIG_JSON",
    "DEV_RUNTIME_TENANT_PROJECTION_SECRET_RESOLVER_CONFIG_JSON",
)


def _renderer() -> ModuleType:
    path = _REPO_ROOT / "scripts" / "render_runtime_policy_env.py"
    spec = importlib.util.spec_from_file_location("render_runtime_policy_env", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rendered_config(variable: str) -> dict[str, object]:
    renderer = _renderer()
    env = renderer.render_env(renderer.load_contract())
    assert variable in env, f"the renderer emits no {variable}"
    config = json.loads(env[variable])
    assert isinstance(config, dict)
    return config


@pytest.mark.parametrize("variable", _DEV_CONFIG_VARIABLES)
def test_linear_api_key_resolves_from_every_dev_resolver_config(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_LOGICAL_NAME, _SYNTHETIC_VALUE)
    resolver = SecretResolver(
        config=ModelSecretResolverConfig.model_validate(_rendered_config(variable))
    )

    secret = resolver.get_secret(_LOGICAL_NAME, required=True)

    assert secret is not None
    assert secret.get_secret_value() == _SYNTHETIC_VALUE


@pytest.mark.parametrize("variable", _DEV_CONFIG_VARIABLES)
def test_linear_api_key_refused_when_the_mapping_is_absent(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failing control: the same config without the mapping is the defect."""
    monkeypatch.setenv(_LOGICAL_NAME, _SYNTHETIC_VALUE)
    config = _rendered_config(variable)
    mappings = config["mappings"]
    assert isinstance(mappings, list)
    config["mappings"] = [m for m in mappings if m["logical_name"] != _LOGICAL_NAME]
    assert len(config["mappings"]) == len(mappings) - 1, "the mapping was not there"
    resolver = SecretResolver(config=ModelSecretResolverConfig.model_validate(config))

    with pytest.raises(SecretResolutionError):
        resolver.get_secret(_LOGICAL_NAME, required=True)


def test_the_mapping_reads_the_environment_variable_of_the_same_name() -> None:
    config = _rendered_config("DEV_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON")
    mappings = config["mappings"]
    assert isinstance(mappings, list)
    [mapping] = [m for m in mappings if m["logical_name"] == _LOGICAL_NAME]
    assert mapping["source"] == {"source_type": "env", "source_path": _LOGICAL_NAME}
