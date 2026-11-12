# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every logical secret a node contract names resolves on the dev lane (OMN-23903).

Node contracts name the secrets they read by logical name, for example
``config.analytics_dsn_secret: "omnidash.analytics.db_url"`` in omnimarket's
tenant read nodes. The dev lane resolves a logical name only when its rendered
resolver config maps it: the profile sets ``enable_convention_fallback: false``,
so an unmapped name resolves to nothing and the node answers ``not_configured``
while the container env already holds the value. That is how delegation
scenario R8 failed on 2026-11-12 (totals_read_before, "delegation analytics
source not configured").

This repository may not import omnimarket (OMN-19998), so the omnimarket side of
the seam is the literal table below. omnimarket pins its half in its own tests
(``tests/nodes/node_tenant_savings_read_effect/test_mount_contract.py`` asserts
the contract names the same logical secret). A new contract-named secret that a
dev node reads goes in this table and in the dev profile's
``secret_resolver_mappings`` in the same change.

The check renders every dev resolver config the policy env carries through the
real renderer the runtime entrypoint runs, then asks the real ``SecretResolver``
for each logical name with its declared environment source set, the same call
the node makes (``get_secret_async(name, required=False)``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.render_secret_resolver_config import (
    render_secret_resolver_config,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"

pytestmark = pytest.mark.unit

# Logical secret name -> the node contracts (repo-relative) that name it and run
# on the dev lane. Keep in step with the contracts: a name missing here is a
# name no test checks.
CONTRACT_NAMED_DEV_SECRETS: dict[str, tuple[str, ...]] = {
    "omnidash.analytics.db_url": (
        "omnimarket:src/omnimarket/nodes/node_tenant_savings_read_effect/contract.yaml",
        "omnimarket:src/omnimarket/nodes/node_tenant_delegations_read_effect/contract.yaml",
    ),
}


def _dev_resolver_configs() -> dict[str, str]:
    """Every dev resolver config JSON the policy renderer emits, by env name."""
    from omnibase_infra.runtime.models.model_runtime_policy_contract import (
        ModelRuntimePolicyContract,
    )
    from scripts.render_runtime_policy_env import render_env

    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    return {
        name: value
        for name, value in render_env(contract).items()
        if name.startswith("DEV_RUNTIME_")
        and name.endswith("_SECRET_RESOLVER_CONFIG_JSON")
        and value
    }


def test_dev_policy_renders_resolver_configs() -> None:
    # Guard the guard: an empty set would turn the check below into a no-op.
    assert _dev_resolver_configs(), "the dev profile renders no resolver config"


@pytest.mark.parametrize("logical_name", sorted(CONTRACT_NAMED_DEV_SECRETS))
def test_every_contract_named_secret_resolves_on_every_dev_resolver(
    logical_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared_by = ", ".join(CONTRACT_NAMED_DEV_SECRETS[logical_name])
    for env_name, config_json in sorted(_dev_resolver_configs().items()):
        target = tmp_path / f"{env_name}.yaml"
        rendered = render_secret_resolver_config(
            target_path=target,
            environ={"ONEX_SECRET_RESOLVER_CONFIG_JSON": config_json},
        )
        assert rendered == target
        config = ModelSecretResolverConfig.model_validate(
            yaml.safe_load(target.read_text(encoding="utf-8"))
        )
        mapping = next(
            (m for m in config.mappings if m.logical_name == logical_name), None
        )
        assert mapping is not None, (
            f"{env_name} does not map {logical_name!r}, which {declared_by} "
            "names; the dev profile's secret_resolver_mappings must map it"
        )
        assert mapping.source.source_type == "env", (
            f"{env_name} maps {logical_name!r} to {mapping.source.source_type}; "
            "the dev lane delivers it through the container environment"
        )
        monkeypatch.setenv(mapping.source.source_path, "placeholder-not-a-secret")
        resolver = SecretResolver(config=config)
        value = asyncio.run(resolver.get_secret_async(logical_name, required=False))
        assert value is not None, (
            f"{env_name}: {logical_name!r} is mapped but does not resolve"
        )
        assert value.get_secret_value() == "placeholder-not-a-secret"
