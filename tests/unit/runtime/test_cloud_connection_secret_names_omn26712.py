# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26712 -- cloud mode declares the gateway and tenant identity connection names.

The list the cloud owner stores secrets for is ``ModelCloudSecretNames.REQUIRED``. In cloud
mode a rendered resolver config that omits one refuses to boot with ``CLOUD_SECRET_NAMES_MISSING``
and the missing name; outside cloud mode nothing changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.runtime.model_cloud_secret_names import (
    ModelCloudSecretNames,
)
from omnibase_infra.runtime.entrypoint_preflight import render_resolver_config
from omnibase_infra.runtime.render_secret_resolver_config import (
    render_secret_resolver_config,
)

pytestmark = [pytest.mark.unit]


def _mapping(name: str) -> dict[str, object]:
    return {
        "logical_name": name,
        "source": {"source_type": "infisical", "source_path": f"/dev/{name}"},
    }


def _env(names: list[str], *, cloud: bool = True) -> dict[str, str]:
    env = {
        "ONEX_SECRET_RESOLVER_CONFIG_JSON": json.dumps(
            {
                "enable_convention_fallback": False,
                "mappings": [_mapping(n) for n in names],
            }
        )
    }
    if cloud:
        env[ModelCloudSecretNames.MODE_ENV_VAR] = "1"
    return env


def test_declared_names_are_the_gateway_and_tenant_identity_connections() -> None:
    assert ModelCloudSecretNames.REQUIRED == (
        "gateway.database.dsn",
        "database.tenant_identity_writer.dsn",
    )


def test_cloud_mode_with_every_name_renders(tmp_path: Path) -> None:
    target = tmp_path / "resolver.yaml"
    rendered = render_secret_resolver_config(
        target_path=target, environ=_env(list(ModelCloudSecretNames.REQUIRED))
    )
    assert rendered == target


@pytest.mark.parametrize("missing", ModelCloudSecretNames.REQUIRED)
def test_cloud_mode_refuses_to_start_when_a_name_is_missing(
    tmp_path: Path, missing: str
) -> None:
    present = [n for n in ModelCloudSecretNames.REQUIRED if n != missing]
    with pytest.raises(ProtocolConfigurationError) as raised:
        render_secret_resolver_config(
            target_path=tmp_path / "resolver.yaml", environ=_env(present)
        )
    assert "CLOUD_SECRET_NAMES_MISSING" in str(raised.value)
    assert missing in str(raised.value)


def test_outside_cloud_mode_a_missing_name_is_not_refused(tmp_path: Path) -> None:
    rendered = render_secret_resolver_config(
        target_path=tmp_path / "resolver.yaml", environ=_env([], cloud=False)
    )
    assert rendered is not None


def test_cloud_mode_never_skips_rendering_in_the_boot_preflight() -> None:
    # No ONEX_SECRET_RESOLVER_CONFIG_PATH and no source: the pre-OMN-26712 preflight returned 0.
    with pytest.raises(
        ProtocolConfigurationError, match="Secret resolver render requires"
    ):
        render_resolver_config({ModelCloudSecretNames.MODE_ENV_VAR: "1"})
    assert render_resolver_config({}) == 0
