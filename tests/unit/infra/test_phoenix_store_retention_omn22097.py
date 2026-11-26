# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Phoenix store location and retention default (OMN-22097).

Measured on the dev lane Phoenix: its sqlite file held about 71.7 million
traces (63.7 GB), and even an index-only ``count(*)`` over ``traces`` took 18 s,
so the project page and the GraphQL ``traceCount`` hung past 60 s. The old file
stays in place, untouched, at the volume root. The service writes a new store in
a subdirectory of the same volume and expires spans after a declared number of
days, so the store stays small enough to query.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_VOLUME_ROOT = "/data"
_RETENTION_VAR = "PHOENIX_DEFAULT_RETENTION_POLICY_DAYS"


def _catalog_env() -> dict[str, str]:
    service = yaml.safe_load(
        (_ROOT / "docker/catalog/services/phoenix.yaml").read_text()
    )
    return {k: str(v) for k, v in service["hardcoded_env"].items()}


def _infra_compose_env() -> dict[str, str]:
    compose = yaml.safe_load((_ROOT / "docker/docker-compose.infra.yml").read_text())
    return {k: str(v) for k, v in compose["services"]["phoenix"]["environment"].items()}


@pytest.mark.parametrize("env_of", [_catalog_env, _infra_compose_env])
def test_working_dir_is_below_the_volume_root(
    env_of: Callable[[], dict[str, str]],
) -> None:
    """A new store below the volume root leaves the 63.7 GB file unread."""
    working_dir = env_of()["PHOENIX_WORKING_DIR"]
    assert working_dir.startswith(_VOLUME_ROOT + "/")


@pytest.mark.parametrize("env_of", [_catalog_env, _infra_compose_env])
def test_retention_default_is_days_not_zero(
    env_of: Callable[[], dict[str, str]],
) -> None:
    days = int(env_of()[_RETENTION_VAR])
    assert 1 <= days <= 30


def test_catalog_and_infra_compose_agree() -> None:
    catalog, compose = _catalog_env(), _infra_compose_env()
    for key in ("PHOENIX_WORKING_DIR", _RETENTION_VAR):
        assert catalog[key] == compose[key]
