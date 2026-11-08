# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19513: the dev lane's emit-effect spool lives on a volume, not in /tmp.

node_event_emit_effect spools every owed event to ONEX_EMIT_EFFECT_SPOOL_DIR,
else $XDG_RUNTIME_DIR, else /tmp. With neither set, a recreate of the runtime
dropped the events still in the spool. These tests read the rendered dev lane
and assert that each runtime that hosts the effect points the spool at a path
under a named volume mounted at /app/data, and that no two services share it.
"""

from __future__ import annotations

from functools import cache
from pathlib import PurePosixPath
from typing import Any

import pytest

from tests.unit.infra.test_broker_profile_omn19419 import _render

pytestmark = pytest.mark.unit

# OMN-22930 (T8): primary dev folds the effects role into omninode-runtime, so
# the rendered dev lane holds one runtime and no runtime-effects service. The
# hosts list names both; the tests below read which of them the render holds.
_SERVICES = ("omninode-runtime",)
_RETIRED_BY_FOLD = "runtime-effects"
_DATA = PurePosixPath("/app/data")


@cache
def _dev() -> dict[str, Any]:
    return _render("dev")


def _volume_at(service: dict[str, Any], target: PurePosixPath) -> str | None:
    for mount in service.get("volumes", []):
        if mount.get("type") == "volume" and PurePosixPath(mount["target"]) == target:
            return str(mount["source"])
    return None


@pytest.mark.parametrize("name", _SERVICES)
def test_spool_dir_is_under_the_services_data_volume(name: str) -> None:
    service = _dev()["services"][name]
    spool = service["environment"].get("ONEX_EMIT_EFFECT_SPOOL_DIR")
    assert spool, f"{name} sets no ONEX_EMIT_EFFECT_SPOOL_DIR: the spool falls to /tmp"
    path = PurePosixPath(spool)
    assert path.is_absolute()
    assert _DATA in path.parents, f"{name} spools to {spool}, not under {_DATA}"
    assert _volume_at(service, _DATA), f"{name} mounts no named volume at {_DATA}"


def test_the_effects_role_runs_inside_the_one_runtime_so_it_has_no_second_spool() -> (
    None
):
    services = _dev()["services"]
    assert _RETIRED_BY_FOLD not in services, (
        f"{_RETIRED_BY_FOLD} renders on primary dev: the effects fold (OMN-22930) "
        "should have folded it into omninode-runtime"
    )
    assert "omninode-runtime" in services


def test_no_two_services_drain_the_same_spool_volume() -> None:
    services = _dev()["services"]
    hosts = [name for name in (*_SERVICES, _RETIRED_BY_FOLD) if name in services]
    sources = [_volume_at(services[name], _DATA) for name in hosts]
    assert hosts, "no emit-effect host renders on the dev lane"
    assert len(set(sources)) == len(sources), sources
