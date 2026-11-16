# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""load_dev_boot_guard keeps modules already imported in the process (OMN-24017).

The guard's early return only covers the named effect module. When that module is
not yet loaded, the loop re-executed every listed file and overwrote the sys.modules
entries for classes the rest of the process had already imported, so tests that
imported the real classes then failed with class-identity errors. This regression
test imports the real model module, remembers its class, removes only the llm
endpoint env module from sys.modules, calls the guard, and asserts the remembered
class is still the one in sys.modules.
"""

from __future__ import annotations

import sys

import pytest
from deploy_agent import executor as executor_mod

import omnibase_infra.models.model_dev_lane_code_ref_pin as model_module

pytestmark = pytest.mark.unit

LLM_ENDPOINT_ENV_MODULE = (
    "omnibase_infra.nodes.node_setup_local_provision_effect.runtime_llm_endpoint_env"
)
MODEL_MODULE = "omnibase_infra.models.model_dev_lane_code_ref_pin"


@pytest.fixture
def restore_omnibase_modules(monkeypatch: pytest.MonkeyPatch):
    """Snapshot every omnibase_infra sys.modules entry and restore it on teardown.

    Restores each snapshotted key and deletes any key the test added, so the test
    never pollutes other tests even when it fails.
    """
    snapshot = {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key.startswith("omnibase_infra")
    }
    yield
    for key, module in snapshot.items():
        sys.modules[key] = module
    for key in list(sys.modules):
        if key.startswith("omnibase_infra") and key not in snapshot:
            del sys.modules[key]


def test_load_dev_boot_guard_keeps_already_imported_modules(
    monkeypatch: pytest.MonkeyPatch, restore_omnibase_modules: None
) -> None:
    remembered = model_module.ModelDevLaneCodeRefPin
    monkeypatch.delitem(sys.modules, LLM_ENDPOINT_ENV_MODULE, raising=False)

    module = executor_mod.load_dev_boot_guard(executor_mod._LLM_ENV_MODULE)

    assert sys.modules[MODEL_MODULE].ModelDevLaneCodeRefPin is remembered
    assert module is sys.modules[LLM_ENDPOINT_ENV_MODULE]
