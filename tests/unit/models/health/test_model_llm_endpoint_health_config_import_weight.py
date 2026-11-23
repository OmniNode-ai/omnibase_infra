# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Building the LLM endpoint health config must not import the adapter chain.

The config's ``_drop_coding_plan_endpoints`` validator imports
``omnibase_infra.adapters.llm.coding_plan_endpoint``. While the package
``__init__`` imported every adapter eagerly, that one import pulled the
similarity adapter and, through it, the Qdrant client into the running event
loop: the composed boot check measured a 1.66 s loop hold there at runtime pin
dev-20261123105315-1e0559bbc392. The check runs in a fresh interpreter because
the import cost is paid once per process, so an in-process assertion would read
whatever an earlier test already loaded.
"""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_PROBE = """
import json, sys
from omnibase_infra.models.health.model_llm_endpoint_health_config import (
    ModelLlmEndpointHealthConfig,
)

ModelLlmEndpointHealthConfig(endpoints={"local": "http://localhost:8000/v1"})
heavy = sorted(
    name
    for name in sys.modules
    if name.startswith("qdrant_client")
    or name.startswith("omnibase_infra.adapters.llm.adapter_")
)
print(json.dumps(heavy))
"""


def test_config_construction_imports_no_adapter_or_qdrant_module() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_package_init_reexports_nothing() -> None:
    from omnibase_infra.adapters import llm

    assert not hasattr(llm, "AdapterModelRouter")
    assert not hasattr(llm, "__all__")
