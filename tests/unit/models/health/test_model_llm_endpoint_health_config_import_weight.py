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
import os
import subprocess
import sys
from pathlib import Path

import pytest

import omnibase_infra

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


def test_config_construction_imports_no_adapter_or_qdrant_module(
    tmp_path: Path,
) -> None:
    # CI syncs with ``--no-install-project``: the package reaches the test
    # process through pytest's ``pythonpath``, not the venv, so the fresh child
    # gets the same source root explicitly. It runs from an empty directory so
    # nothing else on the working path can be imported.
    src_root = str(Path(omnibase_infra.__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": src_root},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_package_init_reexports_nothing() -> None:
    from omnibase_infra.adapters import llm

    assert not hasattr(llm, "AdapterModelRouter")
    assert not hasattr(llm, "__all__")
