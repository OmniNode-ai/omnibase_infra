# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every LLM adapter imports from its own module in a fresh interpreter (OMN-26336).

The ``omnibase_infra.adapters.llm`` package re-exports nothing, so each caller
imports the adapter it needs from that adapter's module. This reads the adapter
modules from the package directory, imports all of them in one fresh
interpreter that holds no earlier import, and checks that the endpoint health
config still builds beside them. A caller that still reached an adapter through
the package would fail here with an ``ImportError`` instead of at runtime boot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import omnibase_infra
from omnibase_infra.adapters import llm

pytestmark = pytest.mark.integration

_PACKAGE_DIR = Path(llm.__file__).resolve().parent

_PROBE = """
import importlib, json, sys

failed = {}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        failed[name] = repr(exc)

from omnibase_infra.models.health.model_llm_endpoint_health_config import (
    ModelLlmEndpointHealthConfig,
)

ModelLlmEndpointHealthConfig(endpoints={"local": "http://localhost:8000/v1"})
print(json.dumps(failed))
"""


def _adapter_module_names() -> list[str]:
    return sorted(
        f"omnibase_infra.adapters.llm.{path.stem}"
        for path in _PACKAGE_DIR.glob("adapter_*.py")
    )


def test_adapter_modules_are_found_on_disk() -> None:
    assert _adapter_module_names(), "no adapter_*.py module found in the package"


def test_every_adapter_imports_from_its_own_module(tmp_path: Path) -> None:
    src_root = str(Path(omnibase_infra.__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(_adapter_module_names())],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": src_root},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {}
