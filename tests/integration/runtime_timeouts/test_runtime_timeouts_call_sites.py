# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test: call sites carry the timeouts declared on disk (OMN-21719)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.runtime_timeouts import (
    RUNTIME_TIMEOUTS,
    ModelRuntimeTimeouts,
    load_runtime_timeouts,
)
from omnibase_infra.models.runtime_timeouts import (
    model_runtime_timeouts as model_module,
)

_SHIPPED = Path(model_module.__file__).parent / "runtime_timeouts.yaml"

# A TTL and a probe wait that differ from the shipped 300 and 1.0.
_SENTINEL_TTL = 601
_SENTINEL_PROBE_SECONDS = 7.25

_PROBE_BINDING_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import omnibase_infra.models.runtime_timeouts as pkg
    from omnibase_infra.models.runtime_timeouts import model_runtime_timeouts as m

    m._TIMEOUTS_RESOURCE = Path(sys.argv[1])
    m.load_runtime_timeouts.cache_clear()
    pkg.RUNTIME_TIMEOUTS = m.load_runtime_timeouts()

    from omnibase_infra.probes import capability_probe

    print(capability_probe.PROBE_TTL_SECONDS, capability_probe._PROBE_SECONDS)
    """
)


@pytest.mark.integration
def test_shipped_file_is_the_one_the_runtime_loads() -> None:
    """The shipped file read as plain YAML equals the model the runtime holds."""
    declared = yaml.safe_load(_SHIPPED.read_text(encoding="utf-8"))["timeouts"]
    assert ModelRuntimeTimeouts.model_validate(declared) == RUNTIME_TIMEOUTS
    assert load_runtime_timeouts() is RUNTIME_TIMEOUTS
    assert declared["capability_probe_ttl_seconds"] != _SENTINEL_TTL


@pytest.mark.integration
def test_capability_probe_reads_its_ttl_and_wait_from_the_declared_file(
    tmp_path: Path,
) -> None:
    """A probe module imported against a file with other values holds those values."""
    declared = yaml.safe_load(_SHIPPED.read_text(encoding="utf-8"))["timeouts"]
    declared["capability_probe_ttl_seconds"] = _SENTINEL_TTL
    declared["capability_probe_seconds"] = _SENTINEL_PROBE_SECONDS
    path = tmp_path / "runtime_timeouts.yaml"
    path.write_text(yaml.safe_dump({"timeouts": declared}), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE_BINDING_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.split() == [
        str(_SENTINEL_TTL),
        str(_SENTINEL_PROBE_SECONDS),
    ]
