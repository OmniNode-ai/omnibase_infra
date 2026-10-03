# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Console entry points import in a fresh interpreter (OMN-19444).

Package ``__init__`` files resolve their re-exports lazily (PEP 562). The old
eager imports pinned an import order that hid latent circular imports, so a
module can import fine inside a warm test process and still fail when it is
the first thing an interpreter loads. Each ``[project.scripts]`` target and the
delegate CLI is therefore imported in its own subprocess, and the callable the
entry point names must resolve.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    "omnibase_infra.cli.commands:cli",
    "omnibase_infra.runtime.kernel:main",
    "omnibase_infra.runtime.gateway_forwarder:main",
    "omnibase_infra.runtime.gateway_canary_probe:main",
    "omnibase_infra.runtime.action_authorization_claim.cli:main",
    "omnibase_infra.cli.infra_test.cli:cli",
    "omnibase_infra.cli.git_hook_relay:main",
    "omnibase_infra.cli.linear_relay:main",
    "omnibase_infra.cli.cli_delegate:delegate_command",
]


@pytest.mark.integration
@pytest.mark.parametrize("target", ENTRY_POINTS)
def test_entry_point_resolves_in_fresh_interpreter(target: str) -> None:
    module, attr = target.split(":")
    code = (
        "import importlib, sys;"
        f"m = importlib.import_module({module!r});"
        f"sys.exit(0 if callable(getattr(m, {attr!r}, None)) else 3)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
