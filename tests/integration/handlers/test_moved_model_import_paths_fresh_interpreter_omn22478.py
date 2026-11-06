# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each restored import path loads first in a fresh interpreter (OMN-22478).

An installed tool reaches these paths with nothing of this package imported yet, so a
cycle between the old path and the shared models shows only there, and never in a test
process that has already loaded the package.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("old_path", "old_name", "new_path", "new_name"),
    [
        (
            "omnibase_infra.handlers.models.infisical",
            "ModelInfisicalHandlerConfig",
            "omnibase_infra.models.security.model_infisical_config",
            "ModelInfisicalConfig",
        ),
        (
            "omnibase_infra.handlers.models.model_retry_state",
            "ModelRetryState",
            "omnibase_infra.models.resilience.model_retry_state",
            "ModelRetryState",
        ),
    ],
)
def test_old_path_loads_first_and_is_the_moved_model(
    old_path: str, old_name: str, new_path: str, new_name: str
) -> None:
    program = (
        f"from {old_path} import {old_name} as old\n"
        f"from {new_path} import {new_name} as new\n"
        "assert old is new\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
