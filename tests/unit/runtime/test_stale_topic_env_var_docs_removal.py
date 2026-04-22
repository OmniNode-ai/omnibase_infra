# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-9409: Stale ONEX_OUTPUT_TOPIC / ONEX_INPUT_TOPIC refs must stay removed.

OMN-8784 removed `ONEX_INPUT_TOPIC` and `ONEX_OUTPUT_TOPIC` from the runtime
kernel; topics are now derived from each node's `contract.yaml`
`event_bus.subscribe_topics` / `event_bus.publish_topics`. Setting either env
var now raises `ProtocolConfigurationError` at kernel bootstrap.

This test locks in OMN-9409's documentation cleanup: if someone re-introduces
an *active-looking* reference (uncommented assignment in env example files,
`os.getenv("ONEX_*_TOPIC", ...)` in integration test helpers, or a live table
row in operator docs), the next deploy-from-main would create containers with
`ONEX_*_TOPIC` baked in and trigger the crash loop documented in OMN-9409.

Historical / deprecation references (comments, docstrings, deprecated-guard
unit tests, and this file itself) are allowed — the scan looks for patterns
that would regress the removal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]

# Env-example-style files must not define ONEX_*_TOPIC as a live default.
# An uncommented `ONEX_INPUT_TOPIC=...` or `ONEX_OUTPUT_TOPIC=...` at the start
# of a line re-introduces the banned env var on the next deploy.
ENV_EXAMPLE_FILES: tuple[str, ...] = (
    "docker/env-example-full.txt",
    "docs/env-example-full.txt",
)
ACTIVE_ENV_ASSIGNMENT = re.compile(r"(?m)^(ONEX_INPUT_TOPIC|ONEX_OUTPUT_TOPIC)\s*=")

# E2E integration test helpers must not use ONEX_*_TOPIC as an override knob.
# The kernel rejects these vars, so `os.getenv("ONEX_*_TOPIC", default)` is a
# foot-gun: tests would either silently fall back to `default` (misleading)
# or poison a runtime container that reads the same env and crash it.
INTEGRATION_TEST_FILES: tuple[str, ...] = (
    "tests/integration/registration/e2e/test_runtime_e2e.py",
    "tests/integration/registration/e2e/conftest.py",
)
ENV_OVERRIDE_READ = re.compile(
    r'os\.(?:getenv|environ\.get|environ\s*\[)\s*\(?\s*["\']'
    r"(ONEX_INPUT_TOPIC|ONEX_OUTPUT_TOPIC)"
    r"['\"]"
)


class TestStaleTopicEnvVarDocsRemoval:
    """Guard OMN-8784's removal against regressions at the edges the kernel
    test (test_kernel_topic_env_var_removal.py) does not cover: env example
    files and integration-test helpers."""

    @pytest.mark.parametrize("rel_path", ENV_EXAMPLE_FILES)
    def test_no_active_env_assignment(self, rel_path: str) -> None:
        path = REPO_ROOT / rel_path
        assert path.exists(), f"env example file missing: {rel_path}"
        content = path.read_text(encoding="utf-8")
        matches = ACTIVE_ENV_ASSIGNMENT.findall(content)
        assert not matches, (
            f"{rel_path} contains active ONEX_*_TOPIC assignment(s): "
            f"{matches}. OMN-8784 removed these env vars; OMN-9409 cleaned up "
            "the stale defaults. Re-introducing an uncommented default would "
            "cause the next deploy to bake the banned env var into the "
            "runtime container and trigger a crash loop."
        )

    @pytest.mark.parametrize("rel_path", INTEGRATION_TEST_FILES)
    def test_no_env_override_read(self, rel_path: str) -> None:
        path = REPO_ROOT / rel_path
        assert path.exists(), f"integration test file missing: {rel_path}"
        content = path.read_text(encoding="utf-8")
        matches = ENV_OVERRIDE_READ.findall(content)
        assert not matches, (
            f"{rel_path} reads ONEX_*_TOPIC from the environment: {matches}. "
            "Use the contract-declared topic string directly — the kernel "
            "hard-fails if ONEX_INPUT_TOPIC or ONEX_OUTPUT_TOPIC is set "
            "(OMN-8784), so an override knob is a foot-gun."
        )
