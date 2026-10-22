# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A real interpreter booted through configure_logging never writes a provider key.

The unit regression swaps the root handler in-process. This runs the kernel's
logging bootstrap in a fresh interpreter, so the handler basicConfig installs
on stderr is the one judged, and reads the bytes that process actually wrote.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from omnibase_infra.utils.util_log_credential_redaction import LOG_REDACTION_MARKER

pytestmark = pytest.mark.integration

# Constructed, deliberately repetitive fixtures; none is a real credential.
PROVIDER_KEYS = {
    "openai": "sk-" + "testOnly0" * 6,
    "openai-project": "sk-proj-" + "testOnly_-0" * 16,
    "anthropic": "sk-ant-api03-" + "testOnly_-0" * 9,
    "openrouter": "sk-or-v1-" + "0123456789abcdef" * 4,
    "gemini": "AIza" + "test0_-" * 5,
    "glm": "0123456789abcdef" * 2 + "." + "testOnly0" * 2,
}

_CHILD = textwrap.dedent(
    """
    import json
    import logging
    import os

    from omnibase_infra.runtime.service_kernel import configure_logging

    configure_logging()
    logger = logging.getLogger("provider-redaction-process")
    for name, key in json.loads(os.environ["PROVIDER_KEYS_JSON"]).items():
        logger.warning("provider %s returned 401 for %s", name, key)
        try:
            raise RuntimeError(f"upstream echoed {key}")
        except RuntimeError:
            logger.exception("call to %s failed", name)
    logger.warning("ordinary diagnostic survives")
    """
)


def test_booted_runtime_logging_redacts_every_provider_key() -> None:
    env = {**os.environ, "PROVIDER_KEYS_JSON": json.dumps(PROVIDER_KEYS)}
    env["ONEX_LOG_LEVEL"] = "INFO"
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    stderr = completed.stderr
    for name, key in PROVIDER_KEYS.items():
        assert key not in stderr, f"{name} key escaped the booted runtime log"
        assert f"provider {name} returned 401" in stderr
    assert stderr.count(LOG_REDACTION_MARKER) >= 2 * len(PROVIDER_KEYS)
    assert "ordinary diagnostic survives" in stderr
