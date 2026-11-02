# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the runner compose file with its defaulted path variables resolved.

``docker/docker-compose.runners.yml`` spells the runner user's home and the
runner host's deploy home as ``${VAR:-default}`` so the file holds no
machine-specific path. Compose resolves them to the default when the variable is
unset, which is how every deploy runs. Tests that assert on the mount targets
and credential paths the fleet actually gets read the file through this module.
"""

from __future__ import annotations

from pathlib import Path

RUNNER_HOME = "/home/runner"
RUNNER_HOST_HOME = "~"
RUNNER_HOME_VARIABLE = "${RUNNER_USER_HOME:-" + RUNNER_HOME + "}"


def resolve_runner_compose_defaults(text: str) -> str:
    """Replace the defaulted runner-home variable with its default."""
    return text.replace(RUNNER_HOME_VARIABLE, RUNNER_HOME)


def read_runner_compose(path: Path) -> str:
    """The compose file's text as compose renders it with no variable set."""
    return resolve_runner_compose_defaults(path.read_text(encoding="utf-8"))
