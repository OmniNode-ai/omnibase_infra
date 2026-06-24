# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared rpk subprocess environment helper for verification probes.

The publication and subscription probes both shell out to ``rpk``. The broker
address is provided via ``RPK_BROKERS`` (and friends), which lives in
``~/.omnibase/.env`` on operator hosts but is not always exported into the
process environment. Both probes must merge that file so rpk dials the
configured broker instead of falling back to the rpk default ``127.0.0.1:9092``.
"""

from __future__ import annotations

import os
from pathlib import Path

_OMNIBASE_ENV = Path.home() / ".omnibase" / ".env"


def rpk_env() -> dict[str, str]:
    """Return ``os.environ`` merged with ``~/.omnibase/.env`` for rpk subprocesses.

    Existing process-environment values win over file values, so an explicitly
    exported ``RPK_BROKERS`` is never overridden by a stale ``.env`` entry.
    """
    env = dict(os.environ)
    if _OMNIBASE_ENV.exists():
        with open(_OMNIBASE_ENV) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env.setdefault(key.strip(), value.strip())
    return env


__all__: list[str] = ["rpk_env"]
