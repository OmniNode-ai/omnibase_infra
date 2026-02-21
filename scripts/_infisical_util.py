#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared utilities for Infisical provisioning scripts.

This module contains helpers used by both provision-infisical.py and
register-repo.py.  It is intentionally a lightweight, stdlib-only module so
that it can be imported before any project dependencies are installed.

.. versionadded:: 0.10.0
    Extracted from provision-infisical.py and register-repo.py (OMN-2287).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a key-value dict.

    Skips blank lines and comment lines (starting with ``#``).
    Handles ``export KEY=value`` syntax.
    Strips inline comments and surrounding quotes from values.

    Args:
        env_path: Path to the ``.env`` file.  Returns an empty dict if the
            file does not exist.

    Returns:
        A mapping of environment variable names to their string values.
    """
    values: dict[str, str] = {}
    if not env_path.is_file():
        logger.warning("Env file not found: %s", env_path)
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:]
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        is_quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"')
        if is_quoted:
            # Quoted values are taken verbatim (minus the surrounding quotes).
            # Inline comments inside quotes are part of the value, not comments,
            # so no comment-stripping is needed here.  The elif below only runs
            # for unquoted values, where a space-hash / tab-hash sequence marks
            # the start of a genuine inline comment.
            value = value[1:-1]
        elif " #" in value or "\t#" in value or "#" in value[1:]:
            # Split on the first inline comment marker.  We recognise three
            # forms for unquoted values:
            #   - space-hash  (VALUE=abc #comment)
            #   - tab-hash    (VALUE=abc\t#comment)
            #   - bare hash after the first character (VALUE=abc#comment)
            # The bare-hash case is intentionally anchored to value[1:] so
            # that a value that *starts* with '#' is not misidentified as a
            # comment (which would be an unusual but valid value like '#000').
            # Quoted values are handled by the is_quoted branch above, so
            # legitimate '#' characters in quoted strings (e.g. hex colours,
            # URLs) are already protected and never reach this branch.
            space_pos = value.find(" #")
            tab_pos = value.find("\t#")
            bare_pos = value.find("#", 1)  # search from index 1, not 0
            candidates = [p for p in (space_pos, tab_pos, bare_pos) if p != -1]
            cut = min(candidates)
            value = value[:cut].strip()
        if key:
            values[key] = value
    return values
