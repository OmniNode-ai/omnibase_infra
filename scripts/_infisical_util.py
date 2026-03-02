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
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a key-value dict.

    Skips blank lines and comment lines (starting with ``#``).
    Handles ``export KEY=value`` syntax.
    Strips inline comments and surrounding quotes from values.

    Warning:
        **Unquoted values containing ``#`` will be truncated at the ``#``.**
        This affects any unquoted value where ``#`` appears after the first
        character — including URLs with fragments (``http://host/#fragment``)
        and passwords or tokens that contain ``#``.  This is intentional: a
        bare ``#`` in an unquoted value is treated as the start of an inline
        comment, matching standard ``.env`` file convention.

        To preserve ``#`` characters in a value, quote the value::

            VALUE='http://host/#fragment'   # single quotes — verbatim
            VALUE="p@ss#word"               # double quotes — also verbatim

        Quoted values are taken verbatim (minus the surrounding quotes) and
        are never subject to comment stripping.

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
    logger.info("Parsed %d values from %s", len(values), env_path)
    return values


def update_env_file(env_path: Path, updates: dict[str, str]) -> None:
    """Write or update key=value pairs in a .env file, always overwriting existing keys.

    Unlike the conservative ``_write_env_vars`` in ``provision-infisical.py``,
    this function always overwrites existing keys.  This is intentional for
    credential-rotation use cases (e.g. ``provision-keycloak.py``) where client
    secrets are regenerated on every run and must be written unconditionally.

    Behaviour:
    - Existing uncommented keys are updated in-place (value always overwritten).
    - Keys that do not exist yet are appended at the end of the file.
    - Commented-out lines (``# KEY=value``) are left untouched; the key is
      appended as a new uncommented entry.
    - Preserves ``export KEY=value`` prefix for lines that already use it.
    - Writes atomically via a ``.tmp`` rename (POSIX ``os.rename``).
    - Creates the parent directory with mode ``0o700`` if it does not exist.
    - The file (and its ``.tmp`` staging file) is created/kept at ``0o600``.

    Args:
        env_path: Path to the ``.env`` file.
        updates: Mapping of variable names to their new string values.

    .. versionadded:: 0.12.0
        Added for use by ``provision-keycloak.py`` (OMN-3362).
    """
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    existing: dict[str, int] = {}  # key -> line index (uncommented lines only)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "=" in stripped:
            key = stripped.partition("=")[0].strip().removeprefix("export ").strip()
            existing[key] = i

    appended: list[str] = []
    for key, value in updates.items():
        if key in existing:
            current_line = lines[existing[key]]
            had_export = current_line.lstrip().startswith("export ")
            prefix = "export " if had_export else ""
            lines[existing[key]] = f"{prefix}{key}={value}"
            logger.info("  Updated %s", key)
        else:
            appended.append(f"{key}={value}")
            logger.info("  Added %s", key)

    if appended:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# --- Keycloak credentials (provisioned automatically) ---")
        lines.extend(appended)

    content = "\n".join(lines) + "\n"
    env_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = env_path.with_name(env_path.name + ".tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        tmp.replace(env_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Wrote %d key(s) to %s", len(updates), env_path)
