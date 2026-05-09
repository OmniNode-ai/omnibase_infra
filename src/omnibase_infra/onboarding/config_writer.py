# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Config writer for onboarding-generated env files.

Separate, explicit invocation only: callers choose when to write, and dry-run
paths should call ``ConfigWriter.render`` instead of touching the filesystem.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


class ConfigWriterError(ValueError):
    """Raised when a key or value contains characters that would corrupt env output."""


def _validate_env_pair(key: str, value: str) -> None:
    """Reject keys or values that would corrupt KEY=value env lines."""
    if "=" in key:
        msg = f"env key {key!r} contains an equals sign"
        raise ConfigWriterError(msg)

    for char, label in (("\n", "newline"), ("\r", "carriage return")):
        if char in key:
            msg = f"env key {key!r} contains a {label} character"
            raise ConfigWriterError(msg)
        if char in value:
            msg = f"env value for key {key!r} contains a {label} character"
            raise ConfigWriterError(msg)


class ConfigWriter:
    """Writes env key=value files with merge-and-preserve semantics.

    Explicit invocation only. Never write to ``~/.omnibase/`` from tests.
    """

    def render(
        self,
        env_dict: dict[str, str],
        existing_content: str | None = None,
    ) -> str:
        """Return merged env content without writing to disk."""
        merged: dict[str, str] = {}

        if existing_content:
            for line in existing_content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped:
                    key, _, value = stripped.partition("=")
                    merged[key.strip()] = value.strip()

        for key, value in {**merged, **env_dict}.items():
            _validate_env_pair(key, value)

        merged.update(env_dict)

        lines = [f"{key}={value}" for key, value in sorted(merged.items())]
        return "\n".join(lines) + ("\n" if lines else "")

    def write(self, env_dict: dict[str, str], target_path: Path) -> str:
        """Merge env_dict with an existing file and atomically write the result."""
        existing_content: str | None = None
        if target_path.exists():
            existing_content = target_path.read_text(encoding="utf-8")

        content = self.render(env_dict, existing_content)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=target_path.parent,
            prefix=f".{target_path.name}.tmp.",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file_handle:
                file_handle.write(content)
            Path(tmp_path).replace(target_path)
        except Exception:
            with suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
            raise

        return content


def write_env_file(env_dict: dict[str, str], target_path: Path) -> str:
    """Explicit convenience wrapper for callers that do not need a writer instance."""
    return ConfigWriter().write(env_dict, target_path)


__all__ = ["ConfigWriter", "ConfigWriterError", "write_env_file"]
