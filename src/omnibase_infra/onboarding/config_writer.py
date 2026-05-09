# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Config writer for onboarding-generated env files (OMN-10783).

Separate, explicit invocation only — never called automatically.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


class ConfigWriterError(ValueError):
    """Raised when a key or value contains characters that would corrupt the env format."""


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

    Explicit invocation only. Never write to ~/.omnibase/ from tests.
    """

    def render(
        self,
        env_dict: dict[str, str],
        existing_content: str | None = None,
    ) -> str:
        """Pure render — no file I/O.

        Merges env_dict into existing_content (if any), preserving keys not
        present in env_dict and overwriting keys that are.

        Args:
            env_dict: New key=value pairs to write.
            existing_content: Existing file content to merge with, or None.

        Returns:
            Merged content as a string.
        """
        merged: dict[str, str] = {}

        if existing_content:
            for line in existing_content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped:
                    key, _, value = stripped.partition("=")
                    merged[key.strip()] = value.strip()

        for k, v in {**merged, **env_dict}.items():
            _validate_env_pair(k, v)

        merged.update(env_dict)

        lines = [f"{k}={v}" for k, v in sorted(merged.items())]
        return "\n".join(lines) + ("\n" if lines else "")

    def write(self, env_dict: dict[str, str], target_path: Path) -> str:
        """Merge env_dict with existing file and atomically write result.

        Preserves keys not in env_dict; overwrites keys that are.
        Uses tmp-file + rename for atomicity.

        Args:
            env_dict: New key=value pairs to write.
            target_path: Destination path.

        Returns:
            Merged content as a string.
        """
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
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            Path(tmp_path).replace(target_path)
        except Exception:
            with suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
            raise

        return content


__all__ = ["ConfigWriter", "ConfigWriterError"]
