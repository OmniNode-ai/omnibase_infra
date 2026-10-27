# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The two ``~/.onex`` files, and the rules for reading and writing them.

OMN-18432. Split out of ``store_lane_credential`` so each object stays small
enough to read whole: this one owns HOW the files behave -- the YAML round trip
that preserves every other writer's keys, the 0600 mode enforced on read as
well as on write, the JSON refusals -- and the credential stores own WHAT they
hold.

The 0600 check on READ is the one that matters. The file survives ``chmod``,
backup/restore and ``scp``, so a write-time check alone proves nothing about
the file actually being loaded (OMN-15922 established this and the reasoning is
unchanged).

Both files are replaced atomically (OMN-20827). A direct ``write_text``
truncates the target before writing it, so a crash in between leaves a short or
empty file, and an empty ``credentials.json`` is refused by the next
``load_secrets`` -- the refusal that would stop a rerun of ``lane-login`` from
repairing the machine. Each write goes to a temporary file in the same
directory, created at 0600, flushed, renamed over the target, and the directory
entry is flushed after the rename.

``StoreGatewayCredential`` still carries its own copy of this plumbing. It is
not migrated here in the same change: its refusals are load-bearing for the
cloud credential path and re-homing them needs its own proof, not a rider on a
change about a bus identity.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
from pathlib import Path

import yaml

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError

__all__ = ["StoreOnexHomeFiles"]


class StoreOnexHomeFiles:
    """The two ``~/.onex`` files, and the rules for reading and writing them."""

    def __init__(self, onex_home: Path) -> None:
        self._onex_home = onex_home

    @property
    def config_path(self) -> Path:
        return self._onex_home / "config.yaml"

    @property
    def credentials_path(self) -> Path:
        return self._onex_home / "credentials.json"

    def load_config(self, *, must_exist: bool) -> dict[str, object]:
        if not self.config_path.exists():
            if not must_exist:
                return {}
            raise ModelOnexError(
                f"no ONEX config at {self.config_path} -- this machine holds "
                "no bus identity for any lane.",
                error_code=EnumCoreErrorCode.CONFIGURATION_NOT_FOUND,
            )
        # yaml-ok: user-authored config file with several writers (OMN-16037);
        # a Pydantic model here would either reject another writer's keys or
        # silently drop them on the round trip.
        document = yaml.safe_load(self.config_path.read_text())
        if document is None:
            return {}
        if not isinstance(document, dict):
            raise ModelOnexError(
                f"{self.config_path} must be a YAML mapping, found "
                f"{type(document).__name__}.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            )
        return {str(key): value for key, value in document.items()}

    def write_config(self, document: dict[str, object]) -> None:
        """Re-dump the WHOLE document, so every other writer's keys survive."""
        self._replace(self.config_path, yaml.safe_dump(document, sort_keys=False))

    def load_secrets(self) -> dict[str, str]:
        if not self.credentials_path.exists():
            return {}
        try:
            document = json.loads(self.credentials_path.read_text())
        except json.JSONDecodeError as exc:
            raise ModelOnexError(
                f"{self.credentials_path} is not valid JSON; refusing to "
                "overwrite it and lose the credentials it may hold.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            ) from exc
        if not isinstance(document, dict):
            raise ModelOnexError(
                f"{self.credentials_path} must be a JSON object.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            )
        return {str(key): str(value) for key, value in document.items()}

    def write_secrets(self, secrets: dict[str, str]) -> None:
        """Write the secret file so it is never briefly world-readable.

        The temporary file is created at 0600 whatever the umask, so the value
        is never on disk at a wider mode, and the rename keeps that mode.
        """
        self._replace(
            self.credentials_path, json.dumps(secrets, indent=2, sort_keys=True)
        )

    def _replace(self, target: Path, text: str) -> None:
        """Write ``text`` to a 0600 temporary file and rename it over ``target``.

        On any failure the temporary file is removed and ``target`` is left as
        it was, byte for byte.
        """
        self._onex_home.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=self._onex_home, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            try:
                data = text.encode("utf-8")
                while data:
                    data = data[os.write(fd, data) :]
                os.fsync(fd)
            finally:
                os.close(fd)
            Path(temporary).replace(target)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                Path(temporary).unlink()
            raise
        directory = os.open(self._onex_home, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def read_secret(self, ref: str, remediation: str) -> str:
        """Resolve one reference, refusing a file anyone but the owner can read."""
        if not self.credentials_path.exists():
            raise ModelOnexError(
                f"no credentials.json at {self.credentials_path}, but config "
                f"references '{ref}'. To restore it, {remediation}.",
                error_code=EnumCoreErrorCode.CONFIGURATION_NOT_FOUND,
            )

        mode = stat.S_IMODE(self.credentials_path.stat().st_mode)
        if mode & 0o077:
            raise ModelOnexError(
                f"{self.credentials_path} is mode {mode:04o}; it must be 0600 "
                "(owner-only). Refusing to read a group- or world-readable "
                f"credential file. Fix with: chmod 600 {self.credentials_path}",
                error_code=EnumCoreErrorCode.PERMISSION_DENIED,
            )

        secrets = self.load_secrets()
        if ref not in secrets:
            raise ModelOnexError(
                f"{self.credentials_path} has no entry for '{ref}' named by "
                f"{self.config_path}. To restore it, {remediation}.",
                error_code=EnumCoreErrorCode.CONFIGURATION_NOT_FOUND,
            )
        secret = secrets[ref]
        if not secret:
            raise ModelOnexError(
                f"{self.credentials_path}: entry '{ref}' must be a non-empty string.",
                error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            )
        return secret
