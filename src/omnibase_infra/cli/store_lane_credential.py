# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""StoreLaneCredential -- a lane's bus identity under ``~/.onex`` (OMN-18432).

Same two files and the same single axis as ``StoreGatewayCredential``
(OMN-15922): whether the content is a secret.

``~/.onex/config.yaml`` gains a ``lanes:`` block of references and principal
NAMES. It never carries a password. That is not a style preference --
config.yaml is world-readable by default and is the file operators paste into
issues and screen-shares -- so an inline ``sasl_password`` is refused outright
rather than accepted with a warning. A convenience fallback would make the
by-reference rule advisory, and the next credential would take the easy path.

``~/.onex/credentials.json`` (mode 0600, enforced on READ as well as on write)
holds ``{<ref>: <secret>}`` and is shared with the gateway credential: one
secret file per machine, many references into it. Enforcing the mode on read is
the check that matters, because the file survives ``chmod``, backup/restore and
``scp``, so a write-time check alone proves nothing about the file actually
being loaded.

WHY A SECOND STORE CLASS RATHER THAN A WIDER GATEWAY ONE
    The gateway store answers "who is this machine to the cloud gateway" and
    holds exactly one credential, which is why it REFUSES a machine carrying
    two kinds at once. A bus identity is a different question with a different
    cardinality: a machine legitimately holds one per lane. Folding lanes into
    that class would mean relaxing the one-credential refusal that is the
    gateway store's most important property.

WHY THE PLUMBING IS ITS OWN CLASS
    ``StoreOnexHomeFiles`` owns HOW the two files are read and written -- the YAML
    round trip that preserves every other writer's keys, the 0600 enforcement,
    the JSON refusals. ``StoreLaneCredential`` owns WHAT a lane identity is.
    Splitting them keeps each object small enough to read whole, and it is the
    seam the gateway store would move onto if the duplication between the two
    is ever collapsed -- which is a separate change, because that class's
    refusals are load-bearing and re-homing them needs its own proof.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Final

import yaml
from pydantic import SecretStr

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.model_lane_credential import ModelLaneCredential
from omnibase_infra.cli.store_onex_home_files import StoreOnexHomeFiles

__all__ = ["StoreLaneCredential"]

_LANES_BLOCK: Final[str] = "lanes"
_USERNAME_KEY: Final[str] = "sasl_username"
_PASSWORD_REF_KEY: Final[str] = "sasl_password_ref"  # pragma: allowlist secret
_INLINE_PASSWORD_KEY: Final[str] = "sasl_password"  # pragma: allowlist secret


def _remediation(lane: str) -> str:
    return (
        f"run 'onex auth lane-login --lane {lane} --sasl-username <principal> "
        "--sasl-password-stdin'"
    )


class StoreLaneCredential:
    """Reads and writes per-lane bus identities under an ``~/.onex`` root."""

    def __init__(self, *, onex_home: Path) -> None:
        """Bind the store to a directory.

        Args:
            onex_home: Directory holding ``config.yaml`` and
                ``credentials.json``. Injected rather than derived from
                ``Path.home()`` inside the class so tests drive a real
                directory instead of patching the home lookup.
        """
        self._files = StoreOnexHomeFiles(onex_home)

    @property
    def config_path(self) -> Path:
        return self._files.config_path

    @property
    def credentials_path(self) -> Path:
        return self._files.credentials_path

    @staticmethod
    def password_ref(lane: str) -> str:
        """The key this lane's value is filed under in the 0600 file.

        Derived rather than stored so the reference cannot drift from the lane
        it belongs to, and namespaced so it cannot collide with the gateway
        store's own refs in the file both stores share.
        """
        return f"{lane}-lane-sasl"

    def declared_lanes(self) -> tuple[str, ...]:
        """Lane ids this machine holds an identity for, sorted.

        Returns an empty tuple for a machine holding none, rather than raising:
        the caller that asks this question is composing a refusal message and
        must not be refused while doing so.
        """
        try:
            document = self._files.load_config(must_exist=False)
        except ModelOnexError:
            return ()
        block = document.get(_LANES_BLOCK)
        if not isinstance(block, dict):
            return ()
        return tuple(sorted(str(key) for key in block))

    def load(self, lane: str) -> ModelLaneCredential:
        """Resolve this lane's identity, or raise naming what to do about it.

        Raises:
            ModelOnexError: On any missing, blank, malformed, mis-permissioned
                or value-carrying configuration. Never returns a partially
                resolved credential: a half-resolved bus identity becomes an
                anonymous connect against an auth-required listener, which is
                the failure this store exists to make impossible.
        """
        username, password_ref = self._resolve_entry(lane)
        return ModelLaneCredential(
            lane=lane,
            sasl_username=username,
            sasl_password_ref=password_ref,
            sasl_password=SecretStr(
                self._files.read_secret(password_ref, _remediation(lane))
            ),
        )

    def save(self, *, lane: str, sasl_username: str, sasl_password: str) -> None:
        """Write the reference-only lane entry and the 0600 value.

        Only this lane's entry is replaced. Another lane's identity, the
        ``gateway:`` block, and every other top-level key survive the round
        trip -- storing one identity must never be a way to lose another.
        """
        password_ref = self.password_ref(lane)

        document = self._files.load_config(must_exist=False)
        block = document.get(_LANES_BLOCK)
        entries: dict[str, object] = (
            {str(key): value for key, value in block.items()}
            if isinstance(block, dict)
            else {}
        )
        entries[lane] = {
            _USERNAME_KEY: sasl_username,
            _PASSWORD_REF_KEY: password_ref,
        }
        document[_LANES_BLOCK] = entries
        self._files.write_config(document)

        secrets = self._files.load_secrets()
        secrets[password_ref] = sasl_password
        self._files.write_secrets(secrets)

    def clear(self, lane: str) -> None:
        """Remove this lane's value and then its reference.

        Order matters and is the same one the gateway store settled: the value
        goes first, so a process that dies between the two writes leaves a
        config naming a missing secret -- which ``load`` refuses loudly --
        rather than an orphaned value sitting on disk with nothing pointing at
        it.
        """
        password_ref = self.password_ref(lane)

        if self.credentials_path.exists():
            secrets = self._files.load_secrets()
            if password_ref in secrets:
                del secrets[password_ref]
                self._files.write_secrets(secrets)

        document = self._files.load_config(must_exist=False)
        block = document.get(_LANES_BLOCK)
        if isinstance(block, dict) and lane in block:
            del block[lane]
            document[_LANES_BLOCK] = block
            self._files.write_config(document)

    def _resolve_entry(self, lane: str) -> tuple[str, str]:
        """The validated ``(username, password_ref)`` pair for one lane."""
        document = self._files.load_config(must_exist=True)
        lanes = document.get(_LANES_BLOCK)
        if lanes is None:
            raise ModelOnexError(
                f"{self.config_path} has no '{_LANES_BLOCK}:' block -- this "
                f"machine holds no bus identity for any lane. To store one, "
                f"{_remediation(lane)}.",
                error_code=EnumCoreErrorCode.CONFIGURATION_NOT_FOUND,
            )
        if not isinstance(lanes, dict):
            raise ModelOnexError(
                f"{self.config_path}: '{_LANES_BLOCK}' must be a mapping of "
                f"lane id to identity, found {type(lanes).__name__}.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            )

        entry = lanes.get(lane)
        if entry is None:
            held = ", ".join(str(key) for key in sorted(lanes)) or "none"
            raise ModelOnexError(
                f"{self.config_path}: '{_LANES_BLOCK}' holds no identity for "
                f"lane '{lane}' (this machine holds: {held}). To store one, "
                f"{_remediation(lane)}.",
                error_code=EnumCoreErrorCode.CONFIGURATION_NOT_FOUND,
            )
        if not isinstance(entry, dict):
            raise ModelOnexError(
                f"{self.config_path}: '{_LANES_BLOCK}.{lane}' must be a "
                f"mapping, found {type(entry).__name__}.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            )
        block = {str(key): value for key, value in entry.items()}

        if _INLINE_PASSWORD_KEY in block:
            raise ModelOnexError(
                f"{self.config_path} carries an inline "
                f"'{_LANES_BLOCK}.{lane}.{_INLINE_PASSWORD_KEY}'. The value "
                f"must live only in {self.credentials_path} (mode 0600), "
                f"referenced from config by '{_PASSWORD_REF_KEY}'. Remove it "
                f"and {_remediation(lane)}.",
                error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            )

        resolved: list[str] = []
        for key in (_USERNAME_KEY, _PASSWORD_REF_KEY):
            if key not in block:
                raise ModelOnexError(
                    f"{self.config_path}: '{_LANES_BLOCK}.{lane}.{key}' is "
                    f"missing. To rewrite the entry, {_remediation(lane)}.",
                    error_code=EnumCoreErrorCode.MISSING_REQUIRED_PARAMETER,
                )
            value = block[key]
            if not isinstance(value, str) or not value.strip():
                raise ModelOnexError(
                    f"{self.config_path}: '{_LANES_BLOCK}.{lane}.{key}' must "
                    f"be a non-empty string. To rewrite the entry, "
                    f"{_remediation(lane)}.",
                    error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
                )
            resolved.append(value)
        return resolved[0], resolved[1]
