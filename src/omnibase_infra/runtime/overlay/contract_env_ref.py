# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared ``${env.VAR}`` overlay expansion for contract string values.

This is the sanctioned overlay-resolution surface for the ``${env.VAR}`` /
``${env.VAR:default}`` convention that contract descriptors use to bind a
lane-specific value (a workspace root, a credential home, a contracts dir) from
the operator environment WITHOUT hardcoding the value in source. It lives in the
overlay package — the one module family allowed to read ``os.environ`` — so node
contract helpers (e.g. node_coding_agent_invoke_effect's contract_descriptor) call
through here instead of each duplicating an env read.

An unset var with no default expands to the empty string so the caller's
fail-closed check rejects it (rather than leaving a literal placeholder).
"""

from __future__ import annotations

import os
import re

# ``${env.VAR}`` / ``${env.VAR:default}`` — the same env-overlay convention
# node_contract_loader_effect uses for paths.
_ENV_REF = re.compile(
    r"\$\{env\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>[^}]*))?\}"
)


def expand_contract_env_refs(value: str) -> str:
    """Expand ``${env.VAR}`` / ``${env.VAR:default}`` references in ``value``.

    Resolves each reference from the operator environment; an unset var with no
    inline default expands to the empty string (so the caller fails closed rather
    than passing a literal ``${env.…}`` placeholder downstream).
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        return os.environ.get(name, default if default is not None else "")

    return _ENV_REF.sub(_sub, value)


__all__: list[str] = ["expand_contract_env_refs"]
