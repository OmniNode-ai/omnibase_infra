# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read contract-declared descriptor values for the coding-agent invoke effect.

The invoke effect's ``descriptor.agent_credential_home`` is the single source of
truth for the HOME the spawned claude/codex subprocess must use to find its
ambient OAuth credentials (OMN-13247 Phase B dev-verify fix). It is declared with
the ``${env.VAR}`` overlay convention so an operator overlay / the service env
sets the real path per lane — never a hardcoded ``/home/omniinfra`` in source.

The handler resolves it fail-closed: an unset/empty value raises rather than
letting the child inherit the runtime container's ``HOME`` (``/root`` under the
dev runtime, which has no bind-mounted creds).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs


def _load_contract(contract_path: Path) -> dict[str, object]:
    # ONEX_EXCLUDE: io_audit - Module-level contract load keeps handler policy contract-owned
    with contract_path.open(encoding="utf-8") as contract_file:
        raw = yaml.safe_load(contract_file)
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract_path} must contain a mapping")
    return raw


def contract_agent_credential_home(contract_path: Path) -> str:
    """Return the resolved ``descriptor.agent_credential_home`` for the contract.

    The value is contract-declared (overridable by an operator overlay contract)
    via the ``${env.VAR}`` convention — never hardcoded in source. Fails closed:
    raises ``ValueError`` when the field is absent or resolves to an empty string,
    so the spawned subprocess never silently inherits the runtime container's HOME
    (``/root`` under the dev runtime, where no creds are bind-mounted).
    """
    raw = _load_contract(contract_path)
    descriptor = raw.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"contract {contract_path} must declare a descriptor mapping with "
            "agent_credential_home"
        )
    declared = descriptor.get("agent_credential_home")
    if not isinstance(declared, str):
        raise ValueError(
            f"contract {contract_path} must declare a string "
            "descriptor.agent_credential_home (the ${env.CODING_AGENT_CRED_HOME} "
            "overlay value the subprocess uses as HOME)"
        )
    resolved = expand_contract_env_refs(declared).strip()
    if not resolved:
        raise ValueError(
            "descriptor.agent_credential_home resolved empty — set "
            "CODING_AGENT_CRED_HOME (the credential-home path the claude/codex "
            "subprocess uses as HOME). The coding-agent invoke effect fails closed "
            "rather than let the child inherit the runtime container's HOME (/root "
            "under the dev runtime has no bind-mounted creds)."
        )
    return resolved


__all__: list[str] = ["contract_agent_credential_home"]
