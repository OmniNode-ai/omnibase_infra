# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Refuse to start in cloud mode while a setting holds a lab address (OMN-26709).

Cloud mode is a database-topology profile whose instance is not ``local``,
selected by ``ONEX_DATABASE_TOPOLOGY_PROFILE``. In that mode these settings must
not hold a lab address: every ``LLM_*_URL`` variable (the model-host list),
``INFISICAL_ADDR`` (the secrets store) and the content of the file named by
``RUNNER_FLEET_CONFIG_PATH`` (the runner fleet). A lab address is a lab-LAN
IP address or a lab host alias. The error names the setting, never the address.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import Path

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.runtime.model_cloud_mode_lab_address_request import (
    ModelCloudModeLabAddressRequest,
)
from omnibase_infra.models.runtime.model_cloud_mode_lab_address_result import (
    ModelCloudModeLabAddressResult,
)
from omnibase_infra.topology.application_database import (
    TOPOLOGY_PROFILE_INSTANCE_MAP,
)

TOPOLOGY_PROFILE_ENV = "ONEX_DATABASE_TOPOLOGY_PROFILE"
FLEET_PATH_ENV = "RUNNER_FLEET_CONFIG_PATH"
SECRETS_STORE_ENV = "INFISICAL_ADDR"

_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
#: Lab host aliases: a numbered lab host, and the lab overlay's name.
_ALIAS = re.compile(r"omni-\d+|\bh\d{3}\b|onex-l[a]b")


def holds_lab_address(text: str) -> bool:
    """True when ``text`` names a lab LAN IPv4 address or a lab host alias.

    The lab LAN is the private range a home or office router hands out: private,
    but not loopback, link-local or the 10/8 and 172.16/12 blocks a cloud VPC uses.
    """
    if _ALIAS.search(text):
        return True
    for token in _IPV4.findall(text):
        try:
            ip = ip_address(token)
        except ValueError:
            continue
        if (
            ip.is_private
            and not (ip.is_loopback or ip.is_link_local)
            and ip.packed[0] not in (10, 172)
        ):
            return True
    return False


class CloudModeLabAddressError(RuntimeHostError):
    """Raised at boot when a cloud-mode setting still holds a lab address."""

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONFIGURATION_ERROR,
            context=context,
            **extra_context,
        )


def is_cloud_mode(profile: str) -> bool:
    """A known topology profile whose instance is not the local one."""
    instance = TOPOLOGY_PROFILE_INSTANCE_MAP.get(profile.strip())
    return instance is not None and instance != "local"


class HandlerCloudLabGuard:
    """Raises :class:`CloudModeLabAddressError` for a lab address in cloud mode."""

    def handle(
        self, request: ModelCloudModeLabAddressRequest
    ) -> ModelCloudModeLabAddressResult:
        env = dict(request.environment)
        if not is_cloud_mode(env.get(TOPOLOGY_PROFILE_ENV, "")):
            return ModelCloudModeLabAddressResult(cloud_mode=False)
        offenders = [
            name
            for name, value in sorted(env.items())
            if (
                name == SECRETS_STORE_ENV
                or (name[:4] == "LLM_" and name[-4:] == "_URL")
            )
            and holds_lab_address(value)
        ]
        fleet_path = env.get(FLEET_PATH_ENV, "").strip()
        if fleet_path and holds_lab_address(_read_text(Path(fleet_path))):
            offenders.append(FLEET_PATH_ENV)
        if offenders:
            raise CloudModeLabAddressError(
                "cloud mode refuses to start: these settings hold a lab address: "
                + ", ".join(offenders),
                settings=tuple(offenders),
            )
        return ModelCloudModeLabAddressResult(cloud_mode=True)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


_GUARD = HandlerCloudLabGuard()


def refuse_in_cloud_mode(env: Mapping[str, str]) -> int:
    """Entrypoint step: 0 to boot, 1 (with the named error on stderr) to abort."""
    try:
        _GUARD.handle(
            ModelCloudModeLabAddressRequest(environment=tuple(sorted(env.items())))
        )
    except CloudModeLabAddressError as exc:
        sys.stderr.write(f"[entrypoint] ERROR: CloudModeLabAddressError: {exc}\n")
        return 1
    return 0
