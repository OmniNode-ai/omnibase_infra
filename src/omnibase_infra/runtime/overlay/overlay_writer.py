# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

import yaml

from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)

logger = logging.getLogger(__name__)

_SECRET_PATTERN = re.compile(r"PASSWORD|SECRET|TOKEN|KEY|CREDENTIAL", re.IGNORECASE)


class OverlayWriter:
    def __init__(self) -> None:
        self._transport_map = TransportConfigMap()

    def write(
        self,
        *,
        env_dict: dict[str, str],
        output_path: Path,
        environment: str,
        scope: str,
    ) -> None:
        transports = self._classify_by_transport(env_dict)

        secret_count = sum(1 for key in env_dict if _SECRET_PATTERN.search(key))
        if secret_count:
            logger.warning("Overlay contains %d secret-pattern keys", secret_count)

        overlay_data = {
            "overlay_version": "1.0.0",
            "environment": environment,
            "scope": scope,
            "transports": transports,
        }

        # Atomic write: tempfile + rename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(output_path.parent), suffix=".yaml.tmp"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(
                    overlay_data, f, sort_keys=True, default_flow_style=False
                )
            tmp_path.chmod(0o600)
            tmp_path.replace(output_path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _classify_by_transport(
        self, env_dict: dict[str, str]
    ) -> dict[str, dict[str, str]]:
        """Classify env keys into transport buckets. Sorted iteration for determinism."""
        from omnibase_infra.enums import EnumInfraTransportType

        result: dict[str, dict[str, str]] = {}
        classified_keys: set[str] = set()

        # Sorted enum iteration ensures deterministic output regardless of enum definition order
        for transport in sorted(EnumInfraTransportType, key=lambda t: t.value):
            transport_keys = self._transport_map.keys_for_transport(transport)
            bucket: dict[str, str] = {}
            for key in sorted(transport_keys):
                if key in env_dict:
                    bucket[key] = env_dict[key]
                    classified_keys.add(key)
            if bucket:
                result[transport.value] = bucket

        # Unclassified keys in sorted order
        unclassified = {
            k: env_dict[k] for k in sorted(env_dict.keys()) if k not in classified_keys
        }
        if unclassified:
            result["custom"] = unclassified

        return result
