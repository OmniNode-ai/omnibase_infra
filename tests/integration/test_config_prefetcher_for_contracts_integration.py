# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for ConfigPrefetcher contract-directory prefetch."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.config_prefetcher import ConfigPrefetcher
from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)


class _RecordingResolver:
    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = secrets
        self.calls: list[tuple[str, str]] = []

    def get_secret_sync(
        self,
        *,
        secret_name: str,
        secret_path: str,
    ) -> SecretStr | None:
        self.calls.append((secret_name, secret_path))
        value = self._secrets.get(secret_name)
        return SecretStr(value) if value is not None else None


def test_prefetch_for_contracts_extracts_contract_yaml_and_resolves_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        """
metadata:
  transport_type: db
""",
        encoding="utf-8",
    )

    for key in TransportConfigMap.keys_for_transport(EnumInfraTransportType.DATABASE):
        monkeypatch.delenv(key, raising=False)

    resolver = _RecordingResolver({"POSTGRES_HOST": "db.integration.local"})
    result = ConfigPrefetcher(handler=resolver).prefetch_for_contracts(tmp_path)

    assert result.specs_attempted == 1
    assert result.resolved["POSTGRES_HOST"].get_secret_value() == "db.integration.local"
    assert ("POSTGRES_HOST", "/shared/db/") in resolver.calls
