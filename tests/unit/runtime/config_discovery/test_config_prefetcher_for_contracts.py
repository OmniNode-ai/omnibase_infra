# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ConfigPrefetcher.prefetch_for_contracts (OMN-10586)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import SecretStr

from omnibase_infra.runtime.config_discovery.config_prefetcher import ConfigPrefetcher


class _FakeSyncResolver:
    """Minimal ProtocolSecretResolver backed by a dict for testing."""

    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = secrets

    def get_secret_sync(
        self,
        *,
        secret_name: str,
        secret_path: str,
    ) -> SecretStr | None:
        value = self._secrets.get(secret_name)
        return SecretStr(value) if value is not None else None


class TestPrefetchForContracts:
    """Tests for ConfigPrefetcher.prefetch_for_contracts."""

    def _write_contract(self, directory: Path, name: str, content: str) -> Path:
        path = directory / name
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def test_prefetch_for_contracts_resolves_db_keys(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should extract DB transport requirements from contract and resolve keys."""
        self._write_contract(
            tmp_path,
            "contract.yaml",
            """\
            metadata:
              transport_type: db
            """,
        )

        # Remove POSTGRES_HOST from env so it must come from the resolver
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("POSTGRES_PORT", raising=False)

        resolver = _FakeSyncResolver({"POSTGRES_HOST": "db.test.local"})
        prefetcher = ConfigPrefetcher(handler=resolver)

        result = prefetcher.prefetch_for_contracts(tmp_path)

        assert result.specs_attempted == 1
        assert "POSTGRES_HOST" in result.resolved
        assert result.resolved["POSTGRES_HOST"].get_secret_value() == "db.test.local"

    def test_prefetch_for_contracts_two_contracts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two contracts with different transports should each contribute specs."""
        self._write_contract(
            tmp_path,
            "contract_a.yaml",
            """\
            metadata:
              transport_type: db
            """,
        )
        self._write_contract(
            tmp_path,
            "contract_b.yaml",
            """\
            metadata:
              transport_type: kafka
            """,
        )

        # Remove env keys so they go through the resolver
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.runtime.config_discovery.transport_config_map import (
            TransportConfigMap,
        )

        tcm = TransportConfigMap()
        for transport in (
            EnumInfraTransportType.DATABASE,
            EnumInfraTransportType.KAFKA,
        ):
            for key in tcm.keys_for_transport(transport):
                monkeypatch.delenv(key, raising=False)

        resolver = _FakeSyncResolver(
            {
                "POSTGRES_HOST": "pg.test.local",
                "KAFKA_GROUP_ID": "test-group",
            }
        )
        prefetcher = ConfigPrefetcher(handler=resolver)

        result = prefetcher.prefetch_for_contracts(tmp_path)

        # specs_attempted = 1 per unique transport type (db + kafka = 2)
        assert result.specs_attempted == 2
        assert "POSTGRES_HOST" in result.resolved
        assert result.resolved["POSTGRES_HOST"].get_secret_value() == "pg.test.local"
        assert "KAFKA_GROUP_ID" in result.resolved
        assert result.resolved["KAFKA_GROUP_ID"].get_secret_value() == "test-group"

    def test_prefetch_for_contracts_missing_keys_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keys not in resolver and not in env should appear in result.missing."""
        self._write_contract(
            tmp_path,
            "contract.yaml",
            """\
            metadata:
              transport_type: db
            """,
        )

        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.runtime.config_discovery.transport_config_map import (
            TransportConfigMap,
        )

        for key in TransportConfigMap.keys_for_transport(
            EnumInfraTransportType.DATABASE
        ):
            monkeypatch.delenv(key, raising=False)

        resolver = _FakeSyncResolver({})
        prefetcher = ConfigPrefetcher(handler=resolver)

        result = prefetcher.prefetch_for_contracts(tmp_path)

        assert len(result.missing) > 0
        assert "POSTGRES_HOST" in result.missing

    def test_prefetch_for_contracts_empty_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """Empty contracts directory should return zero-count result without error."""
        resolver = _FakeSyncResolver({})
        prefetcher = ConfigPrefetcher(handler=resolver)

        result = prefetcher.prefetch_for_contracts(tmp_path)

        assert result.specs_attempted == 0
        assert result.success_count == 0
        assert result.failure_count == 0
