# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelPostgresPoolConfig SSL fields (OMN-14597).

Covers:
- Default (unset) ssl_mode/ssl_ca_file — backward-compatible, no behavior
  change for local Docker/dev.
- Fail-fast validation: unknown ssl_mode, verify-ca/verify-full without an
  explicit ssl_ca_file.
- from_env() / from_dsn() thread POSTGRES_SSL_MODE / POSTGRES_SSL_CA_FILE
  (and the equivalent kwargs) through to the constructed config.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)

pytestmark = pytest.mark.unit


class TestSslDefaults:
    def test_ssl_mode_defaults_to_empty_string(self) -> None:
        config = ModelPostgresPoolConfig(host="db.example.com", database="testdb")
        assert config.ssl_mode == ""
        assert config.ssl_ca_file == ""


class TestSslModeValidation:
    def test_unknown_ssl_mode_raises(self) -> None:
        with pytest.raises(ValidationError, match="ssl_mode"):
            ModelPostgresPoolConfig(
                host="db.example.com", database="testdb", ssl_mode="verify-everything"
            )

    @pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require"])
    def test_no_ca_file_modes_do_not_require_ssl_ca_file(self, mode: str) -> None:
        config = ModelPostgresPoolConfig(
            host="db.example.com", database="testdb", ssl_mode=mode
        )
        assert config.ssl_mode == mode
        assert config.ssl_ca_file == ""

    @pytest.mark.parametrize("mode", ["verify-ca", "verify-full"])
    def test_verify_modes_without_ca_file_raise(self, mode: str) -> None:
        with pytest.raises(ValidationError, match="ssl_ca_file"):
            ModelPostgresPoolConfig(
                host="db.example.com", database="testdb", ssl_mode=mode
            )

    @pytest.mark.parametrize("mode", ["verify-ca", "verify-full"])
    def test_verify_modes_with_ca_file_succeed(self, mode: str) -> None:
        config = ModelPostgresPoolConfig(
            host="db.example.com",
            database="testdb",
            ssl_mode=mode,
            ssl_ca_file="/etc/rds/ca.pem",
        )
        assert config.ssl_mode == mode
        assert config.ssl_ca_file == "/etc/rds/ca.pem"


class TestFromDsnSslPassthrough:
    def test_from_dsn_default_leaves_ssl_unset(self) -> None:
        config = ModelPostgresPoolConfig.from_dsn(
            "postgresql://user:pass@db.example.com:5432/testdb"
        )
        assert config.ssl_mode == ""
        assert config.ssl_ca_file == ""

    def test_from_dsn_threads_ssl_kwargs_through(self) -> None:
        config = ModelPostgresPoolConfig.from_dsn(
            "postgresql://user:pass@db.example.com:5432/testdb",
            ssl_mode="verify-full",
            ssl_ca_file="/etc/rds/ca.pem",
        )
        assert config.ssl_mode == "verify-full"
        assert config.ssl_ca_file == "/etc/rds/ca.pem"


class TestFromEnvSslPassthrough:
    def test_from_env_reads_ssl_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "OMNIBASE_INFRA_DB_URL",
            "postgresql://user:pass@db.example.com:5432/testdb",
        )
        monkeypatch.setenv("POSTGRES_SSL_MODE", "verify-full")
        monkeypatch.setenv("POSTGRES_SSL_CA_FILE", "/etc/rds/ca.pem")

        config = ModelPostgresPoolConfig.from_env()

        assert config.ssl_mode == "verify-full"
        assert config.ssl_ca_file == "/etc/rds/ca.pem"

    def test_from_env_defaults_ssl_unset_when_env_vars_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "OMNIBASE_INFRA_DB_URL",
            "postgresql://user:pass@db.example.com:5432/testdb",
        )
        monkeypatch.delenv("POSTGRES_SSL_MODE", raising=False)
        monkeypatch.delenv("POSTGRES_SSL_CA_FILE", raising=False)

        config = ModelPostgresPoolConfig.from_env()

        assert config.ssl_mode == ""
        assert config.ssl_ca_file == ""
