# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for infra-test CLI helper functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.cli.infra_test._helpers import (
    get_broker,
    get_postgres_dsn,
)


@pytest.mark.unit
class TestGetBroker:
    """Test Kafka broker address resolution."""

    def test_default_value(self) -> None:
        """Returns localhost:19092 when env var is unset."""
        with patch.dict("os.environ", {}, clear=True):
            assert get_broker() == "localhost:19092"

    def test_from_env(self) -> None:
        """Returns value from KAFKA_BOOTSTRAP_SERVERS."""
        with patch.dict("os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "broker:9092"}):
            assert get_broker() == "broker:9092"


@pytest.mark.unit
class TestGetPostgresDsn:
    """Test PostgreSQL DSN resolution from OMNIBASE_INFRA_DB_URL."""

    def test_returns_env_var_directly(self) -> None:
        """Returns OMNIBASE_INFRA_DB_URL value as-is."""
        url = "postgresql://myuser:mypass@myhost:5433/mydb"
        env = {"OMNIBASE_INFRA_DB_URL": url}
        with patch.dict("os.environ", env, clear=True):
            dsn = get_postgres_dsn()
            assert dsn == url

    def test_raises_when_env_var_not_set(self) -> None:
        """Raises ValueError when OMNIBASE_INFRA_DB_URL is not set."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OMNIBASE_INFRA_DB_URL is required"):
                get_postgres_dsn()

    def test_invalid_scheme_raises(self) -> None:
        """Rejects non-postgresql:// schemes."""
        env = {"OMNIBASE_INFRA_DB_URL": "mysql://user:pass@host:3306/db"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match=r"(?i)invalid.*scheme"):
                get_postgres_dsn()

    def test_missing_database_name_raises(self) -> None:
        """Rejects DSN with no database name in path."""
        env = {"OMNIBASE_INFRA_DB_URL": "postgresql://user:pass@host:5432/"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="missing a database name"):
                get_postgres_dsn()

    def test_missing_database_name_no_slash_raises(self) -> None:
        """Rejects DSN with no path at all."""
        env = {"OMNIBASE_INFRA_DB_URL": "postgresql://user:pass@host:5432"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="missing a database name"):
                get_postgres_dsn()
