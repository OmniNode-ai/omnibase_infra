# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for infra-test CLI helper functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.cli.infra_test._helpers import (
    get_broker,
    get_consul_addr,
    get_postgres_dsn,
)


@pytest.mark.unit
class TestGetBroker:
    """Test Kafka broker address resolution."""

    def test_default_value(self) -> None:
        """Returns localhost:29092 when env var is unset."""
        with patch.dict("os.environ", {}, clear=True):
            assert get_broker() == "localhost:29092"

    def test_from_env(self) -> None:
        """Returns value from KAFKA_BOOTSTRAP_SERVERS."""
        with patch.dict("os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "broker:9092"}):
            assert get_broker() == "broker:9092"


@pytest.mark.unit
class TestGetConsulAddr:
    """Test Consul address resolution."""

    def test_default_value(self) -> None:
        """Returns http://localhost:8500 when env vars are unset."""
        with patch.dict("os.environ", {}, clear=True):
            assert get_consul_addr() == "http://localhost:8500"

    def test_custom_host_and_port(self) -> None:
        """Respects CONSUL_HOST and CONSUL_PORT."""
        env = {"CONSUL_HOST": "consul.local", "CONSUL_PORT": "28500"}
        with patch.dict("os.environ", env, clear=True):
            assert get_consul_addr() == "http://consul.local:28500"

    def test_custom_scheme(self) -> None:
        """Respects CONSUL_SCHEME for HTTPS."""
        with patch.dict("os.environ", {"CONSUL_SCHEME": "https"}, clear=True):
            assert get_consul_addr() == "https://localhost:8500"

    def test_invalid_scheme_raises(self) -> None:
        """Rejects non-http/https schemes."""
        with patch.dict("os.environ", {"CONSUL_SCHEME": "ftp"}, clear=True):
            with pytest.raises(ValueError, match="CONSUL_SCHEME must be"):
                get_consul_addr()

    def test_non_numeric_port_raises(self) -> None:
        """Rejects non-numeric CONSUL_PORT."""
        with patch.dict("os.environ", {"CONSUL_PORT": "abc"}, clear=True):
            with pytest.raises(ValueError, match="CONSUL_PORT must be numeric"):
                get_consul_addr()


@pytest.mark.unit
class TestGetPostgresDsn:
    """Test PostgreSQL DSN construction."""

    def test_default_value_raises_without_database(self) -> None:
        """Raises ValueError when no database is configured."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No database configured"):
                get_postgres_dsn()

    def test_default_value_with_database(self) -> None:
        """Returns default DSN when POSTGRES_DATABASE is set."""
        with patch.dict(
            "os.environ", {"POSTGRES_DATABASE": "omnibase_infra"}, clear=True
        ):
            dsn = get_postgres_dsn()
            assert (
                dsn
                == "postgresql://postgres:test-password@localhost:5433/omnibase_infra"
            )

    def test_custom_env(self) -> None:
        """Respects all POSTGRES_* env vars."""
        env = {
            "POSTGRES_HOST": "db.local",
            "POSTGRES_PORT": "5436",
            "POSTGRES_DATABASE": "mydb",
            "POSTGRES_USER": "admin",
            "POSTGRES_PASSWORD": "secret",
        }
        with patch.dict("os.environ", env, clear=True):
            dsn = get_postgres_dsn()
            assert dsn == "postgresql://admin:secret@db.local:5436/mydb"

    def test_special_chars_in_password(self) -> None:
        """URL-encodes special characters in password."""
        env = {"POSTGRES_PASSWORD": "p@ss:w/rd", "POSTGRES_DATABASE": "testdb"}
        with patch.dict("os.environ", env, clear=True):
            dsn = get_postgres_dsn()
            assert "p%40ss%3Aw%2Frd" in dsn
            assert "p@ss:w/rd" not in dsn

    def test_host_with_at_raises(self) -> None:
        """Rejects POSTGRES_HOST containing '@'."""
        env = {"POSTGRES_HOST": "evil@host", "POSTGRES_DATABASE": "testdb"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="POSTGRES_HOST contains '@'"):
                get_postgres_dsn()

    def test_non_numeric_port_raises(self) -> None:
        """Rejects non-numeric POSTGRES_PORT."""
        env = {"POSTGRES_PORT": "abc", "POSTGRES_DATABASE": "testdb"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="POSTGRES_PORT must be numeric"):
                get_postgres_dsn()
