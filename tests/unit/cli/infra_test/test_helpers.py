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
