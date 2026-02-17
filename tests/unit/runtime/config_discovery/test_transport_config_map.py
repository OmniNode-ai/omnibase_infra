# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for TransportConfigMap (OMN-2287)."""

from __future__ import annotations

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.models.model_transport_config_spec import (
    ModelTransportConfigSpec,
)
from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)


class TestTransportConfigMap:
    """Tests for TransportConfigMap."""

    def setup_method(self) -> None:
        self.tcm = TransportConfigMap()

    # --- keys_for_transport ---

    def test_database_has_keys(self) -> None:
        """DATABASE transport should have standard PostgreSQL keys."""
        keys = TransportConfigMap.keys_for_transport(EnumInfraTransportType.DATABASE)
        assert "POSTGRES_DSN" in keys
        assert "POSTGRES_POOL_MIN" in keys
        assert len(keys) > 0

    def test_kafka_has_keys(self) -> None:
        """KAFKA transport should have bootstrap servers and group ID."""
        keys = TransportConfigMap.keys_for_transport(EnumInfraTransportType.KAFKA)
        assert "KAFKA_BOOTSTRAP_SERVERS" in keys
        assert "KAFKA_GROUP_ID" in keys

    def test_inmemory_has_no_keys(self) -> None:
        """INMEMORY transport has no external config."""
        keys = TransportConfigMap.keys_for_transport(EnumInfraTransportType.INMEMORY)
        assert keys == ()

    def test_runtime_has_no_keys(self) -> None:
        """RUNTIME transport has no external config."""
        keys = TransportConfigMap.keys_for_transport(EnumInfraTransportType.RUNTIME)
        assert keys == ()

    def test_infisical_has_keys(self) -> None:
        """INFISICAL transport should have client credentials."""
        keys = TransportConfigMap.keys_for_transport(EnumInfraTransportType.INFISICAL)
        assert "INFISICAL_ADDR" in keys
        assert "INFISICAL_CLIENT_ID" in keys

    # --- shared_spec ---

    def test_shared_spec_path(self) -> None:
        """Shared spec should use /shared/<transport>/ path."""
        spec = self.tcm.shared_spec(EnumInfraTransportType.DATABASE)
        assert spec.infisical_folder == "/shared/db/"
        assert spec.transport_type == EnumInfraTransportType.DATABASE
        assert spec.service_slug == ""
        assert len(spec.keys) > 0

    def test_shared_spec_consul(self) -> None:
        """Consul shared spec should use consul slug."""
        spec = self.tcm.shared_spec(EnumInfraTransportType.CONSUL)
        assert spec.infisical_folder == "/shared/consul/"
        assert "CONSUL_HOST" in spec.keys

    def test_shared_spec_required(self) -> None:
        """Shared spec should propagate required flag."""
        spec = self.tcm.shared_spec(EnumInfraTransportType.DATABASE, required=True)
        assert spec.required is True

    # --- service_spec ---

    def test_service_spec_path(self) -> None:
        """Service spec should use /services/<service>/<transport>/ path."""
        spec = self.tcm.service_spec(
            EnumInfraTransportType.DATABASE,
            service_slug="omnibase-runtime",
        )
        assert spec.infisical_folder == "/services/omnibase-runtime/db/"
        assert spec.service_slug == "omnibase-runtime"

    def test_service_spec_empty_name_raises(self) -> None:
        """Service spec should raise ValueError for empty service name."""
        with pytest.raises(ValueError, match="service_slug must not be empty"):
            self.tcm.service_spec(
                EnumInfraTransportType.DATABASE,
                service_slug="",
            )

    # --- specs_for_transports ---

    def test_specs_for_multiple_transports(self) -> None:
        """Should build specs for multiple transports."""
        specs = self.tcm.specs_for_transports(
            [EnumInfraTransportType.DATABASE, EnumInfraTransportType.KAFKA]
        )
        assert len(specs) == 2
        assert specs[0].transport_type == EnumInfraTransportType.DATABASE
        assert specs[1].transport_type == EnumInfraTransportType.KAFKA

    def test_specs_skip_no_key_transports(self) -> None:
        """Should skip transports with no config keys."""
        specs = self.tcm.specs_for_transports(
            [
                EnumInfraTransportType.DATABASE,
                EnumInfraTransportType.INMEMORY,
                EnumInfraTransportType.RUNTIME,
            ]
        )
        # Only DATABASE should be included
        assert len(specs) == 1
        assert specs[0].transport_type == EnumInfraTransportType.DATABASE

    def test_specs_with_service_slug(self) -> None:
        """Should use per-service paths when service_slug is provided."""
        specs = self.tcm.specs_for_transports(
            [EnumInfraTransportType.DATABASE],
            service_slug="my-service",
        )
        assert len(specs) == 1
        assert "/services/my-service/db/" in specs[0].infisical_folder

    # --- all_shared_specs ---

    def test_all_shared_specs(self) -> None:
        """Should return specs for all transports with keys."""
        specs = self.tcm.all_shared_specs()
        # At least DATABASE, KAFKA, CONSUL, INFISICAL should be present
        transport_types = {s.transport_type for s in specs}
        assert EnumInfraTransportType.DATABASE in transport_types
        assert EnumInfraTransportType.KAFKA in transport_types
        assert EnumInfraTransportType.CONSUL in transport_types
        # INMEMORY and RUNTIME should NOT be present
        assert EnumInfraTransportType.INMEMORY not in transport_types
        assert EnumInfraTransportType.RUNTIME not in transport_types

    # --- ModelTransportConfigSpec immutability ---

    def test_spec_is_frozen(self) -> None:
        """ModelTransportConfigSpec should be immutable."""
        spec = self.tcm.shared_spec(EnumInfraTransportType.DATABASE)
        with pytest.raises(Exception):
            spec.infisical_folder = "/modified/"  # type: ignore[misc]

    def test_all_transport_types_have_mapping(self) -> None:
        """Every EnumInfraTransportType should be handled (even if empty keys)."""
        for transport in EnumInfraTransportType:
            # Should not raise
            keys = TransportConfigMap.keys_for_transport(transport)
            assert isinstance(keys, tuple)
