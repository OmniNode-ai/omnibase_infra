# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for tiered resolution contract YAML loader.

Tests the Phase 7 Part 2 (OMN-2896) contract loader that reads
``tiered_resolution`` and ``trust_domains`` sections from ONEX contract
YAML files.

Test categories:
    - Tiered resolution config parsing (valid, invalid, edge cases)
    - Trust domain config parsing (valid, invalid, edge cases)
    - Backward compatibility (existing contracts without new sections)
    - Bus descriptor bridging
    - Combined loader (load_tiered_resolution_from_contract)
    - Error handling (invalid tiers, missing fields, bad YAML)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.contract_loaders.tiered_resolution_contract_loader import (
    bridge_trust_domains_to_bus_descriptors,
    load_tiered_resolution_configs,
    load_tiered_resolution_from_contract,
    load_trust_domain_configs,
)
from omnibase_infra.services.resolution.model_tiered_resolution_config_local import (
    ModelTieredResolutionConfigLocal,
)
from omnibase_infra.services.resolution.model_trust_domain_config_local import (
    ModelTrustDomainConfigLocal,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Test YAML constants
# =============================================================================

CONTRACT_WITH_TIERED_RESOLUTION = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution:
      min_tier: "local_exact"
      max_tier: "org_trusted"
      require_proofs: ["node_identity", "capability_attested"]
      classification: "internal"
  - alias: "cache"
    capability: "cache.kv"
    tiered_resolution:
      min_tier: "local_exact"
      max_tier: "local_compatible"
"""

CONTRACT_WITH_TRUST_DOMAINS = """\
name: "test_node"
trust_domains:
  - domain_id: "local.default"
    tier: "local_exact"
  - domain_id: "org.omninode"
    tier: "org_trusted"
    trust_root_ref: "secrets://keys/org-omninode-trust-root"
"""

CONTRACT_WITH_BOTH = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution:
      min_tier: "local_exact"
      max_tier: "org_trusted"
      require_proofs: ["node_identity"]
      classification: "internal"
trust_domains:
  - domain_id: "local.default"
    tier: "local_exact"
  - domain_id: "org.omninode"
    tier: "org_trusted"
    trust_root_ref: "secrets://keys/org-omninode-trust-root"
"""

CONTRACT_WITHOUT_NEW_SECTIONS = """\
name: "existing_node"
version: "1.0.0"
dependencies:
  - name: "protocol_event_bus"
    type: "protocol"
    class_name: "ProtocolEventBus"
    module: "omnibase_spi.protocols"
"""

CONTRACT_EMPTY = ""

CONTRACT_WITH_INVALID_TIER = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution:
      min_tier: "nonexistent_tier"
"""

CONTRACT_WITH_INVALID_PROOF = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution:
      require_proofs: ["invalid_proof_type"]
"""

CONTRACT_WITH_INVALID_CLASSIFICATION = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution:
      classification: "top_secret"
"""

CONTRACT_WITH_INVALID_TIER_RANGE = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution:
      min_tier: "org_trusted"
      max_tier: "local_exact"
"""

CONTRACT_WITH_TRUST_DOMAIN_MISSING_ID = """\
name: "test_node"
trust_domains:
  - tier: "local_exact"
"""

CONTRACT_WITH_TRUST_DOMAIN_MISSING_TIER = """\
name: "test_node"
trust_domains:
  - domain_id: "local.default"
"""

CONTRACT_WITH_TRUST_DOMAIN_INVALID_TIER = """\
name: "test_node"
trust_domains:
  - domain_id: "local.default"
    tier: "nonexistent_tier"
"""

CONTRACT_DEPS_NO_ALIAS = """\
name: "test_node"
dependencies:
  - capability: "database.relational"
    tiered_resolution:
      min_tier: "local_exact"
"""

CONTRACT_DEPS_MIXED = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution:
      min_tier: "local_exact"
  - alias: "bus"
    capability: "messaging"
  - alias: "kv"
    capability: "cache.kv"
    tiered_resolution:
      max_tier: "local_compatible"
"""

CONTRACT_MINIMAL_TIERED = """\
name: "test_node"
dependencies:
  - alias: "db"
    capability: "database.relational"
    tiered_resolution: {}
"""


# =============================================================================
# Fixtures
# =============================================================================


def _write_contract(tmp_path: Path, content: str) -> Path:
    """Write a contract YAML to a temp file and return its path."""
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(content)
    return contract_file


# =============================================================================
# Tests: Tiered Resolution Config Parsing
# =============================================================================


class TestLoadTieredResolutionConfigs:
    """Tests for load_tiered_resolution_configs()."""

    def test_full_tiered_resolution(self, tmp_path: Path) -> None:
        """Load a contract with full tiered_resolution blocks."""
        path = _write_contract(tmp_path, CONTRACT_WITH_TIERED_RESOLUTION)
        configs = load_tiered_resolution_configs(path)

        assert len(configs) == 2
        assert "db" in configs
        assert "cache" in configs

        db_config = configs["db"]
        assert db_config.min_tier == "local_exact"
        assert db_config.max_tier == "org_trusted"
        assert db_config.require_proofs == ("node_identity", "capability_attested")
        assert db_config.classification == "internal"

        cache_config = configs["cache"]
        assert cache_config.min_tier == "local_exact"
        assert cache_config.max_tier == "local_compatible"
        assert cache_config.require_proofs == ()
        assert cache_config.classification is None

    def test_minimal_tiered_resolution(self, tmp_path: Path) -> None:
        """Empty tiered_resolution dict produces defaults."""
        path = _write_contract(tmp_path, CONTRACT_MINIMAL_TIERED)
        configs = load_tiered_resolution_configs(path)

        assert len(configs) == 1
        config = configs["db"]
        assert config.min_tier is None
        assert config.max_tier is None
        assert config.require_proofs == ()
        assert config.classification is None

    def test_no_alias_uses_index(self, tmp_path: Path) -> None:
        """Dependencies without alias use their index as key."""
        path = _write_contract(tmp_path, CONTRACT_DEPS_NO_ALIAS)
        configs = load_tiered_resolution_configs(path)

        assert len(configs) == 1
        assert "0" in configs

    def test_mixed_deps_with_and_without_tiered(self, tmp_path: Path) -> None:
        """Only deps with tiered_resolution are included."""
        path = _write_contract(tmp_path, CONTRACT_DEPS_MIXED)
        configs = load_tiered_resolution_configs(path)

        assert len(configs) == 2
        assert "db" in configs
        assert "kv" in configs
        assert "bus" not in configs

    def test_backward_compat_no_tiered_sections(self, tmp_path: Path) -> None:
        """Existing contract without tiered_resolution returns empty dict."""
        path = _write_contract(tmp_path, CONTRACT_WITHOUT_NEW_SECTIONS)
        configs = load_tiered_resolution_configs(path)
        assert configs == {}

    def test_empty_contract(self, tmp_path: Path) -> None:
        """Empty contract returns empty dict."""
        path = _write_contract(tmp_path, CONTRACT_EMPTY)
        configs = load_tiered_resolution_configs(path)
        assert configs == {}

    def test_invalid_tier_raises(self, tmp_path: Path) -> None:
        """Invalid tier name raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, CONTRACT_WITH_INVALID_TIER)
        with pytest.raises(ProtocolConfigurationError, match="Invalid tier"):
            load_tiered_resolution_configs(path)

    def test_invalid_proof_type_raises(self, tmp_path: Path) -> None:
        """Invalid proof type raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, CONTRACT_WITH_INVALID_PROOF)
        with pytest.raises(ProtocolConfigurationError, match="Invalid proof type"):
            load_tiered_resolution_configs(path)

    def test_invalid_classification_raises(self, tmp_path: Path) -> None:
        """Invalid classification raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, CONTRACT_WITH_INVALID_CLASSIFICATION)
        with pytest.raises(ProtocolConfigurationError, match="Invalid classification"):
            load_tiered_resolution_configs(path)

    def test_invalid_tier_range_raises(self, tmp_path: Path) -> None:
        """min_tier > max_tier raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, CONTRACT_WITH_INVALID_TIER_RANGE)
        with pytest.raises(ProtocolConfigurationError, match="TIER_RANGE_INVALID"):
            load_tiered_resolution_configs(path)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Nonexistent contract file raises ProtocolConfigurationError."""
        path = tmp_path / "nonexistent" / "contract.yaml"
        with pytest.raises(ProtocolConfigurationError, match="Contract file not found"):
            load_tiered_resolution_configs(path)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Invalid YAML syntax raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, "name: [unclosed bracket")
        with pytest.raises(ProtocolConfigurationError, match="YAML"):
            load_tiered_resolution_configs(path)


# =============================================================================
# Tests: Trust Domain Config Parsing
# =============================================================================


class TestLoadTrustDomainConfigs:
    """Tests for load_trust_domain_configs()."""

    def test_valid_trust_domains(self, tmp_path: Path) -> None:
        """Load valid trust domain declarations."""
        path = _write_contract(tmp_path, CONTRACT_WITH_TRUST_DOMAINS)
        domains = load_trust_domain_configs(path)

        assert len(domains) == 2

        assert domains[0].domain_id == "local.default"
        assert domains[0].tier == "local_exact"
        assert domains[0].trust_root_ref is None

        assert domains[1].domain_id == "org.omninode"
        assert domains[1].tier == "org_trusted"
        assert domains[1].trust_root_ref == "secrets://keys/org-omninode-trust-root"

    def test_backward_compat_no_trust_domains(self, tmp_path: Path) -> None:
        """Contract without trust_domains returns empty list."""
        path = _write_contract(tmp_path, CONTRACT_WITHOUT_NEW_SECTIONS)
        domains = load_trust_domain_configs(path)
        assert domains == []

    def test_empty_contract(self, tmp_path: Path) -> None:
        """Empty contract returns empty list."""
        path = _write_contract(tmp_path, CONTRACT_EMPTY)
        domains = load_trust_domain_configs(path)
        assert domains == []

    def test_missing_domain_id_raises(self, tmp_path: Path) -> None:
        """Trust domain without domain_id raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, CONTRACT_WITH_TRUST_DOMAIN_MISSING_ID)
        with pytest.raises(ProtocolConfigurationError, match="Missing 'domain_id'"):
            load_trust_domain_configs(path)

    def test_missing_tier_raises(self, tmp_path: Path) -> None:
        """Trust domain without tier raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, CONTRACT_WITH_TRUST_DOMAIN_MISSING_TIER)
        with pytest.raises(ProtocolConfigurationError, match="Missing 'tier'"):
            load_trust_domain_configs(path)

    def test_invalid_tier_raises(self, tmp_path: Path) -> None:
        """Trust domain with invalid tier raises ProtocolConfigurationError."""
        path = _write_contract(tmp_path, CONTRACT_WITH_TRUST_DOMAIN_INVALID_TIER)
        with pytest.raises(ProtocolConfigurationError, match="Invalid tier"):
            load_trust_domain_configs(path)


# =============================================================================
# Tests: Bus Descriptor Bridging
# =============================================================================


class TestBridgeTrustDomainsToBusDescriptors:
    """Tests for bridge_trust_domains_to_bus_descriptors()."""

    def test_local_domain_no_prefix(self) -> None:
        """Local domains get empty namespace prefix."""
        domains = [
            ModelTrustDomainConfigLocal(
                domain_id="local.default",
                tier="local_exact",
            )
        ]
        descriptors = bridge_trust_domains_to_bus_descriptors(domains)

        assert len(descriptors) == 1
        desc = descriptors[0]
        assert desc.bus_id == "bus.local.default"
        assert desc.trust_domain == "local.default"
        assert desc.transport_type == EnumInfraTransportType.KAFKA
        assert desc.namespace_prefix == ""

    def test_org_domain_gets_prefix(self) -> None:
        """Non-local domains get domain_id as namespace prefix."""
        domains = [
            ModelTrustDomainConfigLocal(
                domain_id="org.omninode",
                tier="org_trusted",
            )
        ]
        descriptors = bridge_trust_domains_to_bus_descriptors(domains)

        assert len(descriptors) == 1
        desc = descriptors[0]
        assert desc.bus_id == "bus.org.omninode"
        assert desc.trust_domain == "org.omninode"
        assert desc.namespace_prefix == "org.omninode."

    def test_federated_domain_gets_prefix(self) -> None:
        """Federated domains get domain_id as namespace prefix."""
        domains = [
            ModelTrustDomainConfigLocal(
                domain_id="fed.partner-a",
                tier="federated_trusted",
            )
        ]
        descriptors = bridge_trust_domains_to_bus_descriptors(domains)

        assert len(descriptors) == 1
        assert descriptors[0].namespace_prefix == "fed.partner-a."

    def test_empty_list_returns_empty(self) -> None:
        """Empty domain list returns empty descriptor list."""
        descriptors = bridge_trust_domains_to_bus_descriptors([])
        assert descriptors == []

    def test_multiple_domains(self) -> None:
        """Multiple domains produce one descriptor each."""
        domains = [
            ModelTrustDomainConfigLocal(domain_id="local.default", tier="local_exact"),
            ModelTrustDomainConfigLocal(domain_id="org.omninode", tier="org_trusted"),
            ModelTrustDomainConfigLocal(
                domain_id="fed.partner-a", tier="federated_trusted"
            ),
        ]
        descriptors = bridge_trust_domains_to_bus_descriptors(domains)
        assert len(descriptors) == 3
        assert descriptors[0].namespace_prefix == ""
        assert descriptors[1].namespace_prefix == "org.omninode."
        assert descriptors[2].namespace_prefix == "fed.partner-a."


# =============================================================================
# Tests: Combined Loader
# =============================================================================


class TestLoadTieredResolutionFromContract:
    """Tests for load_tiered_resolution_from_contract()."""

    def test_full_contract_with_both_sections(self, tmp_path: Path) -> None:
        """Load a contract with both tiered_resolution and trust_domains."""
        path = _write_contract(tmp_path, CONTRACT_WITH_BOTH)
        result = load_tiered_resolution_from_contract(path)

        assert len(result.tiered_configs) == 1
        assert "db" in result.tiered_configs
        assert result.tiered_configs["db"].min_tier == "local_exact"
        assert result.tiered_configs["db"].classification == "internal"

        assert len(result.trust_domains) == 2
        assert result.trust_domains[0].domain_id == "local.default"
        assert result.trust_domains[1].domain_id == "org.omninode"

        assert len(result.bus_descriptors) == 2
        assert result.bus_descriptors[0].namespace_prefix == ""
        assert result.bus_descriptors[1].namespace_prefix == "org.omninode."

    def test_backward_compat_existing_contract(self, tmp_path: Path) -> None:
        """Existing contract without new sections loads with empty results."""
        path = _write_contract(tmp_path, CONTRACT_WITHOUT_NEW_SECTIONS)
        result = load_tiered_resolution_from_contract(path)

        assert result.tiered_configs == {}
        assert result.trust_domains == []
        assert result.bus_descriptors == []

    def test_empty_contract(self, tmp_path: Path) -> None:
        """Empty contract file returns all-empty result."""
        path = _write_contract(tmp_path, CONTRACT_EMPTY)
        result = load_tiered_resolution_from_contract(path)

        assert result.tiered_configs == {}
        assert result.trust_domains == []
        assert result.bus_descriptors == []

    def test_only_tiered_resolution(self, tmp_path: Path) -> None:
        """Contract with only tiered_resolution (no trust_domains) works."""
        path = _write_contract(tmp_path, CONTRACT_WITH_TIERED_RESOLUTION)
        result = load_tiered_resolution_from_contract(path)

        assert len(result.tiered_configs) == 2
        assert result.trust_domains == []
        assert result.bus_descriptors == []

    def test_only_trust_domains(self, tmp_path: Path) -> None:
        """Contract with only trust_domains (no tiered_resolution) works."""
        path = _write_contract(tmp_path, CONTRACT_WITH_TRUST_DOMAINS)
        result = load_tiered_resolution_from_contract(path)

        assert result.tiered_configs == {}
        assert len(result.trust_domains) == 2
        assert len(result.bus_descriptors) == 2


# =============================================================================
# Tests: Model Validation
# =============================================================================


class TestModelTieredResolutionConfigLocal:
    """Tests for ModelTieredResolutionConfigLocal model."""

    def test_frozen(self) -> None:
        """Model is frozen (immutable)."""
        config = ModelTieredResolutionConfigLocal(min_tier="local_exact")
        with pytest.raises(Exception):
            config.min_tier = "org_trusted"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Model rejects extra fields."""
        with pytest.raises(Exception):
            ModelTieredResolutionConfigLocal(unknown_field="value")  # type: ignore[call-arg]

    def test_all_defaults(self) -> None:
        """Model with no args uses all defaults."""
        config = ModelTieredResolutionConfigLocal()
        assert config.min_tier is None
        assert config.max_tier is None
        assert config.require_proofs == ()
        assert config.classification is None

    def test_serialization_round_trip(self) -> None:
        """Model serializes and deserializes correctly."""
        config = ModelTieredResolutionConfigLocal(
            min_tier="local_exact",
            max_tier="org_trusted",
            require_proofs=("node_identity",),
            classification="internal",
        )
        data = config.model_dump()
        restored = ModelTieredResolutionConfigLocal(**data)
        assert restored == config


class TestModelTrustDomainConfigLocal:
    """Tests for ModelTrustDomainConfigLocal model."""

    def test_frozen(self) -> None:
        """Model is frozen (immutable)."""
        domain = ModelTrustDomainConfigLocal(
            domain_id="local.default",
            tier="local_exact",
        )
        with pytest.raises(Exception):
            domain.tier = "org_trusted"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Model rejects extra fields."""
        with pytest.raises(Exception):
            ModelTrustDomainConfigLocal(  # type: ignore[call-arg]
                domain_id="local.default",
                tier="local_exact",
                unknown_field="value",
            )

    def test_required_fields(self) -> None:
        """Model requires domain_id and tier."""
        with pytest.raises(Exception):
            ModelTrustDomainConfigLocal()  # type: ignore[call-arg]

    def test_optional_trust_root_ref(self) -> None:
        """trust_root_ref defaults to None."""
        domain = ModelTrustDomainConfigLocal(
            domain_id="local.default",
            tier="local_exact",
        )
        assert domain.trust_root_ref is None

    def test_serialization_round_trip(self) -> None:
        """Model serializes and deserializes correctly."""
        domain = ModelTrustDomainConfigLocal(
            domain_id="org.omninode",
            tier="org_trusted",
            trust_root_ref="secrets://keys/org-omninode-trust-root",
        )
        data = domain.model_dump()
        restored = ModelTrustDomainConfigLocal(**data)
        assert restored == domain


# =============================================================================
# Tests: All Valid Tier Combinations
# =============================================================================


class TestAllValidTiers:
    """Verify all tier values from EnumResolutionTier are accepted."""

    @pytest.mark.parametrize(
        "tier",
        [
            "local_exact",
            "local_compatible",
            "org_trusted",
            "federated_trusted",
            "quarantine",
        ],
    )
    def test_valid_min_tier(self, tmp_path: Path, tier: str) -> None:
        """Each valid tier is accepted as min_tier."""
        yaml_content = f"""\
name: "test"
dependencies:
  - alias: "dep"
    tiered_resolution:
      min_tier: "{tier}"
"""
        path = _write_contract(tmp_path, yaml_content)
        configs = load_tiered_resolution_configs(path)
        assert configs["dep"].min_tier == tier

    @pytest.mark.parametrize(
        "tier",
        [
            "local_exact",
            "local_compatible",
            "org_trusted",
            "federated_trusted",
            "quarantine",
        ],
    )
    def test_valid_trust_domain_tier(self, tmp_path: Path, tier: str) -> None:
        """Each valid tier is accepted in trust domain declarations."""
        yaml_content = f"""\
name: "test"
trust_domains:
  - domain_id: "test.domain"
    tier: "{tier}"
"""
        path = _write_contract(tmp_path, yaml_content)
        domains = load_trust_domain_configs(path)
        assert domains[0].tier == tier
