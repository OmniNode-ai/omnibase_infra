# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for TopicResolver multi-bus trust-domain routing.

Tests the Phase 5 (OMN-2894) extensions to TopicResolver:
- Backward compatibility: without bus_descriptors, behavior is identical
- Trust-domain routing: namespace prefix prepended when trust_domain matches
- Error handling: BusDescriptorNotFoundError when no descriptor matches
- Edge cases: empty prefix, trust_domain without descriptors configured

Existing TopicResolver pass-through tests remain in test_topic_resolver.py.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.topics import (
    BusDescriptorNotFoundError,
    TopicResolutionError,
    TopicResolver,
)
from omnibase_infra.topics.model_bus_descriptor import ModelBusDescriptor

pytestmark = [pytest.mark.unit]

# Valid ONEX topic suffix used across tests.
VALID_SUFFIX = "onex.evt.platform.node-registration.v1"
VALID_CMD_SUFFIX = "onex.cmd.platform.request-introspection.v1"


def _local_descriptor(**overrides: object) -> ModelBusDescriptor:
    """Build a local bus descriptor with sensible defaults."""
    defaults: dict[str, object] = {
        "bus_id": "bus.local",
        "trust_domain": "local.default",
        "transport_type": EnumInfraTransportType.KAFKA,
        "namespace_prefix": "",
        "bootstrap_servers": ("localhost:9092",),
        "allowed_classifications": ("public", "internal"),
    }
    defaults.update(overrides)
    return ModelBusDescriptor(**defaults)  # type: ignore[arg-type]


def _org_descriptor(**overrides: object) -> ModelBusDescriptor:
    """Build an org-level bus descriptor with namespace prefix."""
    defaults: dict[str, object] = {
        "bus_id": "bus.org.omninode",
        "trust_domain": "org.omninode",
        "transport_type": EnumInfraTransportType.KAFKA,
        "namespace_prefix": "org.omninode.",
        "bootstrap_servers": ("kafka.org:9092",),
        "allowed_classifications": ("public",),
    }
    defaults.update(overrides)
    return ModelBusDescriptor(**defaults)  # type: ignore[arg-type]


class TestTopicResolverBackwardCompatibility:
    """Regression: TopicResolver without bus_descriptors is identical to original."""

    def test_no_args_constructor(self) -> None:
        """Default constructor (no bus_descriptors) works identically."""
        resolver = TopicResolver()
        assert resolver.resolve(VALID_SUFFIX) == VALID_SUFFIX

    def test_none_bus_descriptors(self) -> None:
        """Explicit None bus_descriptors works identically."""
        resolver = TopicResolver(bus_descriptors=None)
        assert resolver.resolve(VALID_SUFFIX) == VALID_SUFFIX

    def test_empty_bus_descriptors(self) -> None:
        """Empty list of bus_descriptors works identically."""
        resolver = TopicResolver(bus_descriptors=[])
        assert resolver.resolve(VALID_SUFFIX) == VALID_SUFFIX

    def test_trust_domain_ignored_without_descriptors(self) -> None:
        """trust_domain is silently ignored when no descriptors are configured."""
        resolver = TopicResolver()
        result = resolver.resolve(VALID_SUFFIX, trust_domain="org.omninode")
        assert result == VALID_SUFFIX

    def test_trust_domain_ignored_with_empty_descriptors(self) -> None:
        """trust_domain is silently ignored when descriptors list is empty."""
        resolver = TopicResolver(bus_descriptors=[])
        result = resolver.resolve(VALID_SUFFIX, trust_domain="org.omninode")
        assert result == VALID_SUFFIX

    def test_invalid_suffix_still_rejected(self) -> None:
        """Invalid suffixes are still rejected even with bus_descriptors."""
        resolver = TopicResolver(bus_descriptors=[_local_descriptor()])
        with pytest.raises(TopicResolutionError):
            resolver.resolve("bad-topic")

    def test_correlation_id_passthrough(self) -> None:
        """correlation_id is preserved in pass-through mode."""
        resolver = TopicResolver()
        cid = uuid4()
        result = resolver.resolve(VALID_SUFFIX, correlation_id=cid)
        assert result == VALID_SUFFIX

    def test_bus_descriptors_property_empty_by_default(self) -> None:
        """bus_descriptors property returns empty list when not configured."""
        resolver = TopicResolver()
        assert resolver.bus_descriptors == []


class TestTopicResolverMultiBus:
    """Multi-bus routing with trust_domain."""

    def test_local_descriptor_empty_prefix(self) -> None:
        """Local descriptor with empty prefix returns suffix unchanged."""
        resolver = TopicResolver(bus_descriptors=[_local_descriptor()])
        result = resolver.resolve(VALID_SUFFIX, trust_domain="local.default")
        assert result == VALID_SUFFIX

    def test_org_descriptor_prepends_prefix(self) -> None:
        """Org descriptor prepends namespace_prefix to suffix."""
        resolver = TopicResolver(
            bus_descriptors=[_local_descriptor(), _org_descriptor()]
        )
        result = resolver.resolve(VALID_SUFFIX, trust_domain="org.omninode")
        assert result == f"org.omninode.{VALID_SUFFIX}"

    def test_org_descriptor_cmd_topic(self) -> None:
        """Org descriptor works with command topics too."""
        resolver = TopicResolver(bus_descriptors=[_org_descriptor()])
        result = resolver.resolve(VALID_CMD_SUFFIX, trust_domain="org.omninode")
        assert result == f"org.omninode.{VALID_CMD_SUFFIX}"

    def test_multiple_descriptors_correct_lookup(self) -> None:
        """Correct descriptor is selected from multiple options."""
        fed_descriptor = ModelBusDescriptor(
            bus_id="bus.fed.partner",
            trust_domain="fed.partner-a",
            transport_type=EnumInfraTransportType.BRIDGE,
            namespace_prefix="fed.partner-a.",
            bootstrap_servers=(),
            allowed_classifications=("public",),
        )
        resolver = TopicResolver(
            bus_descriptors=[
                _local_descriptor(),
                _org_descriptor(),
                fed_descriptor,
            ]
        )
        result = resolver.resolve(VALID_SUFFIX, trust_domain="fed.partner-a")
        assert result == f"fed.partner-a.{VALID_SUFFIX}"

    def test_no_trust_domain_with_descriptors_is_passthrough(self) -> None:
        """Without trust_domain, resolver is pass-through even with descriptors."""
        resolver = TopicResolver(
            bus_descriptors=[_local_descriptor(), _org_descriptor()]
        )
        result = resolver.resolve(VALID_SUFFIX)
        assert result == VALID_SUFFIX

    def test_none_trust_domain_with_descriptors_is_passthrough(self) -> None:
        """Explicit trust_domain=None is pass-through even with descriptors."""
        resolver = TopicResolver(
            bus_descriptors=[_local_descriptor(), _org_descriptor()]
        )
        result = resolver.resolve(VALID_SUFFIX, trust_domain=None)
        assert result == VALID_SUFFIX

    def test_bus_descriptors_property_returns_configured(self) -> None:
        """bus_descriptors property returns all configured descriptors."""
        descriptors = [_local_descriptor(), _org_descriptor()]
        resolver = TopicResolver(bus_descriptors=descriptors)
        assert len(resolver.bus_descriptors) == 2
        domains = {d.trust_domain for d in resolver.bus_descriptors}
        assert domains == {"local.default", "org.omninode"}


class TestTopicResolverMultiBusErrors:
    """Error handling for multi-bus routing."""

    def test_unknown_trust_domain_raises(self) -> None:
        """Requesting an unconfigured trust domain raises BusDescriptorNotFoundError."""
        resolver = TopicResolver(bus_descriptors=[_local_descriptor()])
        with pytest.raises(BusDescriptorNotFoundError, match=r"unknown\.domain"):
            resolver.resolve(VALID_SUFFIX, trust_domain="unknown.domain")

    def test_error_includes_trust_domain(self) -> None:
        """BusDescriptorNotFoundError exposes the requested trust_domain."""
        resolver = TopicResolver(bus_descriptors=[_local_descriptor()])
        with pytest.raises(BusDescriptorNotFoundError) as exc_info:
            resolver.resolve(VALID_SUFFIX, trust_domain="missing.domain")
        assert exc_info.value.trust_domain == "missing.domain"

    def test_error_includes_correlation_id(self) -> None:
        """BusDescriptorNotFoundError includes correlation_id when provided."""
        resolver = TopicResolver(bus_descriptors=[_local_descriptor()])
        cid = uuid4()
        with pytest.raises(BusDescriptorNotFoundError, match=str(cid)):
            resolver.resolve(
                VALID_SUFFIX,
                trust_domain="missing.domain",
                correlation_id=cid,
            )

    def test_error_has_infra_context(self) -> None:
        """BusDescriptorNotFoundError carries structured infra_context."""
        resolver = TopicResolver(bus_descriptors=[_local_descriptor()])
        with pytest.raises(BusDescriptorNotFoundError) as exc_info:
            resolver.resolve(VALID_SUFFIX, trust_domain="missing.domain")
        err = exc_info.value
        assert err.infra_context is not None
        assert err.infra_context.transport_type == EnumInfraTransportType.KAFKA
        assert err.infra_context.operation == "resolve_topic"

    def test_error_is_topic_resolution_error(self) -> None:
        """BusDescriptorNotFoundError is a subclass of TopicResolutionError."""
        assert issubclass(BusDescriptorNotFoundError, TopicResolutionError)

    def test_invalid_suffix_rejected_before_bus_lookup(self) -> None:
        """Suffix validation happens before bus descriptor lookup."""
        resolver = TopicResolver(bus_descriptors=[_local_descriptor()])
        # bad-topic should raise TopicResolutionError, not BusDescriptorNotFoundError
        with pytest.raises(TopicResolutionError) as exc_info:
            resolver.resolve("bad-topic", trust_domain="local.default")
        assert not isinstance(exc_info.value, BusDescriptorNotFoundError)

    def test_last_descriptor_wins_for_duplicate_domains(self) -> None:
        """When multiple descriptors share a trust_domain, the last one wins."""
        desc1 = _local_descriptor(namespace_prefix="first.")
        desc2 = _local_descriptor(namespace_prefix="second.")
        resolver = TopicResolver(bus_descriptors=[desc1, desc2])
        result = resolver.resolve(VALID_SUFFIX, trust_domain="local.default")
        assert result == f"second.{VALID_SUFFIX}"
