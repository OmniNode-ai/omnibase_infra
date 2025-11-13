"""
Trust zones management for O.N.E. v0.1 protocol.

This module provides trust zone assignment and trust level
requirements for different operations.
"""

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Trust levels for O.N.E. protocol operations."""

    UNVERIFIED = "UNVERIFIED"
    SIGNED = "SIGNED"
    VERIFIED = "VERIFIED"


class TrustZone(str, Enum):
    """Trust zones for network segmentation."""

    LOCAL = "zone.local"
    ORG = "zone.org"
    GLOBAL = "zone.global"


class TrustContext(BaseModel):
    """Context information for trust validation."""

    trust_level: TrustLevel
    trust_zone: TrustZone
    signature: Optional[str] = None
    public_key: Optional[str] = None
    verification_timestamp: Optional[str] = None
    metadata: dict[str, Any] = {}


class TrustZoneManager:
    """
    Manager for trust zone assignment and validation.

    Handles trust zone determination based on source addresses
    and required trust levels for different operations.
    """

    def __init__(self):
        """Initialize trust zone manager."""
        self.zone_assignments = {
            "localhost": TrustZone.LOCAL,
            "127.0.0.1": TrustZone.LOCAL,
            "::1": TrustZone.LOCAL,
            "*.omninode.local": TrustZone.LOCAL,
            "*.omninode.org": TrustZone.ORG,
            "*.internal": TrustZone.ORG,
        }

        # Trust requirements matrix
        self.trust_requirements = {
            (TrustZone.LOCAL, "read"): TrustLevel.UNVERIFIED,
            (TrustZone.LOCAL, "write"): TrustLevel.UNVERIFIED,
            (TrustZone.LOCAL, "delete"): TrustLevel.UNVERIFIED,
            (TrustZone.ORG, "read"): TrustLevel.SIGNED,
            (TrustZone.ORG, "write"): TrustLevel.SIGNED,
            (TrustZone.ORG, "delete"): TrustLevel.VERIFIED,
            (TrustZone.GLOBAL, "read"): TrustLevel.VERIFIED,
            (TrustZone.GLOBAL, "write"): TrustLevel.VERIFIED,
            (TrustZone.GLOBAL, "delete"): TrustLevel.VERIFIED,
        }

    def assign_trust_zone(self, source_address: str) -> TrustZone:
        """
        Assign trust zone based on source address.

        Args:
            source_address: Source IP address or hostname

        Returns:
            TrustZone: Assigned trust zone
        """
        if not source_address:
            logger.warning("No source address provided, defaulting to GLOBAL zone")
            return TrustZone.GLOBAL

        # Direct match
        if source_address in self.zone_assignments:
            zone = self.zone_assignments[source_address]
            logger.debug(f"Direct match: {source_address} -> {zone}")
            return zone

        # Pattern matching
        for pattern, zone in self.zone_assignments.items():
            if self._matches_pattern(source_address, pattern):
                logger.debug(
                    f"Pattern match: {source_address} matches {pattern} -> {zone}"
                )
                return zone

        # Default to global zone for unknown addresses
        logger.debug(f"No match for {source_address}, defaulting to GLOBAL zone")
        return TrustZone.GLOBAL

    def _matches_pattern(self, address: str, pattern: str) -> bool:
        """
        Check if address matches a pattern.

        Args:
            address: Address to check
            pattern: Pattern to match against

        Returns:
            bool: True if matches
        """
        if pattern.startswith("*"):
            suffix = pattern[1:]
            return address.endswith(suffix)
        return address == pattern

    def get_required_trust_level(self, zone: TrustZone, operation: str) -> TrustLevel:
        """
        Get required trust level for operation in zone.

        Args:
            zone: Trust zone
            operation: Operation type (read, write, delete)

        Returns:
            TrustLevel: Required trust level
        """
        # Normalize operation
        if operation.upper() in ["GET", "HEAD", "OPTIONS"]:
            operation = "read"
        elif operation.upper() in ["POST", "PUT", "PATCH"]:
            operation = "write"
        elif operation.upper() == "DELETE":
            operation = "delete"
        else:
            operation = operation.lower()

        # Get requirement from matrix
        requirement = self.trust_requirements.get(
            (zone, operation), TrustLevel.VERIFIED  # Default to highest requirement
        )

        logger.debug(f"Trust requirement: {zone}:{operation} -> {requirement}")
        return requirement

    def validate_trust_level(
        self, required_level: TrustLevel, actual_level: TrustLevel
    ) -> bool:
        """
        Validate if actual trust level meets requirement.

        Args:
            required_level: Required trust level
            actual_level: Actual trust level

        Returns:
            bool: True if requirement met
        """
        # Trust level hierarchy
        level_hierarchy = {
            TrustLevel.UNVERIFIED: 0,
            TrustLevel.SIGNED: 1,
            TrustLevel.VERIFIED: 2,
        }

        required_value = level_hierarchy.get(required_level, 2)
        actual_value = level_hierarchy.get(actual_level, 0)

        return actual_value >= required_value

    def add_zone_assignment(self, pattern: str, zone: TrustZone):
        """
        Add custom zone assignment pattern.

        Args:
            pattern: Address pattern
            zone: Trust zone to assign
        """
        self.zone_assignments[pattern] = zone
        logger.info(f"Added zone assignment: {pattern} -> {zone}")

    def update_trust_requirement(
        self, zone: TrustZone, operation: str, required_level: TrustLevel
    ):
        """
        Update trust requirement for specific zone/operation.

        Args:
            zone: Trust zone
            operation: Operation type
            required_level: Required trust level
        """
        self.trust_requirements[(zone, operation)] = required_level
        logger.info(
            f"Updated trust requirement: {zone}:{operation} -> {required_level}"
        )
