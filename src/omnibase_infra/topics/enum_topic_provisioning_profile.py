# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic-provisioning environment profile (OMN-15395).

The profile classifies the *broker* a provisioning pass is pointed at, not the
deployment lane name. It is derived from the live Kafka client configuration
(``ModelKafkaEventBusConfig``) rather than a caller-supplied label, so a caller
cannot claim ``SELF_HOSTED`` against a managed cluster to dodge the durability
floor.
"""

from __future__ import annotations

from enum import Enum


class EnumTopicProvisioningProfile(str, Enum):
    """Durability class of the broker being provisioned.

    Attributes:
        SELF_HOSTED: A self-hosted broker we own end to end (local Redpanda,
            the ``.201`` lanes, CI sandboxes). Single-broker deployments are
            normal here, so replication factor 1 is legitimate and an
            explicitly declared environment default is allowed.
        MANAGED: A managed cluster (AWS MSK — detected via ``AWS_MSK_IAM`` SASL
            auth). RF1 means a single broker loss is unrecoverable data loss
            and blocks broker updates, so RF1 is rejected fail-closed and an
            undeclared replication factor is refused rather than defaulted.
    """

    SELF_HOSTED = "self_hosted"
    MANAGED = "managed"


__all__: list[str] = ["EnumTopicProvisioningProfile"]
