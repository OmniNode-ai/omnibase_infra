# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kinds of Kafka failure that retrying cannot fix (OMN-19092).

Related Tickets:
    - OMN-19092: One shared answer to "is this Kafka failure permanent".
"""

from __future__ import annotations

from enum import Enum


class EnumKafkaPermanentFailure(str, Enum):
    """Which question a permanent Kafka failure answers.

    The kinds are separate because the questions are different ones, and a
    caller acts on the kinds it owns a decision for and no others. Connect
    raises at once on :attr:`CONNECT_AUTHENTICATION_REJECTED`; publish fails the
    attempt and reports a typed error on :attr:`TOPIC_AUTHORIZATION_REFUSED`;
    the gateway quarantines the record on either authorization kind but keeps
    retrying an authentication rejection, because a credential refresh clears
    that one. Merging the kinds would change one of those decisions.

    Values:
        CONNECT_AUTHENTICATION_REJECTED: The broker refused this client at
            connect time (SASL or version negotiation). Never surfaced as a
            topic authorization error, so the topic kind cannot stand in.
        TOPIC_AUTHORIZATION_REFUSED: The broker refused a send on one topic's
            ACLs. A publish-time verdict about that topic only.
        CLUSTER_AUTHORIZATION_REFUSED: The broker refused the principal a
            cluster-level operation.
    """

    CONNECT_AUTHENTICATION_REJECTED = "connect_authentication_rejected"
    TOPIC_AUTHORIZATION_REFUSED = "topic_authorization_refused"
    CLUSTER_AUTHORIZATION_REFUSED = "cluster_authorization_refused"
