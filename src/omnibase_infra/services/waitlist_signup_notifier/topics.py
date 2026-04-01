# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic constants for the waitlist signup notifier consumer.

Resolved via the ServiceTopicRegistry to avoid hardcoded topic strings
(see CLAUDE.md Agent Behavioral Rule 5: Contract-First Topic Definitions).

Related Tickets:
    - OMN-7199: Slack notification on new waitlist signup
"""

from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

_registry = ServiceTopicRegistry.from_defaults()

# Emitted by omniweb Server Action after a user joins the beta waitlist.
# Payload: { email_domain: string } (no PII — domain only).
TOPIC_WAITLIST_SIGNUP: str = _registry.resolve(topic_keys.WAITLIST_SIGNUP)
