# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic constants for the waitlist signup notifier consumer.

Related Tickets:
    - OMN-7199: Slack notification on new waitlist signup
"""

# Emitted by omniweb Server Action after a user joins the beta waitlist.
# Payload: { email_domain: string } (no PII — domain only).
TOPIC_WAITLIST_SIGNUP: str = "onex.evt.omniweb.waitlist-signup.v1"
