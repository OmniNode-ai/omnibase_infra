# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Waitlist Signup Slack Notifier — Kafka consumer that posts to Slack on new signups."""

from omnibase_infra.services.waitlist_signup_notifier.consumer import (
    WaitlistSignupNotifier,
)

__all__ = ["WaitlistSignupNotifier"]
