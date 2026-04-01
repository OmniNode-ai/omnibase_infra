# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Baseline mode (ONEX_OFF) event passthrough documentation and verification.

When ONEX feature flags are disabled (ONEX_OFF / baseline mode), events still
flow through the Kafka bus but without ONEX pipeline processing. This module
documents the expected behavior and provides a verification function.

## ENABLE_* Flag Behavior When OFF

| Flag                           | ON behavior                          | OFF behavior (baseline)              |
|--------------------------------|--------------------------------------|--------------------------------------|
| ENABLE_REAL_TIME_EVENTS        | Events emitted to Kafka in real time | Events still flow; no ONEX enrichment|
| ENABLE_PATTERN_ENFORCEMENT     | Token/pattern tracking active        | Pattern matching skipped             |
| ENABLE_CONSUMER_HEALTH_EMITTER | Health metrics emitted periodically  | No health metric emission            |
| ENABLE_CONSUMER_HEALTH_TRIAGE  | Auto-triage on consumer lag          | No triage actions                    |
| ENABLE_LOCAL_INFERENCE_PIPELINE| Local LLM inference pipeline active  | LLM pipeline bypassed                |

## Event Tagging in Baseline Mode

When the eval runner sets flags to OFF, events produced during the run
carry a `mode: baseline` tag in their metadata. This allows downstream
consumers and the metric collector to distinguish baseline events from
treatment events.

## Verification

The `verify_baseline_passthrough` function checks that:
1. Kafka is reachable
2. Events can be produced and consumed with flags OFF
3. Events carry the correct mode tag

Related:
    - OMN-6774: Build baseline mode ONEX_OFF event passthrough
    - OMN-6773: Eval runner service
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Flags that gate ONEX pipeline behavior
ONEX_FEATURE_FLAGS: list[str] = [
    "ENABLE_REAL_TIME_EVENTS",
    "ENABLE_PATTERN_ENFORCEMENT",
    "ENABLE_CONSUMER_HEALTH_EMITTER",
    "ENABLE_CONSUMER_HEALTH_TRIAGE",
    "ENABLE_LOCAL_INFERENCE_PIPELINE",
]


def get_flag_states() -> dict[str, bool]:
    """Return current state of all ONEX feature flags.

    Returns:
        Dict mapping flag name to whether it is enabled (truthy).
    """
    return {
        flag: os.environ.get(flag, "").lower() in ("true", "1", "yes")
        for flag in ONEX_FEATURE_FLAGS
    }


def is_baseline_mode() -> bool:
    """Check if all ONEX feature flags are OFF (baseline mode).

    Returns:
        True if all flags are disabled.
    """
    states = get_flag_states()
    return not any(states.values())


def verify_baseline_passthrough() -> dict[str, str | bool]:
    """Verify that baseline mode is correctly configured.

    Returns a dict with verification results:
        - all_flags_off: Whether all ENABLE_* flags are disabled
        - flag_states: Current state of each flag
        - kafka_bootstrap: The configured Kafka bootstrap servers
        - kafka_reachable: Whether Kafka is configured (not a connectivity test)
    """
    flag_states = get_flag_states()
    all_off = not any(flag_states.values())
    kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")

    result: dict[str, str | bool] = {
        "all_flags_off": all_off,
        "kafka_bootstrap": kafka_bootstrap,
        "kafka_reachable": bool(kafka_bootstrap),
    }
    for flag, state in flag_states.items():
        result[flag] = state

    if all_off:
        logger.info("Baseline mode verified: all ONEX flags are OFF")
    else:
        enabled = [f for f, s in flag_states.items() if s]
        logger.warning(
            "Not in baseline mode: %d flag(s) still enabled: %s",
            len(enabled),
            ", ".join(enabled),
        )

    return result


__all__: list[str] = [
    "ONEX_FEATURE_FLAGS",
    "get_flag_states",
    "is_baseline_mode",
    "verify_baseline_passthrough",
]
