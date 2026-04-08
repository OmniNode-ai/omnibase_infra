# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CI test: every topic string defined in platform_topic_suffixes is provisioned.

Prevents future unregistered topics from missing Redpanda provisioning.
See OMN-4306.

Note: ALL_PROVISIONED_SUFFIXES conditionally excludes omnimemory topics when
OMNIMEMORY_ENABLED is unset. This test compares against the full union of all
topic spec groups (including optional ones) to catch any suffix constant that
has no spec entry, regardless of runtime flags.
"""

import pytest

from omnibase_infra.topics import platform_topic_suffixes as pts


def _all_defined_suffix_strings() -> set[str]:
    return {
        v for k, v in vars(pts).items() if isinstance(v, str) and v.startswith("onex.")
    }


def _all_spec_suffixes() -> set[str]:
    """Full set of suffixes across all spec groups, including optional ones.

    Uses the unconditional group constants (ALL_PLATFORM_TOPIC_SPECS,
    ALL_INTELLIGENCE_TOPIC_SPECS, ALL_OMNIMEMORY_TOPIC_SPECS, etc.) rather
    than ALL_PROVISIONED_TOPIC_SPECS, which conditionally excludes omnimemory
    topics based on the OMNIMEMORY_ENABLED environment variable.
    """
    all_specs = (
        pts.ALL_PLATFORM_TOPIC_SPECS
        + pts.ALL_INTELLIGENCE_TOPIC_SPECS
        + pts.ALL_OMNIMEMORY_TOPIC_SPECS
        + pts.ALL_OMNIBASE_INFRA_TOPIC_SPECS
        + pts.ALL_VALIDATION_TOPIC_SPECS
        + pts.ALL_OMNINODE_ROUTING_TOPIC_SPECS
        + pts.ALL_OMNICLAUDE_TOPIC_SPECS
    )
    return {spec.suffix for spec in all_specs}


KNOWN_EXCLUSIONS: set[str] = set()


@pytest.mark.unit
def test_all_topic_strings_are_provisioned() -> None:
    defined = _all_defined_suffix_strings()
    all_spec_suffixes = _all_spec_suffixes()
    unprovisioned = defined - all_spec_suffixes - KNOWN_EXCLUSIONS
    assert not unprovisioned, (
        "Topic strings defined in platform_topic_suffixes.py but not in any topic spec group:\n"
        + "\n".join(f"  {t}" for t in sorted(unprovisioned))
    )


@pytest.mark.unit
def test_provisioned_topics_non_empty() -> None:
    assert len(_all_spec_suffixes()) > 10
