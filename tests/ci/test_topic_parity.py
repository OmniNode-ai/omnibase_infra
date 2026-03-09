"""CI test: every topic string defined in platform_topic_suffixes is provisioned.

Prevents future unregistered topics from missing Redpanda provisioning.
See OMN-4306.
"""

import pytest

from omnibase_infra.topics import ALL_PROVISIONED_SUFFIXES
from omnibase_infra.topics import platform_topic_suffixes as pts


def _all_defined_suffix_strings() -> set[str]:
    return {
        v for k, v in vars(pts).items() if isinstance(v, str) and v.startswith("onex.")
    }


KNOWN_EXCLUSIONS: set[str] = set()


@pytest.mark.unit
def test_all_topic_strings_are_provisioned() -> None:
    defined = _all_defined_suffix_strings()
    unprovisioned = defined - set(ALL_PROVISIONED_SUFFIXES) - KNOWN_EXCLUSIONS
    assert not unprovisioned, (
        "Topic strings defined in platform_topic_suffixes.py but not in ALL_PROVISIONED_SUFFIXES:\n"
        + "\n".join(f"  {t}" for t in sorted(unprovisioned))
    )


@pytest.mark.unit
def test_provisioned_topics_non_empty() -> None:
    assert len(ALL_PROVISIONED_SUFFIXES) > 10
