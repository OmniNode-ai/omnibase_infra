# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""No production code may construct a bare ``TopicResolver()`` (OMN-18891).

A namespace applied at some seams and not others is worse than no namespace:
the runtime reports isolation it does not have, and the one construction that
was missed is the one that talks to the dev lane. Nine bare constructions were
measured before this ticket and each of them silently opted out of the
deployment namespace, so this is a ratchet rather than a style rule — detection
that is not wired as a gate gets ignored.

The allowed forms are ``create_topic_resolver(...)`` from
``omnibase_infra.topics.topic_namespace``, and a direct construction inside the
resolver module and its factory, which are the two places that define the
behaviour rather than consume it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

SRC = Path(__file__).resolve().parents[2] / "src" / "omnibase_infra"

#: The two modules that legitimately name the class directly: the definition
#: and the factory that wraps it.
_DEFINING_MODULES = frozenset(
    {
        "topics/topic_resolver.py",
        "topics/topic_namespace.py",
    }
)

_BARE_CONSTRUCTION = re.compile(r"(?<![\w.])TopicResolver\(\s*\)")


def test_no_bare_topic_resolver_construction_in_src() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in _DEFINING_MODULES:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.lstrip()
            # Docstring examples and comments describe the class; they do not
            # construct one at runtime.
            if stripped.startswith((">>>", "#", "*")):
                continue
            if _BARE_CONSTRUCTION.search(line):
                offenders.append(f"{rel}:{lineno}: {stripped}")
    assert not offenders, (
        "Bare TopicResolver() constructions opt out of the deployment topic "
        "namespace and will talk to the shared lane's topics. Use "
        "omnibase_infra.topics.create_topic_resolver() instead:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_can_actually_see_a_violation(tmp_path: Path) -> None:
    """Positive control: an empty result must mean absence, not a broken sweep.

    A regex guard that silently stops matching reports a clean tree forever.
    """
    probe = tmp_path / "offender.py"
    probe.write_text("resolver = TopicResolver()\n", encoding="utf-8")
    assert _BARE_CONSTRUCTION.search(probe.read_text(encoding="utf-8")) is not None
    assert (
        _BARE_CONSTRUCTION.search("resolver = TopicResolver(bus_descriptors=x)") is None
    )
    assert _BARE_CONSTRUCTION.search("resolver = create_topic_resolver()") is None
