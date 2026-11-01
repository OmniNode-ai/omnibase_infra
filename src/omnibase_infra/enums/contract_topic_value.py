# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve generated enum values from the packaged contract that declares them."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_infra.tools.contract_topic_extractor import (
    _extract_raw_topics_from_contract,
)

if TYPE_CHECKING:
    from omnibase_infra.tools.contract_topic_extractor import ModelContractTopicEntry


def render_contract_topic_member(
    key: str, entry: ModelContractTopicEntry, template: str
) -> str:
    """Render a member with a relocatable source binding and no fallback value."""
    source = entry.source_contracts[0]
    # Hosted checkouts can have package-named ancestors. Use the nearest
    # package directory so checkout names cannot enter generated bindings.
    package_root = next(
        (parent for parent in source.parents if parent.name == "omnibase_infra"),
        None,
    )
    if package_root is not None:
        source = source.relative_to(package_root)
    args = ", ".join(repr(value) for value in (str(source), entry.producer, key))
    return template.format(key=key, args=args, topic=entry.topic)


@cache
def _contract_topics(source: str) -> frozenset[str]:
    """Read a package-relative contract, or an explicitly supplied source file."""
    resource = (
        Path(source)
        if Path(source).is_absolute()
        else files("omnibase_infra").joinpath(source)
    )
    document = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{source}: expected a contract mapping")
    if Path(source).name == "topics.yaml":
        topics = document.get("topics")
        if not isinstance(topics, list) or any(
            type(topic) is not str for topic in topics
        ):
            raise ValueError(f"{source}: expected a string topic list")
        return frozenset(topic.strip() for topic in topics)
    return frozenset(
        declaration.topic
        for declaration in _extract_raw_topics_from_contract(document, Path(source))
    )


def contract_topic_value(source: str, producer: str, member: str) -> str:
    """Bind an enum member to exactly one topic declared by its source contract.

    The generator supplies the source and symbolic member, never a fallback topic.
    Missing or ambiguous declarations fail at import instead of retaining stale values.
    """
    matches = []
    for topic in _contract_topics(source):
        parts = topic.split(".")
        if len(parts) < 5 or parts[2] != producer:
            continue
        key = "_".join(
            (parts[1], "_".join(parts[3:-1]).replace("-", "_"), parts[-1])
        ).upper()
        if key == member:
            matches.append(topic)
    if len(matches) != 1:
        raise ValueError(
            f"{source}: {producer}.{member} must bind one contract topic, found {matches}"
        )
    return matches[0]
