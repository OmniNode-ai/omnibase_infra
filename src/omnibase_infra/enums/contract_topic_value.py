# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve generated enum values from the packaged contract that declares them."""

from __future__ import annotations

import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_infra.tools.contract_topic_extractor import (
    _extract_raw_topics_from_contract,
    _parse_topic,
)

if TYPE_CHECKING:
    from omnibase_infra.tools.contract_topic_extractor import ModelContractTopicEntry


def render_contract_topic_member(
    key: str, entry: ModelContractTopicEntry, template: str
) -> str:
    """Render a member with a relocatable source binding and no fallback value."""
    source = entry.source_contracts[0]
    # Anchor to the source layout, not a directory name that can also occur
    # inside a package or in a hosted checkout path.
    package_root = next(
        (
            parent
            for parent in source.parents
            if parent.name == "omnibase_infra" and parent.parent.name == "src"
        ),
        None,
    )
    if package_root is not None:
        source = source.relative_to(package_root)
    args = ", ".join(repr(value) for value in (str(source), entry.producer, key))
    return template.format(key=key, args=args, topic=entry.topic)


def _contract_topics(source: str) -> frozenset[str]:
    """Read a package-relative contract, or an explicitly supplied source file.

    Package-relative sources go through the package loader so a wheel on
    ``sys.path`` (a zip archive) resolves the same way as a directory install.
    """
    try:
        if Path(source).is_absolute():
            if Path(source).suffix != ".yaml":
                raise ValueError(f"{source}: contract source must be a .yaml file")
            text = Path(source).read_text(encoding="utf-8")
        else:
            data = pkgutil.get_data("omnibase_infra", source)
            if data is None:
                raise FileNotFoundError(source)
            text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(
            f"{source}: cannot read contract ({type(error).__name__})"
        ) from error
    return _parse_contract_topics(source, text)


@lru_cache(maxsize=128)
def _parse_contract_topics(source: str, text: str) -> frozenset[str]:
    """Cache parsing by content so edits never leave stale file-path entries."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(
            f"{source}: cannot read contract ({type(error).__name__})"
        ) from error
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


def contract_topic_member_key(kind: str, event_name: str, version: str) -> str:
    """Use the same member spelling in the generator and runtime resolver."""
    prefix = {
        "evt": "EVT_",
        "cmd": "CMD_",
        "intent": "INTENT_",
        "dlq": "DLQ_",
        "snapshot": "SNAPSHOT_",
    }[kind]
    base_key = event_name.replace("-", "_").replace(".", "_").upper()
    return f"{prefix}{base_key}_V{version[1:]}"


def contract_topic_value(source: str, producer: str, member: str) -> str:
    """Bind an enum member to exactly one topic declared by its source contract.

    The generator supplies the source and symbolic member, never a fallback topic.
    Missing or ambiguous declarations fail at import instead of retaining stale values.
    """
    matches = []
    try:
        topics = _contract_topics(source)
    except ValueError as error:
        raise ValueError(f"{producer}.{member}: {error}") from error
    for topic in sorted(topics):
        entry = _parse_topic(topic, Path(source))
        if entry is None or entry.producer != producer:
            continue
        key = contract_topic_member_key(entry.kind, entry.event_name, entry.version)
        if key == member:
            matches.append(topic)
    if len(matches) != 1:
        raise ValueError(
            f"{source}: {producer}.{member} must bind one contract topic, found {matches}"
        )
    return matches[0]
