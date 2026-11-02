# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract-declared publish topic that holds one ``{principal}`` slot (OMN-22103)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

SLOT = "{principal}"
#: A principal as the lab issuer names one: lower-case, digits and hyphens.
PRINCIPAL_PATTERN = "[a-z0-9][a-z0-9-]{0,62}"


class ModelPrincipalTopic(BaseModel):
    """A declared topic with exactly one ``{principal}`` slot, and the names it admits.

    A per-principal reply topic cannot be a static contract entry: the requester's
    principal is in its name. The contract declares the template, and an emitted topic
    is admitted only when it is that template with a well-formed principal in the slot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    template: str

    def matches(self, candidate: str) -> bool:
        """True when ``candidate`` is this template with a principal filling the slot."""
        if self.template.count(SLOT) != 1:
            return False
        pattern = PRINCIPAL_PATTERN.join(
            re.escape(part) for part in self.template.split(SLOT)
        )
        return re.fullmatch(pattern, candidate) is not None

    @staticmethod
    def admits(candidate: str, declared: Iterable[str]) -> bool:
        """True when ``candidate`` is a declared literal topic or a declared template filled.

        An unfilled template is no topic: the slot is never published to as written.
        """
        topics = tuple(declared)
        if SLOT not in candidate and candidate in topics:
            return True
        return any(ModelPrincipalTopic(template=t).matches(candidate) for t in topics)

    @staticmethod
    def declared_in(contract_path: Path) -> tuple[str, ...]:
        """The templates a contract declares in parts under ``published_events``.

        A ``{principal}`` slot inside a full topic string fails the topic-name lint, so a
        contract states such a topic as ``topic_parts`` (kind, producer, name_prefix,
        version, principal_pattern), which render to
        ``onex.<kind>.<producer>.<name_prefix>{principal}.v<version>``. A row whose
        principal pattern is not the one this runtime applies is refused, not skipped:
        a skipped row would send every answer to the contract's terminal topic.
        """
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        rows = (contract or {}).get("published_events") or []
        templates: list[str] = []
        for row in rows:
            parts = row.get("topic_parts") if isinstance(row, dict) else None
            if not isinstance(parts, dict):
                continue
            pattern = str(parts.get("principal_pattern", "")).removeprefix("^")
            if pattern.removesuffix("$") != PRINCIPAL_PATTERN:
                msg = f"{contract_path}: {row.get('event_type')} declares a principal pattern other than {PRINCIPAL_PATTERN}"
                raise ValueError(msg)
            templates.append(
                f"onex.{parts['kind']}.{parts['producer']}.{parts['name_prefix']}"
                f"{SLOT}.v{parts['version']}"
            )
        return tuple(templates)
