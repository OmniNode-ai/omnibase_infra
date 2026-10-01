# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28 wire evidence and collector results.

Recorded evidence can omit fields and carry extra exposure metadata. Values
which the grader checks for integer-ness stay JSON values: parsing must not
turn a boolean, float or numeric string into a passing integer counter.
"""

from __future__ import annotations

from pydantic import ConfigDict, JsonValue, with_config
from typing_extensions import TypedDict


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowRow(TypedDict, total=False):
    """One exposure row, retaining raw counters and unknown metadata."""

    consumer_group: JsonValue
    topic: JsonValue
    window_start: JsonValue
    messages_in: JsonValue
    messages_out: JsonValue
    flow_state: JsonValue


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowPage(TypedDict, total=False):
    """Cursor metadata copied without coercion from the exposure API."""

    row_count: JsonValue
    row_limit: JsonValue
    next_cursor: JsonValue
    backing: JsonValue
    data_freshness: JsonValue
    end_proof: JsonValue


class TypedDictConsumerFlowResponse(TypedDictConsumerFlowPage):
    """An exposure response has a rows list in addition to page metadata."""

    rows: list[TypedDictConsumerFlowRow]


class TypedDictConsumerFlowWalk(TypedDict):
    """Collector's complete bounded cursor walk."""

    pages: list[TypedDictConsumerFlowPage]
    rows: list[TypedDictConsumerFlowRow]
    terminated: bool
    second_page_differs: bool | None


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowKinds(TypedDict, total=False):
    """Full samples and the convenience view filtered by kind."""

    samples: list[list[TypedDictConsumerFlowRow]]
    view: list[dict[str, list[TypedDictConsumerFlowRow]]]


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowRun(TypedDict, total=False):
    """One pytest run, with optional branch-mutation evidence."""

    returncode: JsonValue
    outcomes: dict[str, str]
    factory: str
    applied: JsonValue
    restored: JsonValue


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowNegative(TypedDict, total=False):
    """Clean run and per-branch negative controls."""

    clean: TypedDictConsumerFlowRun
    mutations: dict[str, TypedDictConsumerFlowRun]
    module_sha256: str


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowCursor(TypedDict, total=False):
    """Cursor evidence compared with the live database window."""

    pages: list[TypedDictConsumerFlowPage]
    terminated: JsonValue
    second_page_differs: JsonValue
    rows: int
    walked_groups: list[str]
    walked_pairs: int
    live_groups: list[str]
    undeclared_cursor_control: dict[str, bool] | None


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowIdentity(TypedDict, total=False):
    """Container facts read before and after collection."""

    id: str
    started_at: JsonValue
    status: JsonValue
    health: JsonValue
    image: JsonValue


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowNatural(TypedDict, total=False):
    """Log counts separating probe errors from natural errors."""

    lines: JsonValue
    total: JsonValue
    probe: JsonValue
    natural: JsonValue
    first_line: str | None


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowInjection(TypedDict, total=False):
    """Publication evidence and observed malformed-trigger effects."""

    marker: str
    correlation_id: str
    published_at: str
    offset: JsonValue
    envelope_copied_from_offset: JsonValue
    validation_errors_after: JsonValue
    boundary_lines: JsonValue
    dlq_copies: JsonValue
    generic_dlq_before: int
    generic_dlq_after: int


@with_config(ConfigDict(extra="allow"))
class TypedDictConsumerFlowBoot(TypedDict, total=False):
    """Boot-scoped natural traffic and injected-error facts."""

    applied_hwm_before: JsonValue
    applied_hwm_after: JsonValue
    applied_window_seconds: JsonValue
    natural: dict[str, TypedDictConsumerFlowNatural]
    injection: TypedDictConsumerFlowInjection
    seam_dlq_hwm: list[int]


class TypedDictConsumerFlowCollected(TypedDict, total=False):
    """Sections assembled before the observation model is constructed."""

    boot_identity: dict[str, TypedDictConsumerFlowIdentity]
    kinds: TypedDictConsumerFlowKinds
    negative: TypedDictConsumerFlowNegative
    cursor: TypedDictConsumerFlowCursor
    boot: TypedDictConsumerFlowBoot
