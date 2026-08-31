# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fence-aware ``Key: value`` trailer parsing for PR bodies (OMN-17294).

WHY THIS EXISTS
---------------
Two CI gates in this repo read a trailer out of a pull-request body and treat
the value as authority:

* ``scripts/resolve_node_migration_source_ref.py`` -- ``Omnimarket-Source-Ref:``
  chooses which omnimarket tree the required *Application Database Domain
  Enforcement (OMN-15361)* job derives its TABLE grants from.
* ``scripts/ci/check_occ_companion_merged.py`` -- ``Evidence-Source:`` names
  the onex_change_control companion whose merge state the STRICT
  ``CI Summary`` gate proves durable.

Both previously walked the body as flat lines and honoured the FIRST match
anywhere in it. A PR body does not have to *declare* a trailer to contain one:
runbook excerpts, pasted CI logs, a diff of another PR's body, and
"the trailer looks like this" examples all put the literal text in the body,
normally inside a fenced code block. Under flat-line matching that quoted text
IS the trailer, and it outranks the author's real one because it appears
first.

This is the OMN-15345 matcher class (table names matched inside SQL comments)
applied to PR bodies: structured text read as flat lines. The failure is
silent and in the dangerous direction -- neither gate errors, both derive
confidently from the wrong source.

WHAT COUNTS AS A TRAILER HERE
-----------------------------
1. **Column 0 only.** Git trailer semantics put the block at the left margin,
   and four-space indentation is a markdown code block. An indented line is
   never a trailer. (``check_occ_companion_merged`` already required this;
   the ref resolver did not, and its ``.strip()`` honoured indented code.)
2. **Outside fenced code blocks.** ``` ``` ``` and ``~~~`` fences are tracked
   per CommonMark: a fence opens at indent <= 3, and closes on a fence of the
   same character, at least as long, carrying no info string. An unterminated
   fence runs to the end of the body -- so it *suppresses* trailers rather
   than exposing them, which is the safe direction.
3. **Outside inline code spans.** Backtick spans are masked before matching,
   so ``` `Omnimarket-Source-Ref: x` ``` is quoted text, not a declaration.
4. **Unambiguous.** Two different values for the trailer (whether repeated
   under one field name or split across accepted aliases) raise
   :class:`TrailerConflictError` rather than silently resolving to whichever
   came first. Repeating the SAME value is legal: evidence stamps get
   re-applied by reruns, and ``scripts/pr_body.py --append`` is idempotent
   precisely because duplicate stamps are expected.

This module is deliberately dependency-free and importable both as
``scripts.ci.pr_trailers`` (tests, ``python -m``) and from a script executed
directly by path, which is how CI invokes both consumers.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Final

#: A code fence: three or more backticks or tildes, plus an optional info
#: string. Matched against the line with leading indentation already removed.
_FENCE_RE: Final = re.compile(r"^(?P<marker>`{3,}|~{3,})(?P<info>.*)$")

#: An inline code span: a run of N backticks, the shortest span closing on a
#: run of the same length. Simplified from CommonMark (no support for spans
#: broken across lines), which is sufficient for single-line trailer matching.
_INLINE_CODE_RE: Final = re.compile(r"(`+)(?:.*?)\1")

#: Markdown treats four or more leading spaces as an indented code block.
_MAX_MARKDOWN_INDENT: Final = 3


class TrailerConflictError(ValueError):
    """Two different values declared for the same trailer in one body.

    Subclasses :class:`ValueError` so callers that already treat a malformed
    trailer as a hard error keep their existing except-clause.
    """


def iter_prose_lines(body: str) -> Iterator[str]:
    """Yield the body's lines that are NOT inside a fenced code block.

    Fence markers themselves are never yielded. A fence that is opened and
    never closed swallows the remainder of the body, matching CommonMark.
    """
    fence_marker: str | None = None
    for raw_line in body.splitlines():
        without_indent = raw_line.lstrip(" ")
        indent = len(raw_line) - len(without_indent)
        fence = (
            _FENCE_RE.match(without_indent) if indent <= _MAX_MARKDOWN_INDENT else None
        )
        if fence is not None:
            marker = fence.group("marker")
            if fence_marker is None:
                # Opening fence. A backtick info string may not itself contain
                # a backtick; when it does the line is not a fence at all.
                if marker[0] == "`" and "`" in fence.group("info"):
                    yield raw_line
                    continue
                fence_marker = marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                # Closing fences carry no info string.
                if not fence.group("info").strip():
                    fence_marker = None
            continue
        if fence_marker is None:
            yield raw_line


def _mask_inline_code(line: str) -> str:
    """Blank out inline code spans, preserving column positions."""
    return _INLINE_CODE_RE.sub(lambda match: " " * len(match.group(0)), line)


def _trailer_on_line(line: str, field_names: Sequence[str]) -> tuple[str, str] | None:
    """``(field_name, value)`` if this line declares one of the trailers."""
    masked = _mask_inline_code(line)
    for field_name in field_names:
        prefix = f"{field_name}:"
        if masked[: len(prefix)].lower() == prefix.lower():
            return field_name, masked[len(prefix) :].strip()
    return None


def parse_trailer(body: str, field_names: Sequence[str]) -> str | None:
    """The single value declared for ``field_names``, or ``None`` if absent.

    ``field_names`` are aliases for ONE trailer: they must agree. The value is
    returned verbatim (an empty string when the field is present but has no
    value); callers own validation of what a legal value looks like.

    Raises:
        TrailerConflictError: two or more distinct values are declared.
    """
    declarations = [
        found
        for found in (
            _trailer_on_line(line, field_names) for line in iter_prose_lines(body or "")
        )
        if found is not None
    ]
    if not declarations:
        return None

    if len({value for _, value in declarations}) > 1:
        rendered = ", ".join(f"{name}: {value!r}" for name, value in declarations)
        raise TrailerConflictError(
            f"conflicting trailer values in the PR body ({rendered}); "
            "declare exactly one value"
        )
    return declarations[0][1]
