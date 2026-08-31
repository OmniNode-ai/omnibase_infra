# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fence grammar + conflict pins for the shared PR-body trailer parser.

OMN-17294. Both consumers of this module treat the parsed value as authority
over a required CI gate:

* ``Omnimarket-Source-Ref:`` picks the omnimarket tree the OMN-15361
  grant-derivation job runs against.
* ``Evidence-Source:`` names the OCC companion the STRICT ``CI Summary``
  gate proves durable.

So "which lines count as a declaration" is a security boundary, not a
formatting preference. These tests pin the boundary itself; the consumers'
own suites pin their behaviour through it.
"""

from __future__ import annotations

import pytest

from scripts.ci.pr_trailers import (
    TrailerConflictError,
    iter_prose_lines,
    parse_trailer,
)

pytestmark = pytest.mark.unit

FIELDS = ("Omnimarket-Source-Ref", "Node-Migration-Source-Ref")


class TestFenceTracking:
    def test_backtick_fence_contents_are_not_prose(self) -> None:
        body = "before\n```\ninside\n```\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_tilde_fence_contents_are_not_prose(self) -> None:
        body = "before\n~~~\ninside\n~~~\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_info_string_on_the_opening_fence(self) -> None:
        body = "before\n```python\ninside\n```\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_fence_of_the_other_character_does_not_close(self) -> None:
        body = "before\n```\n~~~\nstill inside\n```\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_shorter_run_does_not_close_a_longer_fence(self) -> None:
        body = "before\n````\n```\nstill inside\n````\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_longer_run_does_close_a_shorter_fence(self) -> None:
        body = "before\n```\ninside\n`````\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_closing_fence_may_not_carry_an_info_string(self) -> None:
        body = "before\n```\ninside\n```not-a-close\nstill inside\n```\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_fence_indented_up_to_three_spaces_still_opens(self) -> None:
        body = "before\n   ```\ninside\n   ```\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "after"]

    def test_four_space_indented_backticks_are_not_a_fence(self) -> None:
        body = "before\n    ```\nafter\n"
        assert list(iter_prose_lines(body)) == ["before", "    ```", "after"]

    def test_unterminated_fence_runs_to_end_of_body(self) -> None:
        body = "before\n```\ninside\nstill inside\n"
        assert list(iter_prose_lines(body)) == ["before"]

    def test_backtick_info_string_containing_a_backtick_is_not_a_fence(self) -> None:
        # CommonMark forbids a backtick in a backtick fence's info string.
        body = "```a`b\nnot fenced\n"
        assert list(iter_prose_lines(body)) == ["```a`b", "not fenced"]


class TestTrailerRecognition:
    def test_plain_trailer(self) -> None:
        assert parse_trailer("Omnimarket-Source-Ref: dev\n", FIELDS) == "dev"

    def test_field_name_is_case_insensitive(self) -> None:
        assert parse_trailer("omnimarket-source-ref: dev\n", FIELDS) == "dev"

    def test_alias_field_is_accepted(self) -> None:
        assert parse_trailer("Node-Migration-Source-Ref: dev\n", FIELDS) == "dev"

    def test_absent_returns_none(self) -> None:
        assert parse_trailer("no trailer here\n", FIELDS) is None
        assert parse_trailer("", FIELDS) is None

    def test_present_but_empty_returns_empty_string(self) -> None:
        # Distinct from absent: callers own the validation of the value.
        assert parse_trailer("Omnimarket-Source-Ref:\n", FIELDS) == ""

    def test_fenced_declaration_is_ignored(self) -> None:
        body = "```\nOmnimarket-Source-Ref: attacker/branch\n```\n"
        assert parse_trailer(body, FIELDS) is None

    def test_real_trailer_below_a_fenced_decoy_wins(self) -> None:
        body = (
            "```\nOmnimarket-Source-Ref: attacker/branch\n```\n"
            "Omnimarket-Source-Ref: real/branch\n"
        )
        assert parse_trailer(body, FIELDS) == "real/branch"

    def test_indented_declaration_is_ignored(self) -> None:
        assert parse_trailer(" Omnimarket-Source-Ref: x\n", FIELDS) is None
        assert parse_trailer("    Omnimarket-Source-Ref: x\n", FIELDS) is None

    def test_blockquoted_declaration_is_ignored(self) -> None:
        assert parse_trailer("> Omnimarket-Source-Ref: x\n", FIELDS) is None

    def test_whole_line_inline_code_span_is_ignored(self) -> None:
        assert parse_trailer("`Omnimarket-Source-Ref: x`\n", FIELDS) is None

    def test_field_name_mid_sentence_is_ignored(self) -> None:
        body = "Set Omnimarket-Source-Ref: x in the trailer block.\n"
        assert parse_trailer(body, FIELDS) is None


class TestConflicts:
    def test_two_distinct_values_raise(self) -> None:
        body = "Omnimarket-Source-Ref: a\nOmnimarket-Source-Ref: b\n"
        with pytest.raises(TrailerConflictError) as excinfo:
            parse_trailer(body, FIELDS)
        message = str(excinfo.value)
        assert "'a'" in message and "'b'" in message

    def test_disagreeing_aliases_raise(self) -> None:
        body = "Omnimarket-Source-Ref: a\nNode-Migration-Source-Ref: b\n"
        with pytest.raises(TrailerConflictError):
            parse_trailer(body, FIELDS)

    def test_repeated_identical_value_is_not_a_conflict(self) -> None:
        body = "Omnimarket-Source-Ref: a\nOmnimarket-Source-Ref: a\n"
        assert parse_trailer(body, FIELDS) == "a"

    def test_a_fenced_decoy_never_creates_a_conflict(self) -> None:
        body = (
            "```\nOmnimarket-Source-Ref: attacker/branch\n```\n"
            "Omnimarket-Source-Ref: real/branch\n"
        )
        assert parse_trailer(body, FIELDS) == "real/branch"

    def test_conflict_error_is_a_value_error(self) -> None:
        # Consumers that already catch ValueError keep their handling.
        assert issubclass(TrailerConflictError, ValueError)
