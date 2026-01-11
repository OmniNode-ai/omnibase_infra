"""Unit tests for docstring detection edge cases in architecture compliance.

These tests verify that the _scan_file_for_imports function correctly handles
various edge cases in string literal detection to avoid false negatives
(missing violations) or false positives (flagging imports in strings).

Ticket: OMN-255
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.ci.test_architecture_compliance import _scan_file_for_imports

if TYPE_CHECKING:
    from collections.abc import Callable


class TestDocstringDetectionEdgeCases:
    """Test edge cases in multiline string detection."""

    def test_basic_import_detected(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Verify basic import is detected."""
        content = "import kafka"
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1
        assert violations[0].import_pattern == "kafka"

    def test_from_import_detected(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Verify from import is detected."""
        content = "from kafka import KafkaProducer"
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1

    def test_import_in_triple_double_quote_docstring_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import inside triple double-quote docstring should be skipped."""
        content = (
            '"""\nThis docstring mentions import kafka for documentation.\n"""\nx = 1\n'
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0

    def test_import_in_triple_single_quote_docstring_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import inside triple single-quote docstring should be skipped."""
        content = (
            "'''\nThis docstring mentions import kafka for documentation.\n'''\nx = 1\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0

    def test_import_in_single_line_docstring_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import inside single-line docstring should be skipped."""
        content = '"""Example: import kafka for streaming."""\nx = 1\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0

    def test_import_after_docstring_detected(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import after docstring ends should be detected."""
        content = '"""This is a docstring."""\nimport kafka\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1

    def test_import_in_comment_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import in comment line should be skipped."""
        content = "# import kafka is not used here\nx = 1\n"
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0

    def test_triple_single_inside_triple_double_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Triple single quotes inside triple double quotes should be skipped.

        This is a critical edge case: when ''' appears inside a triple-double-quote
        string, the inner quotes should not affect the multiline string tracking.
        """
        content = (
            '"""\n'
            "This docstring has ''' inside and mentions import kafka.\n"
            '"""\n'
            "x = 1\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0, (
            "Import inside docstring with mixed quotes should be skipped"
        )

    def test_triple_double_inside_triple_single_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Triple double quotes inside triple single quotes should be skipped.

        This is a critical edge case: when triple-double-quotes appear inside
        a triple-single-quote string, they should not affect tracking.
        """
        content = (
            "'''\n"
            'This docstring has """ inside and mentions import kafka.\n'
            "'''\n"
            "x = 1\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0, (
            "Import inside docstring with mixed quotes should be skipped"
        )

    def test_raw_string_multiline_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import inside raw multiline string should be skipped."""
        content = 'r"""\nRaw string with import kafka inside.\n"""\nx = 1\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0, "Import inside raw string should be skipped"

    def test_fstring_multiline_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import inside f-string multiline should be skipped."""
        content = 'f"""\nF-string with import kafka inside.\n"""\nx = 1\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0, "Import inside f-string should be skipped"

    def test_bytes_multiline_skipped(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import inside bytes multiline should be skipped."""
        content = 'b"""\nBytes string with import kafka inside.\n"""\nx = 1\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0, "Import inside bytes string should be skipped"

    def test_multiple_docstrings_handled(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Multiple docstrings in sequence should be handled correctly."""
        content = (
            '"""First docstring with import kafka."""\n'
            "\n"
            '"""Second docstring with import kafka."""\n'
            "\n"
            "import kafka  # This should be detected\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1, (
            "Only the actual import should be detected, not those in docstrings"
        )

    def test_string_assignment_not_detected(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Import text in string assignment should not be detected.

        Note: This is about ensuring we don't have false positives when
        import text appears in string literals (not at line start).
        """
        content = 'x = "import kafka"\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        # The regex requires import to be at line start (after optional whitespace)
        # so this should not be detected
        assert len(violations) == 0, (
            "String assignment with import text should not be detected"
        )

    def test_empty_docstring_handled(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Empty docstring should be handled."""
        content = '""""""\nimport kafka\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1

    def test_nested_class_docstring(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Nested class with docstring should be handled."""
        content = (
            "class Outer:\n"
            '    """Outer docstring with import kafka example."""\n'
            "\n"
            "    class Inner:\n"
            '        """Inner docstring."""\n'
            "        pass\n"
            "\n"
            "import kafka  # This should be detected\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1

    def test_docstring_with_code_example(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Docstring containing import in code example should be skipped."""
        content = (
            '"""Example usage:\n'
            "\n"
            "    >>> import kafka\n"
            "    >>> producer = kafka.KafkaProducer()\n"
            "\n"
            '"""\n'
            "x = 1\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0, (
            "Import in docstring code example should be skipped"
        )

    def test_complex_mixed_quotes_pattern(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Complex pattern with both quote types interleaved."""
        content = (
            '"""This has import kafka mentioned."""\n'
            'x = "some value"\n'
            "'''\n"
            "Another multiline with import kafka.\n"
            "'''\n"
            "import kafka  # Real import\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1, (
            "Only the actual import outside strings should be detected"
        )

    def test_single_line_string_with_triple_single_containing_triple_double(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Single-line ''' string with triple-double-quotes inside.

        This is a critical edge case: '''text with \"\"\" inside''' should be
        treated as a single-line string, not as entering a multiline string.
        The inner \"\"\" should not affect the string state.
        """
        content = "'''Text with \"\"\" inside'''\nimport kafka\n"
        test_file = create_test_file(content, "triple_single_with_double.py")
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1, (
            "Import after single-line string with mixed quotes should be detected"
        )

    def test_single_line_string_with_triple_double_containing_triple_single(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Single-line triple-double-quote string with ''' inside.

        The inner ''' should not affect the string state.
        """
        content = '"""Text with \'\'\' inside"""\nimport kafka\n'
        test_file = create_test_file(content, "triple_double_with_single.py")
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1, (
            "Import after single-line string with mixed quotes should be detected"
        )

    def test_multiline_string_with_other_delimiter_on_middle_line(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Multiline string with the other delimiter type on a middle line."""
        content = '"""\nThis multiline has \'\'\' on this line.\n"""\nimport kafka\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1, (
            "Import after multiline with mixed quotes should be detected"
        )


class TestDocstringDetectionRobustness:
    """Test robustness of the docstring detection against malformed input."""

    def test_unclosed_docstring_doesnt_crash(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Unclosed docstring should not cause crash."""
        content = '"""This docstring is never closed\nimport kafka\n'
        test_file = create_test_file(content)
        # Should not crash, and import should be skipped (inside unclosed string)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert isinstance(violations, list)
        # The import is inside an unclosed multiline string, so no violation
        assert len(violations) == 0

    def test_unicode_in_docstring(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Unicode content in docstring should be handled."""
        content = (
            '"""\n'
            "Unicode: \u00e9\u00e0\u00fc\u4e2d\u6587 with import kafka example.\n"
            '"""\n'
            "x = 1\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0

    def test_very_long_docstring(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Very long docstring should be handled efficiently."""
        content = '"""' + "\n" + ("x " * 10000) + "\nimport kafka\n" + '"""' + "\nx = 1"
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 0

    def test_indented_docstring_in_function(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Indented docstring in function should be skipped."""
        content = (
            "def my_function():\n"
            '    """\n'
            "    This function does something.\n"
            "    Example: import kafka\n"
            '    """\n'
            "    pass\n"
            "\n"
            "import kafka  # Real import\n"
        )
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1

    def test_docstring_with_backslash_continuation(
        self,
        create_test_file: Callable[[str, str], Path],
        forbidden_patterns: list[str],
    ) -> None:
        """Docstring followed by line continuation should be handled."""
        content = '"""Docstring."""\nx = 1 + \\\n    2\nimport kafka\n'
        test_file = create_test_file(content)
        violations = _scan_file_for_imports(test_file, forbidden_patterns)
        assert len(violations) == 1
