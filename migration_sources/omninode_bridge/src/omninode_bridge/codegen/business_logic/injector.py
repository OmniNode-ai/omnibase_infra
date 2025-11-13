#!/usr/bin/env python3
"""
Code Injector - Replaces template stubs with LLM-generated business logic.

This module provides intelligent code injection that:
1. Detects various stub patterns (TODO comments, pass statements, NotImplementedError)
2. Preserves code formatting and indentation
3. Maintains method signatures and docstrings
4. Handles edge cases (missing stubs, multiple methods)

ONEX v2.0 Compliance:
- Type-safe injection with validation
- Structured error handling
- Performance tracking
- Code formatting preservation

Example Usage:
    injector = CodeInjector()

    # Find all stubs in a file
    stubs = injector.find_stubs(template_code)

    # Inject generated code
    modified = injector.inject_code(
        template_code=template_code,
        generated_code=llm_output,
        stub_info=stubs[0]
    )
"""

import re

from .models import StubInfo


class CodeInjectionError(Exception):
    """Raised when code injection fails."""

    pass


class CodeInjector:
    """
    Replaces template stubs with LLM-generated business logic.

    Supported stub patterns:
    1. Method stubs with TODO comments:
       def method_name(self, ...) -> ReturnType:
           # TODO: Implement business logic
           pass

    2. Method stubs with NotImplementedError:
       def method_name(...) -> ReturnType:
           raise NotImplementedError("TODO: Implement")

    3. Method stubs with IMPLEMENTATION REQUIRED marker:
       async def method_name(self, ...) -> ReturnType:
           # IMPLEMENTATION REQUIRED
           pass

    4. Simple pass stubs:
       def method_name(self, ...) -> ReturnType:
           pass
    """

    def __init__(self):
        """Initialize CodeInjector with stub detection patterns."""
        # Regex patterns for stub detection
        # Pattern explanation:
        # - Captures method signature (def/async def + name + params + return type)
        # - Captures optional docstring
        # - Matches various stub patterns (TODO, IMPLEMENTATION REQUIRED, pass, NotImplementedError)
        self.stub_patterns = {
            # Pattern 1: TODO comment + pass
            "todo_pass": re.compile(
                r"((?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:(?:\s*#[^\n]*)?\s*\n"
                r'(?:\s*"""(?:(?!""").)*?"""\s*\n)?'  # Optional docstring (non-greedy, won't cross docstrings)
                r"\s*#\s*TODO[^\n]*\n"
                r"\s*)pass",
                re.DOTALL | re.MULTILINE,
            ),
            # Pattern 2: IMPLEMENTATION REQUIRED marker
            "implementation_required": re.compile(
                r"((?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:(?:\s*#[^\n]*)?\s*\n"
                r'(?:\s*"""(?:(?!""").)*?"""\s*\n)?'  # Optional docstring (non-greedy, won't cross docstrings)
                r"\s*#\s*IMPLEMENTATION REQUIRED[^\n]*\n"
                r"\s*)(?:pass|\.\.\.)",
                re.DOTALL | re.MULTILINE,
            ),
            # Pattern 3: NotImplementedError
            "not_implemented": re.compile(
                r"((?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:(?:\s*#[^\n]*)?\s*\n"
                r'(?:\s*"""(?:(?!""").)*?"""\s*\n)?'  # Optional docstring (non-greedy, won't cross docstrings)
                r"\s*)raise\s+NotImplementedError",
                re.DOTALL | re.MULTILINE,
            ),
            # Pattern 4: Simple pass (fallback)
            "simple_pass": re.compile(
                r"((?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:(?:\s*#[^\n]*)?\s*\n"
                r'(?:\s*"""(?:(?!""").)*?"""\s*\n)?'  # Optional docstring (non-greedy, won't cross docstrings)
                r"\s*)pass",
                re.DOTALL | re.MULTILINE,
            ),
        }

    def inject_code(
        self, template_code: str, generated_code: str, stub_info: StubInfo
    ) -> str:
        """
        Inject generated code into template, replacing stub.

        Args:
            template_code: Original template code
            generated_code: LLM-generated implementation (method body only)
            stub_info: Information about the stub to replace

        Returns:
            Modified template code with injected implementation

        Raises:
            CodeInjectionError: If stub cannot be found or injection fails
        """
        try:
            # Find method and stub locations
            method_start, method_end = self._find_method_location(
                template_code, stub_info
            )
            stub_body_start, stub_body_end = self._find_stub_body_in_method(
                template_code, method_start, method_end, stub_info
            )

            # Extract method body indentation
            indentation = self._extract_method_body_indentation(
                template_code, method_start
            )

            # Format generated code with proper indentation
            formatted_code = self._format_with_indentation(generated_code, indentation)

            # Replace ONLY the stub body, preserving method signature and docstring
            modified_code = (
                template_code[:stub_body_start]
                + formatted_code
                + template_code[stub_body_end:]
            )

            return modified_code

        except ValueError as e:
            raise CodeInjectionError(
                f"Failed to inject code for method '{stub_info.method_name}': {e}"
            ) from e
        except Exception as e:
            raise CodeInjectionError(
                f"Failed to inject code for method '{stub_info.method_name}': {e}"
            ) from e

    def _find_method_location(self, code: str, stub_info: StubInfo) -> tuple[int, int]:
        """
        Find start and end positions of method in code.

        Args:
            code: Template code
            stub_info: Stub information with method name

        Returns:
            Tuple of (start_position, end_position) of method

        Raises:
            ValueError: If method cannot be found
        """
        # Search for method signature
        method_pattern = re.compile(
            rf"(?:async\s+)?def\s+{re.escape(stub_info.method_name)}\s*\([^)]*\)",
            re.MULTILINE,
        )

        match = method_pattern.search(code)
        if not match:
            raise ValueError(f"Method '{stub_info.method_name}' not found in template")

        # Find the end of the method (next method or class end)
        start_pos = match.start()
        end_pos = self._find_method_end(code, start_pos)

        return start_pos, end_pos

    def _find_stub_body_in_method(
        self, code: str, method_start: int, method_end: int, stub_info: StubInfo
    ) -> tuple[int, int]:
        """
        Find the stub body within a method (excluding signature and docstring).

        Args:
            code: Template code
            method_start: Position where method starts
            method_end: Position where method ends
            stub_info: Stub information

        Returns:
            Tuple of (stub_body_start, stub_body_end) positions

        Raises:
            ValueError: If stub body cannot be found
        """
        method_text = code[method_start:method_end]

        # Find the colon at end of method signature
        # Strategy: Find last closing parenthesis (end of parameters), then find colon after it
        # This handles multiline signatures and complex type hints (which may contain colons)
        closing_paren_pos = method_text.rfind(")")
        if closing_paren_pos == -1:
            raise ValueError("Invalid method definition (no closing parenthesis found)")

        # Search for : followed by newline after the closing paren
        # Note: Use [ \t]* instead of \s* before # to avoid matching TODO comments on next line
        remaining_after_paren = method_text[closing_paren_pos:]
        colon_match = re.search(r":(?:[ \t]*#[^\n]*)?\s*\n", remaining_after_paren)
        if not colon_match:
            raise ValueError(
                "Invalid method definition (no colon found after parameters)"
            )

        # Start searching after the signature line (closing paren + colon + newline)
        search_start = closing_paren_pos + colon_match.end()

        # Skip docstring if present
        # Use the same signature_end position to search for docstring right after it
        remaining_text = method_text[search_start:]
        docstring_match = re.search(
            r'^\s*"""(?:(?!""").)*?"""\s*\n', remaining_text, re.DOTALL
        )
        if docstring_match:
            search_start += docstring_match.end()

        # Find stub patterns in remaining method body
        stub_body_text = method_text[search_start:]

        # Look for TODO, pass, NotImplementedError, or IMPLEMENTATION REQUIRED
        stub_patterns = [
            (r"\s*#\s*TODO[^\n]*\n\s*pass", "TODO + pass"),
            (
                r"\s*#\s*IMPLEMENTATION REQUIRED[^\n]*\n\s*(?:pass|\.\.\.)",
                "IMPLEMENTATION REQUIRED",
            ),
            (r"\s*raise\s+NotImplementedError[^\n]*", "NotImplementedError"),
            (r"^\s*pass\s*$", "simple pass"),
        ]

        for pattern, pattern_name in stub_patterns:
            match = re.search(pattern, stub_body_text, re.MULTILINE)
            if match:
                # Find the start of the line containing the match
                # (we need to include the leading whitespace in the replacement)
                line_start_in_stub_text = (
                    stub_body_text.rfind("\n", 0, match.start()) + 1
                )

                # Calculate absolute positions
                # Start at beginning of line (including indentation)
                stub_body_start = method_start + search_start + line_start_in_stub_text
                stub_body_end = method_start + search_start + match.end()
                return stub_body_start, stub_body_end

        # If we have line numbers from stub_info, use those as fallback
        if stub_info.line_start and stub_info.line_end:
            lines = code.split("\n")
            stub_body_start = sum(
                len(line) + 1 for line in lines[: stub_info.line_start - 1]
            )
            stub_body_end = sum(len(line) + 1 for line in lines[: stub_info.line_end])
            # Adjust to exclude method signature
            if stub_body_start < method_start + search_start:
                stub_body_start = method_start + search_start
            return stub_body_start, stub_body_end

        raise ValueError(f"No stub body found in method '{stub_info.method_name}'")

    def _find_method_end(self, code: str, start_pos: int) -> int:
        """
        Find the end of a method definition.

        Detects method end by finding where indentation decreases back to
        the method definition level (or file end).

        Args:
            code: Full code string
            start_pos: Position where method definition starts

        Returns:
            Position where method ends
        """
        # Extract method indentation from the original code line containing start_pos
        # This ensures we get the correct indentation even when start_pos points
        # to the middle of the line (e.g., the 'd' in 'def' after leading spaces)
        line_start = code.rfind("\n", 0, start_pos) + 1
        line_end = code.find("\n", start_pos)
        if line_end == -1:
            line_end = len(code)
        first_line = code[line_start:line_end]
        method_indent = len(first_line) - len(first_line.lstrip())

        # Split code from start_pos for iteration
        lines = code[start_pos:].split("\n")
        if not lines:
            return start_pos

        # Find the signature-ending colon first (handles multiline signatures)
        # For multiline signatures, closing ) might be at same indent as def line
        # Check if signature ends on first line (single-line signature)
        signature_ended = lines[0].strip().endswith(":")

        position = start_pos
        for i, line in enumerate(lines[1:], 1):
            position += len(lines[i - 1]) + 1  # +1 for newline

            # Check if we've passed the signature (line ending with :)
            if not signature_ended and line.strip().endswith(":"):
                signature_ended = True
                continue

            # Only check for method end after signature has ended
            if signature_ended and line.strip():
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= method_indent:
                    # Found end of method (next method/class/module-level code)
                    return position

        return len(code)  # End of file

    def _extract_indentation(self, code: str, position: int) -> str:
        """
        Extract indentation from a position in code.

        Args:
            code: Full code string
            position: Position to extract indentation from

        Returns:
            Indentation string (spaces or tabs)
        """
        # Find the start of the line containing this position
        line_start = code.rfind("\n", 0, position) + 1

        # Get the line and extract leading whitespace
        line_end = code.find("\n", position)
        if line_end == -1:
            line_end = len(code)

        line = code[line_start:line_end]
        return line[: len(line) - len(line.lstrip())]

    def _extract_method_body_indentation(self, code: str, method_start: int) -> str:
        """
        Extract the indentation level for method body (not method definition).

        Returns method definition indentation + 4 spaces (standard Python indentation).

        Args:
            code: Full code string
            method_start: Position where method definition starts

        Returns:
            Indentation string for method body (spaces or tabs)
        """
        # Get the indentation of the method definition itself
        method_indent = self._extract_indentation(code, method_start)

        # Method body is indented one level (4 spaces) more than the method definition
        return method_indent + "    "

    def _format_with_indentation(self, code: str, indentation: str) -> str:
        """
        Add indentation to each line of code.

        Preserves relative indentation within the code block.

        Args:
            code: Code to indent
            indentation: Base indentation to add

        Returns:
            Indented code string
        """
        lines = code.split("\n")
        formatted_lines = []

        for line in lines:
            if line.strip():  # Non-empty line
                formatted_lines.append(indentation + line)
            else:  # Empty line - preserve it
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def find_stubs(self, code: str, file_path: str = "") -> list[StubInfo]:
        """
        Find all stubs in code.

        Searches for various stub patterns and returns structured information
        about each detected stub.

        Args:
            code: Code to search for stubs
            file_path: Optional file path for StubInfo objects

        Returns:
            List of StubInfo for each detected stub
        """
        stubs = []
        seen_methods = set()  # Avoid duplicates

        # Search for stubs using all patterns
        for pattern_name, pattern in self.stub_patterns.items():
            for match in pattern.finditer(code):
                method_name = (
                    match.group(2)
                    if match.groups() and len(match.groups()) >= 2
                    else "unknown"
                )

                # Skip if we've already found this method
                if method_name in seen_methods:
                    continue

                seen_methods.add(method_name)

                # Calculate line numbers
                line_start = code[: match.start()].count("\n") + 1
                line_end = code[: match.end()].count("\n") + 1

                # Extract method signature
                signature_match = re.search(
                    rf"(?:async\s+)?def\s+{re.escape(method_name)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
                    code,
                )
                signature = (
                    signature_match.group(0).rstrip(":") if signature_match else ""
                )

                # Extract docstring if present
                docstring = None
                docstring_match = re.search(
                    r'"""(.*?)"""',
                    code[match.start() : match.end()],
                    re.DOTALL,
                )
                if docstring_match:
                    docstring = docstring_match.group(1).strip()

                stubs.append(
                    StubInfo(
                        file_path=file_path,
                        method_name=method_name,
                        stub_code=match.group(0),
                        line_start=line_start,
                        line_end=line_end,
                        signature=signature,
                        docstring=docstring,
                    )
                )

        return stubs


__all__ = ["CodeInjector", "CodeInjectionError"]
