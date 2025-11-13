#!/usr/bin/env python3
"""
Response Parser for LLM Code Generation.

Extracts and parses Python code from LLM responses, handling various
output formats (markdown code blocks, plain code, etc.).

Performance Target: <5ms per parse operation
"""

import ast
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelParsedResponse(BaseModel):
    """Parsed LLM response with extracted code and metadata."""

    raw_response: str = Field(..., description="Original LLM response")
    extracted_code: str = Field(..., description="Extracted Python code")
    code_blocks_found: int = Field(default=0, description="Number of code blocks found")
    parse_method: str = Field(
        ..., description="Method used to extract code (markdown/plain/fallback)"
    )
    syntax_valid: bool = Field(..., description="Whether code is syntactically valid")
    parse_error: Optional[str] = Field(None, description="Parse error if any")
    ast_tree: Optional[ast.Module] = Field(
        None, description="Parsed AST tree (if syntax valid)"
    )

    class Config:
        arbitrary_types_allowed = True  # Allow AST objects


class EnhancedResponseParser:
    """
    Parse and extract code from LLM responses.

    Handles multiple response formats:
    - Markdown code blocks (```python ... ```)
    - Plain Python code
    - Mixed text and code
    - Multiple code blocks

    Attributes:
        prefer_longest: Whether to prefer longest code block if multiple found
        strip_comments: Whether to strip leading comment blocks

    Example:
        >>> parser = EnhancedResponseParser()
        >>> parsed = parser.parse_llm_response(llm_response_text)
        >>> if parsed.syntax_valid:
        ...     print(f"Extracted {len(parsed.extracted_code)} chars of code")
    """

    def __init__(
        self,
        prefer_longest: bool = True,
        strip_comments: bool = False,
    ):
        """
        Initialize response parser.

        Args:
            prefer_longest: If multiple code blocks found, use longest one
            strip_comments: Strip leading comment blocks from extracted code
        """
        self.prefer_longest = prefer_longest
        self.strip_comments = strip_comments

        logger.debug(
            f"EnhancedResponseParser initialized "
            f"(prefer_longest={prefer_longest}, strip_comments={strip_comments})"
        )

    def parse_llm_response(
        self,
        response: str,
    ) -> ModelParsedResponse:
        """
        Parse LLM response and extract Python code.

        Tries multiple extraction strategies:
        1. Markdown code blocks (```python)
        2. Generic code blocks (```)
        3. Plain text parsing
        4. Fallback to entire response

        Args:
            response: Raw LLM response text

        Returns:
            ModelParsedResponse with extracted code and metadata

        Performance: <5ms per parse
        """
        # Try extraction strategies in order
        code_blocks = self._extract_code_blocks(response)

        if code_blocks:
            # Found markdown code blocks
            if self.prefer_longest:
                extracted_code = max(code_blocks, key=len)
            else:
                extracted_code = code_blocks[0]

            parse_method = "markdown"
            logger.debug(
                f"Extracted code from markdown blocks ({len(code_blocks)} found)"
            )

        else:
            # No markdown blocks - try plain text
            extracted_code = response.strip()
            parse_method = "plain"
            logger.debug("No markdown blocks found, using plain text")

        # Clean code
        if self.strip_comments:
            extracted_code = self._strip_leading_comments(extracted_code)

        # Validate syntax
        syntax_valid, parse_error, ast_tree = self._parse_code_to_ast(extracted_code)

        return ModelParsedResponse(
            raw_response=response,
            extracted_code=extracted_code,
            code_blocks_found=len(code_blocks),
            parse_method=parse_method,
            syntax_valid=syntax_valid,
            parse_error=parse_error,
            ast_tree=ast_tree,
        )

    def _extract_code_blocks(
        self,
        response: str,
    ) -> list[str]:
        """
        Extract code blocks from markdown.

        Supports both:
        - ```python ... ```
        - ``` ... ```

        Args:
            response: Response text

        Returns:
            List of extracted code blocks (may be empty)
        """
        # Pattern for markdown code blocks
        # Matches: ```python\ncode\n``` or ```\ncode\n```
        pattern = r"```(?:python)?\s*\n(.*?)\n```"

        matches = re.findall(pattern, response, re.DOTALL)

        # Clean up extracted blocks
        cleaned_blocks = []
        for block in matches:
            cleaned = block.strip()
            if cleaned:
                cleaned_blocks.append(cleaned)

        return cleaned_blocks

    def _strip_leading_comments(
        self,
        code: str,
    ) -> str:
        """
        Strip leading comment blocks from code.

        Useful when LLM adds explanatory comments before actual code.

        Args:
            code: Code with potential leading comments

        Returns:
            Code with leading comments removed
        """
        lines = code.split("\n")

        # Find first non-comment line
        first_code_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_code_line = i
                break

        # Return code starting from first non-comment line
        return "\n".join(lines[first_code_line:])

    def _parse_code_to_ast(
        self,
        code: str,
    ) -> tuple[bool, Optional[str], Optional[ast.Module]]:
        """
        Parse code into AST for syntax validation.

        Args:
            code: Python code string

        Returns:
            Tuple of (syntax_valid, error_message, ast_tree)
        """
        try:
            tree = ast.parse(code)
            return True, None, tree

        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.warning(f"Code syntax error: {error_msg}")
            return False, error_msg, None

        except Exception as e:
            error_msg = f"Parse error: {e!s}"
            logger.warning(f"Code parse error: {error_msg}")
            return False, error_msg, None

    def extract_imports(
        self,
        ast_tree: ast.Module,
    ) -> list[str]:
        """
        Extract import statements from AST.

        Args:
            ast_tree: Parsed AST tree

        Returns:
            List of import statement strings
        """
        imports = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(alias.name for alias in node.names)
                imports.append(f"from {module} import {names}")

        return imports

    def extract_functions(
        self,
        ast_tree: ast.Module,
    ) -> list[str]:
        """
        Extract function names from AST.

        Args:
            ast_tree: Parsed AST tree

        Returns:
            List of function names (both sync and async)
        """
        functions = []

        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        return functions

    def extract_classes(
        self,
        ast_tree: ast.Module,
    ) -> list[str]:
        """
        Extract class names from AST.

        Args:
            ast_tree: Parsed AST tree

        Returns:
            List of class names
        """
        classes = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return classes

    def get_code_statistics(
        self,
        code: str,
    ) -> dict[str, int]:
        """
        Get basic code statistics.

        Args:
            code: Python code string

        Returns:
            Dictionary with statistics (lines, chars, functions, etc.)
        """
        lines = code.split("\n")

        stats = {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "comment_lines": len(
                [line for line in lines if line.strip().startswith("#")]
            ),
            "total_chars": len(code),
        }

        # Parse to get more detailed stats
        syntax_valid, _, ast_tree = self._parse_code_to_ast(code)

        if syntax_valid and ast_tree:
            stats["functions"] = len(self.extract_functions(ast_tree))
            stats["classes"] = len(self.extract_classes(ast_tree))
            stats["imports"] = len(self.extract_imports(ast_tree))
        else:
            stats["functions"] = 0
            stats["classes"] = 0
            stats["imports"] = 0

        return stats


__all__ = ["EnhancedResponseParser", "ModelParsedResponse"]
