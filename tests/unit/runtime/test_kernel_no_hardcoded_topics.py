# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Regression test: no hardcoded env-prefixed topic literals in kernel/runtime.

OMN-1972 Phase 1 removed all hardcoded environment-prefixed topic strings from
the kernel and runtime host process. This test ensures those patterns cannot be
reintroduced.

Topics must be realm-agnostic (no environment prefix). The runtime resolves
topics via TopicResolver and reads subscribe_topics from orchestrator contracts.
Environment isolation is achieved through consumer group naming, not topic
name prefixing.

Checked patterns:
    - f-string patterns: f"{environment}.onex." or f"{env}.onex."
    - Hardcoded env literals: "dev.onex.", "prod.onex.", "staging.onex.", etc.

Note:
    This test uses Python's ``ast`` module to extract only executable code lines,
    excluding comments and docstrings. Docstrings may legitimately contain
    env-prefixed topic examples for documentation purposes.
"""

import ast
import re
from pathlib import Path

import pytest

# Pattern matches: f"{environment}.onex." or f"{env}.onex." or "dev.onex." etc.
ENV_PREFIX_PATTERNS: list[str] = [
    # f-string patterns with environment variable
    r'f"[^"]*\{environment\}\.onex\.',
    r"f'[^']*\{environment\}\.onex\.",
    r'f"[^"]*\{env\}\.onex\.',
    r"f'[^']*\{env\}\.onex\.",
    # Hardcoded env prefix patterns
    r'"dev\.onex\.',
    r'"prod\.onex\.',
    r'"staging\.onex\.',
    r'"local\.onex\.',
    r'"test\.onex\.',
]

# Critical runtime files that must remain free of env-prefixed topics.
# These are the files where hardcoded topics were historically introduced.
CHECKED_FILES: list[str] = [
    "src/omnibase_infra/runtime/service_kernel.py",
    "src/omnibase_infra/runtime/service_runtime_host_process.py",
]


def _get_docstring_lines(source: str) -> set[int]:
    """Return set of 1-indexed line numbers that belong to docstrings.

    Uses the ``ast`` module to find all string expressions (docstrings) in
    classes, functions, modules, and async functions, then returns the set
    of line numbers spanned by those docstrings. These lines are excluded
    from the env-prefix scan since documentation may legitimately contain
    env-prefixed topic examples.
    """
    docstring_lines: set[int] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return docstring_lines

    for node in ast.walk(tree):
        # Docstrings are Expr nodes containing a Constant string as the
        # first statement in a module, class, or function body.
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_node = node.body[0]
                for line_num in range(
                    docstring_node.lineno, docstring_node.end_lineno + 1
                ):
                    docstring_lines.add(line_num)

    return docstring_lines


class TestNoHardcodedTopics:
    """Regression: prevent reintroduction of env-prefixed topic literals."""

    @pytest.mark.parametrize("rel_path", CHECKED_FILES)
    def test_no_env_prefixed_topics(self, rel_path: str) -> None:
        """Ensure no environment-prefixed topic patterns exist in critical runtime files.

        This is a grep-based regression test. If a developer accidentally adds
        back an env-prefixed topic string (e.g., f"{environment}.onex.evt..."),
        this test will catch it and point to the exact line.

        The fix is to use TopicResolver.resolve() with env-free suffixes
        (e.g., "onex.evt.platform.node-introspection.v1") and read
        subscribe_topics from orchestrator contracts.

        Note:
            Lines inside docstrings and comments are excluded from scanning.
            Docstrings may contain env-prefixed examples for documentation.
        """
        repo_root = (
            Path(__file__).resolve().parents[3]
        )  # tests/unit/runtime -> repo root
        file_path = repo_root / rel_path

        if not file_path.exists():
            pytest.skip(f"File not found: {file_path}")

        content = file_path.read_text()
        docstring_lines = _get_docstring_lines(content)
        violations: list[str] = []

        for line_num, line in enumerate(content.splitlines(), 1):
            # Skip docstring lines - they may legitimately reference the pattern
            if line_num in docstring_lines:
                continue
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in ENV_PREFIX_PATTERNS:
                if re.search(pattern, line):
                    violations.append(f"  Line {line_num}: {line.strip()}")

        assert not violations, (
            f"Found env-prefixed topic patterns in {rel_path}:\n"
            + "\n".join(violations)
            + "\n\nTopics must be realm-agnostic. Use TopicResolver.resolve() with "
            "env-free suffixes instead."
        )
