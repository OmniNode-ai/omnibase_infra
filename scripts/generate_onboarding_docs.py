#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Generate onboarding docs from canonical graph (OMN-5272).

Loads canonical graph, resolves each built-in policy, renders via
RendererOnboardingMarkdown, and writes to docs/getting-started/.

Usage:
    # Generate docs
    python scripts/generate_onboarding_docs.py

    # Check mode (CI drift detection)
    python scripts/generate_onboarding_docs.py --check
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import (
    load_builtin_policies,
    resolve_policy,
)
from omnibase_infra.onboarding.renderers.renderer_markdown import (
    RendererOnboardingMarkdown,
)

DOCS_DIR = Path(__file__).parent.parent / "docs" / "getting-started"

POLICY_TO_TITLE = {
    "standalone_quickstart": "Standalone Quickstart",
    "contributor_local": "Contributor Local Setup",
    "full_platform": "Full Platform Setup",
}

POLICY_TO_FILENAME = {
    "standalone_quickstart": "standalone-quickstart.md",
    "contributor_local": "contributor-local.md",
    "full_platform": "full-platform.md",
}


def generate_docs(output_dir: Path) -> dict[str, str]:
    """Generate all policy docs.

    Args:
        output_dir: Directory to write generated docs.

    Returns:
        Dict mapping filename to generated content.
    """
    graph = load_canonical_graph()
    policies = load_builtin_policies()
    renderer = RendererOnboardingMarkdown()
    generated: dict[str, str] = {}

    for policy_name, policy_data in policies.items():
        targets = policy_data.get("target_capabilities", [])
        if not isinstance(targets, list):
            continue

        title = POLICY_TO_TITLE.get(policy_name, policy_name)
        filename = POLICY_TO_FILENAME.get(policy_name, f"{policy_name}.md")

        steps = resolve_policy(graph, target_capabilities=targets)
        content = renderer.render(steps, title=title)
        generated[filename] = content

        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    return generated


def check_docs() -> bool:
    """Check if docs are up to date.

    Returns:
        True if docs match generated content, False if stale.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        generated = generate_docs(tmp_dir)

        all_match = True
        for filename, expected_content in generated.items():
            existing_path = DOCS_DIR / filename
            if not existing_path.exists():
                print(f"MISSING: {existing_path}")
                all_match = False
                continue

            existing_content = existing_path.read_text(encoding="utf-8")
            if existing_content != expected_content:
                diff = difflib.unified_diff(
                    existing_content.splitlines(keepends=True),
                    expected_content.splitlines(keepends=True),
                    fromfile=str(existing_path),
                    tofile=f"generated/{filename}",
                )
                print(f"STALE: {existing_path}")
                sys.stdout.writelines(diff)
                all_match = False

        return all_match


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate onboarding docs from canonical graph"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: exit non-zero if docs are stale",
    )
    args = parser.parse_args()

    if args.check:
        if check_docs():
            print("OK: All onboarding docs are up to date")
            sys.exit(0)
        else:
            print("FAIL: Onboarding docs are stale. Run this script to regenerate.")
            sys.exit(1)
    else:
        generated = generate_docs(DOCS_DIR)
        for filename in generated:
            print(f"Generated: {DOCS_DIR / filename}")


if __name__ == "__main__":
    main()
