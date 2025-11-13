#!/usr/bin/env python3
"""Fix string version anti-patterns by removing __version__ from __init__.py files."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

# Files to fix (from validation output)
FILES_TO_FIX = [
    "src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/__init__.py",
    "src/omninode_bridge/ci/__init__.py",
]


def remove_version_line(file_path: Path) -> bool:
    """Remove __version__ = "..." line from file."""
    try:
        content = file_path.read_text()

        # Pattern to match __version__ = "x.y.z"
        pattern = r'^__version__\s*=\s*["\'][^"\']+["\']\s*\n'

        new_content = re.sub(pattern, "", content, flags=re.MULTILINE)

        if new_content != content:
            file_path.write_text(new_content)
            print(f"✓ Fixed: {file_path.relative_to(PROJECT_ROOT)}")
            return True
        else:
            print(f"⚠ No __version__ found: {file_path.relative_to(PROJECT_ROOT)}")
            return False

    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False


def main():
    """Fix all __version__ declarations."""
    print("Fixing string version anti-patterns...")
    print("=" * 70)

    fixed_count = 0

    for file_rel in FILES_TO_FIX:
        file_path = PROJECT_ROOT / file_rel
        if file_path.exists():
            if remove_version_line(file_path):
                fixed_count += 1

    print("=" * 70)
    print(f"Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
