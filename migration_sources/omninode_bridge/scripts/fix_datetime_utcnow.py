#!/usr/bin/env python3
"""
Script to replace datetime.utcnow() with datetime.now(UTC) for Python 3.12+ compatibility.
"""
import re
import subprocess
from pathlib import Path


def get_files_with_utcnow():
    """Get all Python files containing datetime.utcnow()"""
    result = subprocess.run(
        ["git", "grep", "-l", "datetime\\.utcnow()"],
        capture_output=True,
        text=True,
        cwd="/Volumes/PRO-G40/Code/omninode_bridge",
    )
    files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]
    return files


def fix_file(filepath):
    """Fix datetime.utcnow() occurrences in a file"""
    full_path = Path("/Volumes/PRO-G40/Code/omninode_bridge") / filepath

    if not full_path.exists():
        print(f"⚠️  File not found: {filepath}")
        return False

    with open(full_path, encoding="utf-8") as f:
        content = f.read()

    original_content = content
    changes_made = 0

    # Replace datetime.utcnow() with datetime.now(UTC)
    content, count = re.subn(r"datetime\.utcnow\(\)", "datetime.now(UTC)", content)
    changes_made += count

    # Ensure UTC is imported (handle various import patterns)
    if changes_made > 0:
        # Check if UTC is already imported
        has_utc_import = bool(re.search(r"from datetime import.*\bUTC\b", content))

        if not has_utc_import:
            # Pattern 1: from datetime import datetime
            content = re.sub(
                r"^(from datetime import )datetime(\s*$)",
                r"\1UTC, datetime\2",
                content,
                flags=re.MULTILINE,
            )

            # Pattern 2: from datetime import datetime, X, Y
            content = re.sub(
                r"^(from datetime import )datetime,",
                r"\1UTC, datetime,",
                content,
                flags=re.MULTILINE,
            )

            # Pattern 3: from datetime import timedelta, datetime
            content = re.sub(
                r"^(from datetime import )(\w+)(, datetime)",
                r"\1\2, UTC\3",
                content,
                flags=re.MULTILINE,
            )

    if content != original_content:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Fixed {filepath} ({changes_made} occurrences)")
        return True
    else:
        print(f"⏭️  Skipped {filepath} (already fixed or no changes)")
        return False


def main():
    print("🔍 Finding files with datetime.utcnow()...")
    files = get_files_with_utcnow()

    print(f"\n📝 Found {len(files)} files to process\n")

    fixed_count = 0
    for filepath in files:
        if fix_file(filepath):
            fixed_count += 1

    print(f"\n✨ Complete! Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
