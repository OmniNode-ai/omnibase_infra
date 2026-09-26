# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Kafka fallback hook scans only filenames supplied by pre-commit."""

import shutil
import subprocess
from pathlib import Path


def test_explicit_clean_file_ignores_unrelated_violation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    source = (
        repo_root / "scripts" / "validation" / "check_kafka_no_hardcoded_fallback.sh"
    )
    script = tmp_path / source.name
    shutil.copy2(source, script)
    clean = tmp_path / "clean.py"
    clean.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text(
        'BROKER = "192.168.1.9:9092"\n',  # kafka-fallback-ok
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(script), str(clean)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
