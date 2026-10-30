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


def test_removed_decommissioned_endpoint_rule_is_covered_by_generic_private_address_rules(
    tmp_path: Path,
) -> None:
    """The removed decommissioned-endpoint rule is covered by the generic private-address rules."""
    repo_root = Path(__file__).resolve().parents[4]
    source = repo_root / "scripts/validation/check_kafka_no_hardcoded_fallback.sh"
    script = tmp_path / source.name
    shutil.copy2(source, script)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    host = "192.168." + "86.200"
    planted = [
        (fixtures / f"broker_{port}{extension}", f'BROKER = "{host}:{port}"')
        for port in (9092, 9092 + 10000, 9092 + 20000)
        for extension in (".py", ".sh", ".yaml", ".yml")
    ]
    for path, line in planted:
        path.write_text(line + "\n", encoding="utf-8")
    refused = subprocess.run(
        ["bash", str(script), str(fixtures)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 1, refused.stdout + refused.stderr
    for path, _ in planted:
        assert str(path) in refused.stdout, refused.stdout + refused.stderr

    for path, line in planted:
        path.write_text(line + " # " + "kafka-fallback-ok\n", encoding="utf-8")
    (fixtures / "general.py").write_text(
        planted[0][1] + " # " + "noqa\n", encoding="utf-8"
    )
    (fixtures / "internal.yaml").write_text(
        planted[0][1] + " # " + "onex-allow-internal-ip\n", encoding="utf-8"
    )
    accepted = subprocess.run(
        ["bash", str(script), str(fixtures)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    for path in fixtures.iterdir():
        path.unlink()
    for extension in (".py", ".sh", ".yaml", ".yml"):
        (fixtures / f"docker{extension}").write_text(
            'BROKER = "redpanda:9092"\n', encoding="utf-8"
        )
    negative = subprocess.run(
        ["bash", str(script), str(fixtures)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert negative.returncode == 0, negative.stdout + negative.stderr
