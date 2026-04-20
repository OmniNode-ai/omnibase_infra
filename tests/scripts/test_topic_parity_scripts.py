# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for relocated topic-parity scripts (OMN-9286).

Verifies path resolution and argparse surface for both scripts after relocation
from omni_home/scripts/ to omnibase_infra/scripts/. Full functional coverage
lives at the CI workflow level (.github/workflows/topic-parity.yml in
omni_home) where the scripts run against a real omni_home checkout.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check-topic-parity.py"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync-topic-registry.py"


@pytest.mark.unit
def test_check_topic_parity_has_expected_shebang_and_spdx() -> None:
    content = CHECK_SCRIPT.read_text().splitlines()
    assert content[0] == "#!/usr/bin/env python3"
    assert "SPDX-FileCopyrightText" in content[1]
    assert "SPDX-License-Identifier: MIT" in content[2]


@pytest.mark.unit
def test_sync_topic_registry_has_expected_shebang_and_spdx() -> None:
    content = SYNC_SCRIPT.read_text().splitlines()
    assert content[0] == "#!/usr/bin/env python3"
    assert "SPDX-FileCopyrightText" in content[1]
    assert "SPDX-License-Identifier: MIT" in content[2]


@pytest.mark.unit
def test_check_topic_parity_help() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "--omni-home" in result.stdout
    assert "--registry" in result.stdout


@pytest.mark.unit
def test_sync_topic_registry_help() -> None:
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "--write" in result.stdout
    assert "--dry-run" in result.stdout


@pytest.mark.unit
def test_check_topic_parity_omni_home_env_var_resolution(tmp_path: Path) -> None:
    """--check with a bogus OMNI_HOME fails fast, proving env resolution is wired."""
    bogus_home = tmp_path / "nonexistent-omni-home"
    bogus_home.mkdir()
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--check"],
        env={"OMNI_HOME": str(bogus_home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Registry not found" in result.stderr
    assert str(bogus_home) in result.stderr


@pytest.mark.unit
def test_sync_topic_registry_omni_home_env_var_resolution(tmp_path: Path) -> None:
    """--check with a bogus OMNI_HOME fails fast, proving env resolution is wired."""
    bogus_home = tmp_path / "nonexistent-omni-home"
    bogus_home.mkdir()
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        env={"OMNI_HOME": str(bogus_home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Registry not found" in result.stderr
    assert str(bogus_home) in result.stderr


def _build_fixture_omni_home(root: Path, *, include_arrays: bool) -> Path:
    """Build a minimal omni_home layout with registry + consumer sources."""
    registry_dir = root / "omniclaude" / "src" / "omniclaude" / "hooks"
    registry_dir.mkdir(parents=True)
    (registry_dir / "topic_registry.yaml").write_text(
        "topics:\n"
        "  - topic: onex.evt.omniclaude.foo.v1\n"
        "    event_type: foo.happened\n"
        "    description: Test topic\n"
    )

    server = root / "omnidash" / "server"
    server.mkdir(parents=True)
    shared = root / "omnidash" / "shared"
    shared.mkdir(parents=True)
    (shared / "topics.ts").write_text(
        "export const SUFFIX_OMNICLAUDE_FOO = 'onex.evt.omniclaude.foo.v1';\n"
    )

    if include_arrays:
        (server / "read-model-consumer.ts").write_text(
            "const READ_MODEL_TOPICS: string[] = [SUFFIX_OMNICLAUDE_FOO];\n"
        )
        (server / "event-bus-health-poller.ts").write_text(
            "const EXPECTED_TOPICS: string[] = [SUFFIX_OMNICLAUDE_FOO];\n"
        )
    else:
        (server / "read-model-consumer.ts").write_text("// no subscription array\n")
        (server / "event-bus-health-poller.ts").write_text(
            "const EXPECTED_TOPICS: string[] = [SUFFIX_OMNICLAUDE_FOO];\n"
        )
    return root


@pytest.mark.unit
def test_check_topic_parity_fails_when_required_array_missing(tmp_path: Path) -> None:
    """A renamed/missing top-level subscription array must fail closed (CR #2)."""
    home = _build_fixture_omni_home(tmp_path / "home", include_arrays=False)
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--check"],
        env={"OMNI_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ERROR: Array 'READ_MODEL_TOPICS' not found" in result.stderr
    assert "failing closed" in result.stderr


@pytest.mark.unit
def test_check_topic_parity_rejects_non_mapping_registry(tmp_path: Path) -> None:
    """yaml.safe_load returning non-dict must fail fast (CR #1)."""
    home = tmp_path / "home"
    registry_dir = home / "omniclaude" / "src" / "omniclaude" / "hooks"
    registry_dir.mkdir(parents=True)
    (registry_dir / "topic_registry.yaml").write_text("- just\n- a\n- list\n")

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--check"],
        env={"OMNI_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "root must be a mapping" in result.stderr


@pytest.mark.unit
def test_sync_topic_registry_rejects_non_mapping_registry(tmp_path: Path) -> None:
    """yaml.safe_load returning non-dict must fail fast (CR #4)."""
    home = tmp_path / "home"
    registry_dir = home / "omniclaude" / "src" / "omniclaude" / "hooks"
    registry_dir.mkdir(parents=True)
    (registry_dir / "topic_registry.yaml").write_text("- just\n- a\n- list\n")
    (home / "omnidash" / "shared").mkdir(parents=True)
    (home / "omnidash" / "shared" / "topics.ts").write_text("")

    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        env={"OMNI_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "root must be a mapping" in result.stderr


@pytest.mark.unit
def test_sync_topic_registry_rejects_malformed_topic_entry(tmp_path: Path) -> None:
    """A topics entry missing 'topic' must fail fast, not raise a raw KeyError."""
    home = tmp_path / "home"
    registry_dir = home / "omniclaude" / "src" / "omniclaude" / "hooks"
    registry_dir.mkdir(parents=True)
    (registry_dir / "topic_registry.yaml").write_text(
        "topics:\n  - event_type: foo.bar\n    description: missing topic field\n"
    )
    (home / "omnidash" / "shared").mkdir(parents=True)
    (home / "omnidash" / "shared" / "topics.ts").write_text("")

    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        env={"OMNI_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "missing required 'topic'" in result.stderr


@pytest.mark.unit
def test_sync_topic_registry_escapes_jsdoc_in_descriptions(tmp_path: Path) -> None:
    """Descriptions containing */ or newlines must not break generated JSDoc."""
    home = tmp_path / "home"
    registry_dir = home / "omniclaude" / "src" / "omniclaude" / "hooks"
    registry_dir.mkdir(parents=True)
    (registry_dir / "topic_registry.yaml").write_text(
        "topics:\n"
        "  - topic: onex.evt.omniclaude.foo.v1\n"
        "    event_type: foo.evt\n"
        "    description: 'ends with */ oops and\\nhas a newline'\n"
    )
    shared = home / "omnidash" / "shared"
    shared.mkdir(parents=True)
    (shared / "topics.ts").write_text("")

    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--dry-run"],
        env={"OMNI_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Parse the full JSDoc block(s): "/**" to "*/" on the same line (single-line)
    # or spanning multiple lines. For the description block specifically, there
    # is exactly one, and it must contain the escaped "*\\/" and no raw "*/"
    # inside the payload (the trailing terminator "*/" is excluded by slicing).
    jsdoc_blocks: list[str] = []
    jsdoc_re = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)
    for match in jsdoc_re.finditer(result.stdout):
        jsdoc_blocks.append(match.group(1))

    # Filter to only description blocks — marker blocks like
    # "--- BEGIN GENERATED ... ---" are line comments, not JSDoc, so they
    # never match /** ... */. All matches are description blocks.
    assert jsdoc_blocks, "expected at least one JSDoc description block"
    combined = "\n".join(jsdoc_blocks)
    assert "*\\/" in combined, combined
    assert "*/" not in combined, combined
    # Newlines must be collapsed so the block stays single-line.
    assert "\n" not in combined.strip(), combined


@pytest.mark.unit
def test_check_topic_parity_flags_registry_topic_missing_from_consumer(
    tmp_path: Path,
) -> None:
    """Registry topic not wired into READ_MODEL_TOPICS must fail (CR #3)."""
    home = tmp_path / "home"
    registry_dir = home / "omniclaude" / "src" / "omniclaude" / "hooks"
    registry_dir.mkdir(parents=True)
    (registry_dir / "topic_registry.yaml").write_text(
        "topics:\n"
        "  - topic: onex.evt.omniclaude.covered.v1\n"
        "    event_type: covered.evt\n"
        "  - topic: onex.evt.omniclaude.orphan.v1\n"
        "    event_type: orphan.evt\n"
    )
    server = home / "omnidash" / "server"
    server.mkdir(parents=True)
    shared = home / "omnidash" / "shared"
    shared.mkdir(parents=True)
    (shared / "topics.ts").write_text(
        "export const SUFFIX_OMNICLAUDE_COVERED = 'onex.evt.omniclaude.covered.v1';\n"
    )
    (server / "read-model-consumer.ts").write_text(
        "const READ_MODEL_TOPICS: string[] = [SUFFIX_OMNICLAUDE_COVERED];\n"
    )
    (server / "event-bus-health-poller.ts").write_text(
        "const EXPECTED_TOPICS: string[] = [SUFFIX_OMNICLAUDE_COVERED];\n"
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--check"],
        env={"OMNI_HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NOT subscribed in READ_MODEL_TOPICS" in result.stdout
    assert "onex.evt.omniclaude.orphan.v1" in result.stdout
