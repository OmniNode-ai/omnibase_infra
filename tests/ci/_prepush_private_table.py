# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Invented deployment rows for pre-push tests; no real machine values."""

from __future__ import annotations

from pathlib import Path

PRIVATE_TABLE_REL = (
    "config/onex/overlays/prepush_hosts/prepush_hosts.omnibase_infra.tsv"
)
HOST_A = "gate-host-a"
HOST_B = "gate-host-b"
HOST_B_CONTAINER = "gate-runner-b"
HOST_C = "gate-host-c"
HOST_D = "gate-host-d"
HOST_E = "gate-host-e"

SYNTHETIC_TABLE = (
    "# Synthetic fifteen-column deployment table\n"
    + "\n".join(
        [
            "h200\tcapacity\tgate-host-a\tgate-host-a.invalid\t24\t/opt/synthetic/bin/uv\t0.11.0\t/srv/synthetic-prepush\tlockdir\t1\t-\tauthorizing\tprefer_remote\tdefault\tSynthetic identity host",
            "h201\tcapacity\tgate-host-b\t192.0.2.21\t32\t/opt/synthetic/bin/uv\t0.11.0\t/srv/synthetic-prepush\tqueue\t3\t-\tauthorizing\tprefer_remote\tlast_resort\tSynthetic queue host",
            "h201c\tidentity\tgate-runner-b\t-\t32\t-\t-\t-\tnone\t1\t-\tauthorizing\tprefer_remote\t-\tSynthetic container",
            "h101\tcapacity\tgate-host-c\t192.0.2.31\t12\t/opt/synthetic/bin/uv\t0.12.7\t/srv/synthetic-prepush\tlockdir\t1\t-\tauthorizing\tallowed\tdefault\tSynthetic host",
            "h105\tcapacity\tgate-host-d\t192.0.2.41\t10\t/opt/synthetic/bin/uv\t0.11.0\t/srv/synthetic-prepush\tlockdir\t1\t-\tauthorizing\tallowed\tdefault\tSynthetic host",
            "h051\tcapacity\tgate-host-e\t198.51.100.51\t48\t/opt/synthetic/bin/uv\t0.11.0\t/srv/synthetic-prepush\tlockdir\t4\t-\tauthorizing\tallowed\tdefault\tSynthetic high-slot host",
        ]
    )
    + "\n"
)


def write_private_root(base: Path, table_text: str = SYNTHETIC_TABLE) -> Path:
    """Create a plain workspace-config root holding the synthetic table."""
    root = base / "workspace_config"
    target = root / PRIVATE_TABLE_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(table_text, encoding="utf-8")
    return root


def write_hook_repo(base: Path, source: Path) -> Path:
    """Commit the current hook in an isolated checkout with a neutral table."""
    import os
    import shutil
    import subprocess

    from omnibase_core.validators.no_unguarded_git_subprocess import (
        scrub_git_location_env,
    )

    repo = base / "hook-project"
    shutil.copytree(
        source / "scripts" / "hooks",
        repo / "scripts" / "hooks",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    ci = repo / "scripts" / "ci"
    ci.mkdir()
    shutil.copy2(source / "scripts" / "ci" / "resolve_modern_bash.sh", ci)
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "Synthetic hook",
        ],
    ):
        subprocess.run(
            ["git", *args],
            cwd=repo,
            env=scrub_git_location_env(os.environ),
            check=True,
            capture_output=True,
        )
    return repo
