# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Execute-mode integration tests for the disk-gc.sh multi-tag fix (OMN-15804).

Prove the actual bug end-to-end against a stateful fake `docker`:

  - a multi-tag image candidate must be removed via untag-then-remove, NOT via a
    single `docker rmi <id>` (which real docker refuses once >1 repo:tag points
    at the same id — "must be forced - referenced in multiple repositories").
    Before the fix, `disk-gc.sh` called `docker rmi <id>` once and logged
    "kept/failed" — this fake docker reproduces that exact refusal so the old
    code path is provably RED against it.
  - an image with a live (stopped) container attached must be skipped via the
    execution-time `docker ps -a --filter ancestor=...` re-check.
"""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"

# docker CreatedAt format the planner parses.
_OLD_CREATED = (
    "2026-01-01 00:00:00 +0000 UTC"  # far enough in the past for any min_age_days
)


def _write_fake_docker(bin_dir: Path, state_file: Path, multi_tag_id: str) -> None:
    """A stateful fake `docker` that reproduces the real multi-tag refusal.

    State (JSON on disk, mutated across invocations within one script run):
      tags: [repo:tag, ...] still pointing at multi_tag_id
      ancestor_hit: bool — if true, `ps -a --filter ancestor=...` reports a
                    container so protect_running / fresh-recheck must skip it.
    """
    fake = bin_dir / "docker"
    fake.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json, sys, os

            STATE = {str(state_file)!r}
            MULTI_ID = {multi_tag_id!r}

            def load():
                with open(STATE) as f:
                    return json.load(f)

            def save(s):
                with open(STATE, "w") as f:
                    json.dump(s, f)

            args = sys.argv[1:]
            s = load()

            def is_id_ref(ref):
                return ref == MULTI_ID or ref == MULTI_ID.replace("sha256:", "")[:12]

            if args[:2] == ["image", "ls"]:
                for tag in s["tags"]:
                    repo, _, t = tag.partition(":")
                    print(json.dumps({{
                        "ID": MULTI_ID, "Repository": repo, "Tag": t,
                        "CreatedAt": {_OLD_CREATED!r},
                    }}))
                sys.exit(0)

            # The fresh execution-time re-check (docker ps -a --filter
            # ancestor equals id) is DELIBERATELY the only branch that
            # reports the container. The plan-time inventory snapshot
            # (docker ps --all, below) reports NOTHING for it, simulating a
            # container that started between plan time and execute time --
            # the exact staleness gap this re-check exists to close. If
            # ancestor_hit is only visible here, a passing test proves the
            # EXECUTE-time re-check did the work, not the plan-time
            # protect_running snapshot.
            if args[:2] == ["ps", "-a"] and "--filter" in args:
                fi = args.index("--filter")
                filt = args[fi + 1]
                if filt.startswith("ancestor=") and s.get("ancestor_hit"):
                    tgt = filt.split("=", 1)[1]
                    if is_id_ref(tgt) or tgt in s["tags"]:
                        print("deadbeef0001")
                sys.exit(0)

            if args[:2] == ["ps", "--all"]:
                # Plan-time inventory snapshot: always empty in this fixture.
                sys.exit(0)

            if args[:1] == ["builder"]:
                sys.exit(0)

            if args[:1] == ["rmi"]:
                ref = args[-1]
                if is_id_ref(ref):
                    if s.get("ancestor_hit"):
                        sys.stderr.write("Error: image is being used by a container\\n")
                        sys.exit(1)
                    if len(s["tags"]) > 1:
                        sys.stderr.write(
                            f"Error response from daemon: conflict: unable to delete {{MULTI_ID}} "
                            "(must be forced) - image is referenced in multiple repositories\\n"
                        )
                        sys.exit(1)
                    if s["tags"]:
                        s["tags"] = []
                        save(s)
                        sys.exit(0)
                    sys.stderr.write("Error: No such image\\n")
                    sys.exit(1)
                if ref in s["tags"]:
                    s["tags"].remove(ref)
                    save(s)
                    print(f"Untagged: {{ref}}")
                    sys.exit(0)
                sys.stderr.write(f"Error: No such image: {{ref}}\\n")
                sys.exit(1)

            if args[:2] == ["image", "inspect"]:
                ref = args[-1]
                if is_id_ref(ref) and not s["tags"]:
                    sys.exit(1)  # fully gone, no output needed
                sys.exit(0)

            sys.exit(0)
            """
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_execute(
    tmp_path: Path, multi_tag_id: str, tags: list[str], ancestor_hit: bool
) -> tuple[subprocess.CompletedProcess[str], str]:
    import json

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    state_file = tmp_path / "docker_state.json"
    state_file.write_text(json.dumps({"tags": tags, "ancestor_hit": ancestor_hit}))
    _write_fake_docker(bin_dir, state_file, multi_tag_id)

    keep_list = tmp_path / "keep-list.yaml"
    keep_list.write_text(
        yaml.dump(
            {
                "keep_image_repos": ["myrepo"],
                "keep_image_tags": [],
                "protect_running": True,
                "superseded_image_keep_generations": 0,
                "min_age_days": 3,
            }
        )
    )

    log_dir = tmp_path / "log"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    log_dir.mkdir(exist_ok=True, parents=True)

    proc = subprocess.run(
        [
            "bash",
            str(_SCRIPTS / "disk-gc.sh"),
            "--execute",
            "--keep-list",
            str(keep_list),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    log_file = tmp_path / ".local" / "log" / "onex" / "disk-gc.log"
    log_text = log_file.read_text() if log_file.exists() else ""
    return proc, proc.stderr + log_text


@pytest.mark.unit
class TestDiskGcExecuteMultiTagUntag:
    def test_multi_tag_candidate_removed_via_untag_then_remove(
        self, tmp_path: Path
    ) -> None:
        multi_id = "sha256:" + "d" * 64
        proc, output = _run_execute(
            tmp_path,
            multi_id,
            tags=["myrepo:sha-abc123", "myrepo:pr-999"],
            ancestor_hit=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Untagged: myrepo:sha-abc123" in output
        assert "Untagged: myrepo:pr-999" in output
        assert f"removed image {multi_id}" in output
        assert "kept/failed image" not in output

    def test_in_use_image_skipped_via_fresh_ancestor_recheck(
        self, tmp_path: Path
    ) -> None:
        multi_id = "sha256:" + "e" * 64
        proc, output = _run_execute(
            tmp_path,
            multi_id,
            tags=["myrepo:sha-def456"],
            ancestor_hit=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "fresh in-use re-check" in output
        assert "Untagged:" not in output
