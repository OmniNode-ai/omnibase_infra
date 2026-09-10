# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The deploy record is per lane [OMN-16729].

Defect, read live on the .201 host 2026-09-08T13:53Z: `~/.omnibase/infra/
registry.json` is ONE host-level file that every lane writes, carrying the lane
identity as a FIELD (`compose_project`) rather than in the filename. It is
therefore last-writer-wins and can never attest more than one lane at a time.
When the dev lane's own refresh was interrogated, that file read
`active_version=0.38.22 git_sha=7f6c2b8dbd89 deployed_at=2026-09-08T12:35:28Z`
with `compose_project=omnibase-infra-stability-test` -- simultaneously stale for
dev AND labelled with a different lane. A reader who checked only the version
and the sha (both of which happened to match) would have concluded it described
the dev refresh, with no signal at all that it did not.

The fix moves the lane identity into the filename --
`registry.<compose_project>.json` -- and keeps the historical `registry.json`
name resolvable as a SYMLINK bound to one lane (dev), refreshed only by a
dev-lane deploy, so the old name means exactly one thing instead of "whichever
lane wrote last".

These tests execute the REAL `write_registry()` and the REAL
`lane_registry_file()` extracted from `scripts/deploy-runtime.sh`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"

DEV_PROJECT = "omnibase-infra"
STABILITY_PROJECT = "omnibase-infra-stability-test"

_LOG_FUNCS = "\n".join(
    [
        "log_step() { printf 'STEP: %s\\n' \"$*\" >&2; }",
        "log_info() { printf 'INFO: %s\\n' \"$*\" >&2; }",
        "log_warn() { printf 'WARN: %s\\n' \"$*\" >&2; }",
        "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
    ]
)


def _extract_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        DEPLOY_SCRIPT.read_text(encoding="utf-8"),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"could not extract {name}() from deploy-runtime.sh"
    return match.group(0)


def _write(
    deploy_root: Path,
    *,
    compose_project: str,
    version: str,
    git_sha: str,
) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        [
            "set -euo pipefail",
            f'DEPLOY_ROOT="{deploy_root}"',
            'REGISTRY_ALIAS_FILE="${DEPLOY_ROOT}/registry.json"',
            f'REGISTRY_ALIAS_COMPOSE_PROJECT="{DEV_PROJECT}"',
            'LANE_ATTRIBUTION_RECORD_JSON=""',
            'COMPOSE_PROFILE="runtime"',
            _LOG_FUNCS,
            _extract_function("lane_registry_file"),
            _extract_function("write_registry"),
            f'REGISTRY_FILE="$(lane_registry_file "{compose_project}")"',
            (
                f'write_registry "{version}" "{git_sha}" '
                f'"{deploy_root}/deployed/{version}" "/fake/repo" "{compose_project}"'
            ),
        ]
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
        timeout=60,
    )


@pytest.mark.unit
def test_two_lanes_do_not_overwrite_each_others_deploy_record(tmp_path: Path) -> None:
    """AC1: the exact 2026-09-08 situation. A stability write followed by a dev
    write must leave BOTH records intact and each labelled with its own lane."""
    deploy_root = tmp_path / "deploy_root"
    deploy_root.mkdir()

    stability = _write(
        deploy_root,
        compose_project=STABILITY_PROJECT,
        version="0.38.22",
        git_sha="7f6c2b8dbd89",
    )
    assert stability.returncode == 0, stability.stderr

    dev = _write(
        deploy_root,
        compose_project=DEV_PROJECT,
        version="0.38.23",
        git_sha="aaaaaaaaaaaa",
    )
    assert dev.returncode == 0, dev.stderr

    stability_record = json.loads(
        (deploy_root / f"registry.{STABILITY_PROJECT}.json").read_text(encoding="utf-8")
    )
    dev_record = json.loads(
        (deploy_root / f"registry.{DEV_PROJECT}.json").read_text(encoding="utf-8")
    )

    assert stability_record["compose_project"] == STABILITY_PROJECT
    assert stability_record["git_sha"] == "7f6c2b8dbd89", (
        "the dev write must not have clobbered the stability lane's record"
    )
    assert dev_record["compose_project"] == DEV_PROJECT
    assert dev_record["git_sha"] == "aaaaaaaaaaaa"


@pytest.mark.unit
def test_alias_is_a_symlink_to_the_dev_lane_record(tmp_path: Path) -> None:
    """AC2: readers that still open `registry.json` by name get the DEV lane's
    record, and get it as a relative symlink so it survives a moved root."""
    deploy_root = tmp_path / "deploy_root"
    deploy_root.mkdir()
    result = _write(
        deploy_root,
        compose_project=DEV_PROJECT,
        version="0.38.23",
        git_sha="bbbbbbbbbbbb",
    )
    assert result.returncode == 0, result.stderr

    alias = deploy_root / "registry.json"
    assert alias.is_symlink(), "the compatibility alias must be a symlink, not a copy"
    assert alias.readlink() == Path(f"registry.{DEV_PROJECT}.json"), (
        "the alias must point at the dev record RELATIVELY"
    )
    assert json.loads(alias.read_text(encoding="utf-8"))["git_sha"] == "bbbbbbbbbbbb"


@pytest.mark.unit
def test_a_non_dev_lane_never_touches_the_alias(tmp_path: Path) -> None:
    """AC3: this is the half that closes the defect. A stability (or prod, or
    judge) deploy must NOT repoint the old name at itself -- that silent
    re-labelling is what made `registry.json` unsound to read."""
    deploy_root = tmp_path / "deploy_root"
    deploy_root.mkdir()
    assert (
        _write(
            deploy_root,
            compose_project=DEV_PROJECT,
            version="0.38.23",
            git_sha="cccccccccccc",
        ).returncode
        == 0
    )
    result = _write(
        deploy_root,
        compose_project=STABILITY_PROJECT,
        version="0.38.22",
        git_sha="dddddddddddd",
    )
    assert result.returncode == 0, result.stderr

    alias = deploy_root / "registry.json"
    assert alias.readlink() == Path(f"registry.{DEV_PROJECT}.json")
    assert json.loads(alias.read_text(encoding="utf-8"))["git_sha"] == "cccccccccccc", (
        "the alias must still describe the dev lane after a stability deploy"
    )
    assert "left untouched" in result.stderr


@pytest.mark.unit
def test_main_repoints_the_registry_file_to_this_lane() -> None:
    """Regression fence on the wiring: main() must resolve REGISTRY_FILE through
    lane_registry_file() once the lane is known, otherwise every read below
    (the active-deployment guard, the prune's active-path exclusion, the
    restore path) still answers for whichever lane wrote last."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    assert 'REGISTRY_FILE="$(lane_registry_file "${compose_project}")"' in text
    main_body = re.search(r"^main\(\)\s*\{.*", text, re.DOTALL | re.MULTILINE)
    assert main_body is not None
    assign_idx = main_body.group(0).find('REGISTRY_FILE="$(lane_registry_file')
    guard_idx = main_body.group(0).find("guard_existing_deployment")
    assert assign_idx != -1 and guard_idx != -1
    assert assign_idx < guard_idx, (
        "REGISTRY_FILE must be repointed BEFORE the first read of it"
    )
