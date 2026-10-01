# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The governed-lane compose-config apply refuses without a matched grant (OMN-20260).

Faithful substitution, not mocks: a real git checkout, a real bare
``onex_change_control`` origin whose ``main`` carries the anchor, and a ``docker``
executable on PATH that renders a fixed config and records every ``up`` it is
asked for. Each refusal is paired with the matched-grant control.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from scripts.apply_governed_compose_config import (
    GOVERNED_COMPOSE_LANES,
    canonical_digest,
    main,
)

RENDER = {"name": "omnibase-infra-judge", "services": {"redpanda": {"ports": ["x"]}}}
GRANT_ID = "grant-6f1c2b3a-1d2e-4f50-8a9b-0c1d2e3f4a5b"
NOW = "2026-10-01T12:00:00Z"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.invalid")
    _git(path, "config", "user.name", "t")


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    checkout = tmp_path / "infra"
    _init_repo(checkout)
    (checkout / "docker").mkdir()
    (checkout / "docker" / "docker-compose.judge.yml").write_text("services: {}\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-q", "-m", "c")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    up_log = tmp_path / "up.log"
    fake = bin_dir / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        '  if [ "$a" = config ]; then printf %s "$FAKE_RENDER"; exit 0; fi\n'
        '  if [ "$a" = up ]; then echo "$@" >> "$UP_LOG"; exit 0; fi\n'
        "done\nexit 3\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_RENDER", json.dumps(RENDER))
    monkeypatch.setenv("UP_LOG", str(up_log))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return {"tmp": tmp_path, "checkout": checkout, "up_log": up_log}


def _entry(lab: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "grant_id": GRANT_ID,
        "target_kind": "compose_config",
        "runtime_lane": "judge",
        "compose_project": "omnibase-infra-judge",
        "compose_files": ["docker/docker-compose.judge.yml"],
        "env_files": [],
        "profiles": ["judge"],
        "services": ["redpanda"],
        "compose_ref": _git(lab["checkout"], "rev-parse", "HEAD"),
        "rendered_digest": canonical_digest(json.dumps(RENDER).encode()),
        "requested_by": "jonahgabriel",
        "approved_by": "jake-b-omni",
        "expires_at": "2026-10-02T12:00:00Z",
        "created_at": "2026-10-01T11:00:00Z",
        "reason": "OMN-20260 test",
    }
    entry.update(overrides)
    return entry


def _anchor_file(lab: dict[str, Any], entries: list[dict[str, Any]]) -> Path:
    path: Path = lab["tmp"] / "anchor.yaml"
    path.write_text(yaml.safe_dump({"entries": entries}))
    return path


def _check(
    lab: dict[str, Any], entries: list[dict[str, Any]], lane: str = "judge"
) -> int:
    return main(
        [
            "--repo-root",
            str(lab["checkout"]),
            "apply",
            "--lane",
            lane,
            "--grant-id",
            GRANT_ID,
            "--grants-file",
            str(_anchor_file(lab, entries)),
            "--now",
            NOW,
        ]
    )


def _occ_origin(lab: dict[str, Any], entries: list[dict[str, Any]]) -> Path:
    """A bare origin whose main carries the anchor, and a clone of it."""
    work = lab["tmp"] / "occ-src"
    _init_repo(work)
    (work / "grants").mkdir()
    (work / "grants" / "governed_lane_grants.yaml").write_text(
        yaml.safe_dump({"entries": entries})
    )
    _git(work, "add", ".")
    _git(work, "commit", "-q", "-m", "anchor")
    bare = lab["tmp"] / "occ.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(work), str(bare)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    clone: Path = lab["tmp"] / "occ"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(clone)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return clone


@pytest.mark.unit
def test_governed_lanes_are_pinned() -> None:
    assert frozenset({"stability-test", "judge"}) == GOVERNED_COMPOSE_LANES


@pytest.mark.unit
def test_canonical_digest_ignores_key_order_and_whitespace() -> None:
    a = canonical_digest(b'{"b": 1, "a": [1, 2]}')
    b = canonical_digest(b'{"a":[1,2],"b":1}')
    assert a == b
    assert a != canonical_digest(b'{"a":[2,1],"b":1}')


@pytest.mark.unit
def test_matched_grant_is_allowed(
    lab: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _check(lab, [_entry(lab)]) == 0
    assert "ALLOWED" in capsys.readouterr().out
    assert not lab["up_log"].exists(), "check-only must not recreate anything"


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"grant_id": "grant-00000000-0000-4000-8000-000000000000"},
        {"target_kind": "image"},
        {"runtime_lane": "stability-test"},
        {"compose_project": "omnibase-infra"},
        {"approved_by": "JonahGabriel"},
        {"approved_by": ""},
        {"consumed": True},
        {"expires_at": "2026-10-01T11:59:59Z"},
        {"rendered_digest": "sha256:" + "0" * 64},
        {"compose_ref": "0" * 40},
        {"services": []},
    ],
)
def test_unmatched_grant_is_refused(
    lab: dict[str, Any], overrides: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _check(lab, [_entry(lab, **overrides)]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert not lab["up_log"].exists()


@pytest.mark.unit
def test_empty_anchor_is_refused(lab: dict[str, Any]) -> None:
    assert _check(lab, []) == 1


@pytest.mark.unit
def test_modified_compose_file_is_refused(lab: dict[str, Any]) -> None:
    entry = _entry(lab)
    (lab["checkout"] / "docker" / "docker-compose.judge.yml").write_text(
        "services: {a: {}}\n"
    )
    assert _check(lab, [entry]) == 1


@pytest.mark.unit
def test_changed_render_is_refused(
    lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entry(lab)
    monkeypatch.setenv("FAKE_RENDER", json.dumps({**RENDER, "name": "other"}))
    assert _check(lab, [entry]) == 1


@pytest.mark.unit
def test_execute_never_reads_a_local_anchor(lab: dict[str, Any]) -> None:
    path = _anchor_file(lab, [_entry(lab)])
    argv = ["--repo-root", str(lab["checkout"]), "apply", "--lane", "judge"]
    argv += ["--grant-id", GRANT_ID, "--grants-file", str(path), "--execute"]
    assert main(argv) == 2
    assert not lab["up_log"].exists()


@pytest.mark.unit
def test_execute_reads_main_and_recreates_only_granted_services(
    lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    occ = _occ_origin(lab, [_entry(lab)])
    monkeypatch.setenv("ONEX_DEPLOY_REASON", "OMN-20260 loopback admin bind test")
    argv = ["--repo-root", str(lab["checkout"]), "apply", "--lane", "judge"]
    argv += ["--grant-id", GRANT_ID, "--occ-repo", str(occ), "--now", NOW, "--execute"]
    assert main(argv) == 0
    calls = lab["up_log"].read_text().splitlines()
    assert len(calls) == 1
    assert calls[0].endswith("up -d --no-deps --no-build redpanda")
    assert "-p omnibase-infra-judge" in calls[0]


@pytest.mark.unit
def test_execute_without_a_grant_on_main_recreates_nothing(
    lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    occ = _occ_origin(lab, [])
    monkeypatch.setenv("ONEX_DEPLOY_REASON", "OMN-20260 loopback admin bind test")
    argv = ["--repo-root", str(lab["checkout"]), "apply", "--lane", "judge"]
    argv += ["--grant-id", GRANT_ID, "--occ-repo", str(occ), "--now", NOW, "--execute"]
    assert main(argv) == 1
    assert not lab["up_log"].exists()


@pytest.mark.unit
def test_execute_is_also_gated_by_the_attribution_preflight(
    lab: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    occ = _occ_origin(lab, [_entry(lab)])
    monkeypatch.delenv("ONEX_DEPLOY_REASON", raising=False)
    argv = ["--repo-root", str(lab["checkout"]), "apply", "--lane", "judge"]
    argv += ["--grant-id", GRANT_ID, "--occ-repo", str(occ), "--now", NOW, "--execute"]
    assert main(argv) == 1
    assert not lab["up_log"].exists()
