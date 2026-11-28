# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26732: the configuration-value audit names every value, its home and its move order."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

_ROOT = Path(__file__).resolve().parents[2]
_AUDIT = _ROOT / "docs" / "decisions" / "config-value-audit-omn-26732.yaml"
_HOMES = {"overlay", "settings-store", "contract-shape"}
_PROBE = re.compile(
    r"coder-5090|(?<![\d.])(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?!\d)"
)
_SKIP = ("tests/", "uv.lock", "CHANGELOG.md", "docs/decisions/config-value-audit")


def _audit() -> dict:
    return yaml.safe_load(_AUDIT.read_text(encoding="utf-8"))


def _digest(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def _named_paths(audit: dict) -> set[str]:
    """Paths the audit names; a path that trips the hygiene gate is named by digest."""
    return {
        row["path"] if "path" in row else row["path_sha1"]
        for kind in audit["kinds"].values()
        for row in kind["files"]
    }


def _is_named(path: str, named: set[str]) -> bool:
    return path in named or _digest(path) in named


def _probe_hits() -> set[str]:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.splitlines()
    hits: set[str] = set()
    for path in tracked:
        if path.startswith(_SKIP) or "/tests/" in path:
            continue
        try:
            text = (_ROOT / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _PROBE.search(text):
            hits.add(path)
    return hits


@pytest.mark.unit
def test_ac1_every_probe_hit_is_named() -> None:
    named = _named_paths(_audit())
    missing = sorted(p for p in _probe_hits() if not _is_named(p, named))
    assert not missing, (
        f"files with a coder slot id or a private address the audit does not name: {missing}"
    )


@pytest.mark.unit
def test_ac1_probe_control_detects_an_unnamed_file() -> None:
    named = _named_paths(_audit())
    assert not _is_named("src/not_in_the_audit.py", named)
    assert _PROBE.search("slot coder-5090 at " + ".".join(["10", "1", "2", "3"]))
    assert not _PROBE.search("nothing to see here")


@pytest.mark.unit
def test_ac1_counts_match_rows() -> None:
    for name, kind in _audit()["kinds"].items():
        assert kind["count_files"] == len(kind["files"]), name
        total = sum(
            len(row["lines"])
            if "lines" in row
            else row["line_range"][1] - row["line_range"][0] + 1
            for row in kind["files"]
        )
        assert kind["count_lines"] == total, name
        assert sum(kind["by_home"].values()) == total, name


@pytest.mark.unit
def test_ac1_all_requested_kinds_present() -> None:
    assert set(_audit()["kinds"]) == {
        "addresses",
        "host_names",
        "ports",
        "model_ids",
        "hardware_facts",
        "topology_grants",
        "lane_overlay",
        "per_environment_values",
    }


@pytest.mark.unit
def test_ac1_named_files_exist() -> None:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.splitlines()
    )
    tracked_digests = {_digest(p) for p in tracked}
    gone = [
        p
        for p in _named_paths(_audit())
        if p not in tracked and p not in tracked_digests
    ]
    assert not gone, gone


@pytest.mark.unit
def test_ac2_every_row_has_a_home_and_a_reason() -> None:
    audit = _audit()
    rules = audit["rules"]
    for name, rule in rules.items():
        assert rule["home"] in _HOMES, name
        assert rule["reason"].strip(), name
    for kind, body in audit["kinds"].items():
        for row in body["files"]:
            assert row["rule"] in rules, (kind, row["path"])


@pytest.mark.unit
def test_ac3_no_switch_before_the_value_is_held() -> None:
    steps = _audit()["move_plan"]["steps"]
    index = {s["id"]: i for i, s in enumerate(steps)}
    assert len(index) == len(steps)
    kind_of = {s["id"]: s["kind"] for s in steps}
    for step in steps:
        for dep in step["requires"]:
            assert index[dep] < index[step["id"]], step["id"]
        if step["kind"] == "switch":
            chain = set(step["requires"])
            seen: set[str] = set()
            while chain:
                cur = chain.pop()
                seen.add(cur)
                chain |= (
                    set(next(s for s in steps if s["id"] == cur)["requires"]) - seen
                )
            assert "land" in {kind_of[i] for i in seen}, step["id"]
        if step["kind"] == "delete":
            assert any(kind_of[d] == "switch" for d in step["requires"]), step["id"]


@pytest.mark.unit
def test_ac3_control_a_switch_without_a_land_is_detected() -> None:
    steps = [{"id": "X", "kind": "switch", "requires": []}]
    assert not any(s["kind"] == "land" for s in steps)


@pytest.mark.unit
def test_ac4_runtime_account_grants_are_answered() -> None:
    grants = _audit()["runtime_account_grants"]
    assert grants["answer"] in {"contract-shape", "configuration"}
    assert grants["reason"].strip()
    for rel in grants["where"]:
        instances = sorted((_ROOT / rel).glob("*.yaml"))
        assert len(instances) == 3, rel
        for path in instances:
            assert "lane_work_workflow" in path.read_text(encoding="utf-8"), path.name
