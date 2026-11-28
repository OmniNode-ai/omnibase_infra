# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the core tag, lock, declared pin and moved consumers agree."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from packaging.requirements import Requirement

from scripts.ci.check_sibling_compatible_range import violations

ROOT = Path(__file__).resolve().parents[2]
TAG = "dev-20261127051535-94ed7507315d"
COMMIT = "94ed7507315d434710d69fc71762430d5e646f47"
OMNIMARKET_LINEAGE_TAG = "dev-20261118215158-8bb5b8be7316"
# The published floor is the release that first ships every core module the infra
# nodes import (OMN-21924: node_architecture_validator); the lock tracks the dev tag.
VERSION = "0.47.62"
# The floor the retained core recording was made against; the recording is not remade.
RECORDED_FLOOR = "0.47.58"
LOCKED_VERSION = "0.47.82.dev3+g94ed7507"
GIT_URL = "https://github.com/OmniNode-ai/omnibase_core.git"


def test_core_tag_alone_range_and_lock_agree() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    uv = manifest["tool"]["uv"]
    assert uv["sources"]["omnibase-core"] == {"git": GIT_URL, "tag": TAG}
    assert [
        item for item in uv["override-dependencies"] if item.startswith("omnibase-core")
    ] == []
    assert [
        item
        for item in manifest["project"]["dependencies"]
        if item.startswith("omnibase-core")
    ] == [f"omnibase-core>={VERSION},<0.48.0"]
    locked = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = [p for p in locked["package"] if p["name"] == "omnibase-core"]
    assert len(packages) == 1
    assert packages[0]["version"] == LOCKED_VERSION
    source = urlsplit(packages[0]["source"]["git"])
    assert source._replace(query="", fragment="").geturl() == GIT_URL
    assert parse_qs(source.query) == {"tag": [TAG]}
    assert source.fragment == COMMIT
    assert COMMIT.startswith(TAG.rsplit("-", 1)[1])


def _pin_and_lock_name_different_releases(
    manifest: dict[str, Any],
    lock: dict[str, Any],
    *,
    release_version: str,
) -> list[str]:
    """Ways the pyproject core tag and the uv.lock core entry name different releases.

    OMN-26227: the tag in ``[tool.uv.sources]`` names a core commit by its suffix; the
    lock records the tag, the commit and the version uv derived from it. A lock left
    at an older core release than the tag (or a tag moved without a re-lock) reads
    here as a mismatch, whatever the constants of this module say.
    """
    tag = manifest["tool"]["uv"]["sources"]["omnibase-core"]["tag"]
    (entry,) = (p for p in lock["package"] if p["name"] == "omnibase-core")
    source = urlsplit(entry["source"]["git"])
    problems: list[str] = []
    locked_tags = parse_qs(source.query).get("tag", [])
    if locked_tags != [tag]:
        problems.append(f"lock tag {locked_tags} != pyproject tag {tag!r}")
    if not source.fragment.startswith(tag.rsplit("-", 1)[1]):
        problems.append(f"lock commit {source.fragment!r} is not the commit of {tag!r}")
    if entry["version"] != release_version:
        problems.append(f"lock version {entry['version']!r} != {release_version!r}")
    return problems


def test_pyproject_core_tag_and_lock_core_release_name_the_same_release() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert (
        _pin_and_lock_name_different_releases(
            manifest, lock, release_version=LOCKED_VERSION
        )
        == []
    )


@pytest.mark.parametrize(
    ("lock_source", "lock_version", "expected"),
    [
        # control: pin and lock agree
        (f"{GIT_URL}?tag={TAG}#{COMMIT}", LOCKED_VERSION, []),
        # lock left at the previous core release (OMN-26227 falsifier)
        (
            f"{GIT_URL}?tag={TAG}#{COMMIT}",
            "0.47.75.dev1+ge9f061dc",
            [f"lock version '0.47.75.dev1+ge9f061dc' != '{LOCKED_VERSION}'"],
        ),
        # tag moved in pyproject without a re-lock
        (
            f"{GIT_URL}?tag=dev-20261118065911-e9f061dc9b4c"
            "#e9f061dc9b4c86595359ba7036623c72faa711a7",
            LOCKED_VERSION,
            [
                "lock tag ['dev-20261118065911-e9f061dc9b4c'] != pyproject tag "
                f"{TAG!r}",
                "lock commit 'e9f061dc9b4c86595359ba7036623c72faa711a7' is not the "
                f"commit of {TAG!r}",
            ],
        ),
    ],
)
def test_a_pin_and_lock_naming_different_core_releases_are_found(
    lock_source: str, lock_version: str, expected: list[str]
) -> None:
    manifest = {"tool": {"uv": {"sources": {"omnibase-core": {"tag": TAG}}}}}
    lock = {
        "package": [
            {
                "name": "omnibase-core",
                "version": lock_version,
                "source": {"git": lock_source},
            }
        ]
    }
    assert (
        _pin_and_lock_name_different_releases(
            manifest, lock, release_version=LOCKED_VERSION
        )
        == expected
    )


def _locked_core_version_outside_range(
    manifest: dict[str, Any], lock: dict[str, Any]
) -> list[str]:
    """Version uv.lock locks for core when the published range does not admit it.

    The dev-merge tag source leaves uv.lock with a git URL and no specifier, so uv
    no longer compares the published range with the resolved version (OMN-19655).
    """
    (published,) = (
        Requirement(item)
        for item in manifest["project"]["dependencies"]
        if item.startswith("omnibase-core")
    )
    (locked,) = (p for p in lock["package"] if p["name"] == "omnibase-core")
    return (
        [] if published.specifier.contains(locked["version"]) else [locked["version"]]
    )


def test_published_core_range_admits_the_locked_core() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert _locked_core_version_outside_range(manifest, lock) == []


@pytest.mark.parametrize("version", ["0.47.57", "0.48.0", "0.49.1"])
def test_a_locked_core_outside_the_published_range_is_found(version: str) -> None:
    manifest = {"project": {"dependencies": ["omnibase-core>=0.47.58,<0.48.0"]}}
    lock = {"package": [{"name": "omnibase-core", "version": version}]}
    assert _locked_core_version_outside_range(manifest, lock) == [version]


def test_moved_done_write_consumers_use_the_pinned_node() -> None:
    from omnibase_core.models.ticket.model_done_write_receipt_request import (
        ModelDoneWriteReceiptRequest,
    )
    from omnibase_infra.handlers.done_write_receipt_guard import (
        resolve_done_write_receipt_gate,
    )
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        HandlerEvidenceAutocloseSweep,
    )

    gate = resolve_done_write_receipt_gate()
    handler = HandlerEvidenceAutocloseSweep(gate=gate)
    core = importlib.import_module(
        "omnibase_core.nodes.node_done_write_receipt_gate_compute."
        "handler_done_write_receipt_gate"
    )
    request = ModelDoneWriteReceiptRequest(ticket_id="OMN-17427", description="")
    assert handler._gate is gate
    assert gate.decide(request) == core.evaluate_done_write_receipt(
        ticket_id=request.ticket_id,
        description=request.description,
        verdict=request.verdict,
    )


@pytest.mark.live_contact(
    "tests/fixtures/dep_provenance_core_lineage_recorded_omn17427.json"
)
def test_recorded_core_validator_response_matches_the_pin(
    recorded_response: dict[str, object],
) -> None:
    """The retained core recording binds the real tag response to this lock."""
    cases = recorded_response["cases"]
    assert isinstance(cases, list)
    real = next(case for case in cases if case["label"] == "real")
    assert real["tag"] == TAG
    assert real["ls_remote"]["exit_code"] == 0
    assert real["ls_remote"]["stdout"] == f"{COMMIT}\trefs/tags/{TAG}\n"
    assert real["core"]["exit_code"] == 0
    recorded = tomllib.loads(real["pyproject_toml"])
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert recorded["tool"]["uv"]["sources"] == manifest["tool"]["uv"]["sources"]
    assert recorded["project"]["dependencies"] == [
        f"omnibase-core>={RECORDED_FLOOR},<0.48.0"
    ]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert _locked_core_version_outside_range(recorded, lock) == []
    assert violations(manifest, "omnibase-core") == []
    absent = next(case for case in cases if case["label"] == "absent")
    assert absent["core"]["exit_code"] == 1
    assert "tag_absent" in absent["core"]["stdout"]


def test_ci_enforces_all_core_rules_and_keeps_cascade_identity_selection() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    step = workflow.split("- name: Verify dev-merge tag lineage (OMN-17427)", 1)[
        1
    ].split("- name:", 1)[0]
    assert "--package omnibase-core" in step
    # Core 0.47.76 no longer ships the lineage node: it lives in omnimarket, run from the
    # dev merge tag that core's own check-dev-tag-lineage hook pins. That tag reads a core
    # that takes its version from git, and the step names no omnibase-core tag.
    assert (
        '--with "omnimarket @ git+${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY_OWNER}/omnimarket.git@'
        f'{OMNIMARKET_LINEAGE_TAG}"'
    ) in step
    assert (
        "python -m omnimarket.nodes.node_dev_tag_lineage_effect.runtime_dev_tag_lineage"
    ) in step
    assert "omnibase_core.nodes.node_dev_tag_lineage_effect" not in step
    assert TAG not in step
    assert "--rules" not in step
    assert "CORE_LINEAGE_RULES" not in step
    cascade = (ROOT / ".github/workflows/dependency-cascade.yml").read_text()
    assert '--rules "$CORE_LINEAGE_RULES" --package "$PKG_HYPHEN"' in cascade
    assert (
        'CORE_LINEAGE_RULES: "tag_absent,tag_suffix_mismatch,tag_off_branch,declared_version_mismatch,lock_mismatch,lineage_unresolved"'
        in cascade
    )
