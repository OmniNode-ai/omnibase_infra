# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26715: a node-made deploy-* tag is admitted only in the stability-integration context.

RULING 2026-11-18T15:28:49Z (separation plan S10) admits a node-made ``deploy-*``
lab tag on a declared lab path where a dev-merge tag was required. The declared
path is the ``stability-integration`` build context of ``first_party_sources.yaml``.
Every other context (the dev lane, the pre-PR slot, the open-PR build) still
refuses a deploy tag, and a deploy tag whose twelve hex digits are not the staged
commit is refused everywhere.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER = REPO_ROOT / "src/omnibase_infra/validation/checker_first_party_sources.py"

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fps = _load("checker_first_party_sources_omn26715", CHECKER)
CONTRACT = fps.load_contract()

COMMITS = {
    "omnibase_core": "03e50eb2fade" + "a" * 28,
    "omnibase_spi": "f79fd4dce71e" + "b" * 28,
    "omnimarket": "2b214bdc0c13" + "d" * 28,
    "omnibase_infra": "2e7f72a99bff" + "e" * 28,
}
STAMP = "20261126213000"
DISTRIBUTIONS = {
    "omnibase_core": "omnibase-core",
    "omnibase_spi": "omnibase-spi",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
}


def _tag(repo: str) -> str:
    return f"deploy-{STAMP}-{COMMITS[repo][:12]}"


def _record(repo: str, tag: str) -> dict[str, Any]:
    return {
        "vcs_ref": COMMITS[repo],
        "vcs_dirty": False,
        "vcs_branch": "HEAD",
        "dev_merge_tag": "",
        "deploy_tag": tag,
    }


def _provenance(**tags: str) -> dict[str, Any]:
    siblings = {
        repo: _record(repo, tags.get(repo, _tag(repo)))
        for repo in ("omnibase_core", "omnibase_spi", "omnimarket")
    }
    infra = tags.get("omnibase_infra", _tag("omnibase_infra"))
    return {
        "siblings": siblings,
        "build_context": {"omnibase_infra": _record("omnibase_infra", infra)},
    }


def _site(tmp_path: Path) -> Path:
    site = tmp_path / "site-packages"
    site.mkdir(parents=True)
    for repo, dist in DISTRIBUTIONS.items():
        info = site / f"{dist.replace('-', '_')}-1.0.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0.0\n", encoding="utf-8"
        )
        (info / "direct_url.json").write_text(
            json.dumps({"url": f"file:///workspace/{repo}", "dir_info": {}}),
            encoding="utf-8",
        )
    return site


def _audit(
    tmp_path: Path, provenance: dict[str, Any], context: str | None
) -> dict[str, Any]:
    sources = fps.audit_environment(
        CONTRACT,
        provenance,
        search_paths=[str(_site(tmp_path))],
        required=tuple(DISTRIBUTIONS.values()),
        context=context,
    )
    return {source.repo: source for source in sources}


def test_stability_integration_context_admits_deploy_tags_for_every_package(
    tmp_path: Path,
) -> None:
    found = _audit(tmp_path, _provenance(), "stability-integration")
    assert {repo: s.kind for repo, s in found.items()} == dict.fromkeys(
        COMMITS, "deploy-tag"
    )
    assert fps.violations(found.values()) == []
    assert all(s.accepted_in == "stability-integration" for s in found.values())


def test_every_other_context_still_refuses_a_deploy_tag(tmp_path: Path) -> None:
    for context in (
        None,
        CONTRACT.default_context,
        CONTRACT.pre_pr_slot_context,
        CONTRACT.open_pr_context,
    ):
        found = _audit(tmp_path / str(context), _provenance(), context)
        refused = {s.repo for s in fps.violations(found.values())}
        assert refused == set(COMMITS), context


def test_a_deploy_tag_naming_another_commit_is_refused_in_the_context(
    tmp_path: Path,
) -> None:
    wrong = f"deploy-{STAMP}-{'0' * 12}"
    found = _audit(tmp_path, _provenance(omnibase_core=wrong), "stability-integration")
    assert found["omnibase_core"].kind == "tag-mismatch"
    assert [s.repo for s in fps.violations(found.values())] == ["omnibase_core"]


def test_a_hand_made_deploy_name_is_not_a_deploy_tag(tmp_path: Path) -> None:
    found = _audit(
        tmp_path,
        _provenance(omnimarket="deploy-latest"),
        "stability-integration",
    )
    assert found["omnimarket"].kind != "deploy-tag"
    assert not found["omnimarket"].ok


def test_stamp_records_the_deploy_tag_the_tree_was_checked_out_at(
    tmp_path: Path,
) -> None:
    record = tmp_path / "sibling-vcs-provenance.json"
    record.write_text(
        json.dumps(
            {
                "siblings": {
                    repo: {"vcs_ref": COMMITS[repo], "vcs_dirty": False}
                    for repo in ("omnibase_core", "omnibase_spi", "omnimarket")
                }
            }
        ),
        encoding="utf-8",
    )
    refs = tmp_path / "refs.json"
    refs.write_text(
        json.dumps(
            {
                "repos": {
                    repo: {"ref": _tag(repo)}
                    for repo in ("omnibase_core", "omnibase_spi", "omnimarket")
                }
            }
        ),
        encoding="utf-8",
    )
    tree = tmp_path / "tree"
    tree.mkdir()
    stamped = fps.stamp(record, refs, tree)
    assert stamped["siblings"]["omnibase_core"]["deploy_tag"] == _tag("omnibase_core")
    assert stamped["siblings"]["omnibase_core"]["dev_merge_tag"] == ""
