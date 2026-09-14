# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A sibling repo's merge must publish an omnibase_infra ref, never its own SHA (OMN-18268).

MEASURED, 2026-09-14. The ``.201`` compose dev lane's effects container reported
``org.opencontainers.image.revision=732fd291a98afeac849a4904670b210096649529``
(omnibase_infra) and, in the image's own
``/app/build-provenance.json``, ``per_repo_vcs_provenance.siblings.omnimarket.vcs_ref
= 65c237cdc914a6c7c1e15de308986ec65c24f309`` -- omnimarket#2538, merged 07:53:57Z.
omnimarket#2537 (09:11:34Z) and #2540 (10:09:57Z) had merged since and were NOT on
the lane. Nothing was broken: the omnimarket revision was never CHOSEN. It is
whatever ``origin/dev`` resolved to at the moment an unrelated omnibase_infra merge
fired a rebuild and ``stage_workspace.sh`` staged the siblings
(``DEPLOY_SIBLING_FALLBACK_REF``, OMN-17135).

WHY THE SIBLING SHA CANNOT BE THE ``git_ref``. The deploy agent resets its
build-context clone -- omnibase_infra -- with ``git reset --hard <git_ref>``
(``deploy_agent/executor.py`` ``_git_pull_locked``). An omnimarket merge SHA names
no commit in omnibase_infra, so publishing it would fail every sibling-triggered
job at the git phase, which is the same class of failure OMN-17135 already
measured from the other direction (``ERROR: omnibase_core: cannot resolve ref
'<infra sha>'``). The primary ref and the sibling SHA are two different facts and
this publisher must carry both.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "trigger_rebuild_on_merge.py"
)

INFRA_HEAD = "732fd291a98afeac849a4904670b210096649529"
OMNIMARKET_MERGE_SHA = "4bc2437210939702fa0dc8529362c0a7c231049f"  # omnimarket#2537


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("_trigger_rebuild_sibling", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild_sibling"] = module
    spec.loader.exec_module(module)
    return module


def test_own_repo_publishes_its_own_merge_sha() -> None:
    """The omnibase_infra path is byte-unchanged: git_ref IS the merge SHA."""
    module = _load_module()
    assert (
        module.resolve_publish_ref(
            source_repo="omnibase_infra",
            source_sha=INFRA_HEAD,
            primary_ref="",
        )
        == INFRA_HEAD
    )


def test_sibling_repo_publishes_the_primary_ref_not_its_own_sha() -> None:
    """A sibling merge pins the build-context repo, never itself."""
    module = _load_module()
    resolved = module.resolve_publish_ref(
        source_repo="omnimarket",
        source_sha=OMNIMARKET_MERGE_SHA,
        primary_ref=INFRA_HEAD,
    )
    assert resolved == INFRA_HEAD
    assert resolved != OMNIMARKET_MERGE_SHA


def test_sibling_repo_without_a_primary_ref_is_refused() -> None:
    """Fail closed: no primary ref means no ref the deploy clone can reset to."""
    module = _load_module()
    with pytest.raises(ValueError, match="--primary-ref"):
        module.resolve_publish_ref(
            source_repo="omnimarket",
            source_sha=OMNIMARKET_MERGE_SHA,
            primary_ref="",
        )


def test_own_repo_with_a_primary_ref_is_refused() -> None:
    """Two candidate refs for one slot is ambiguity, not a convenience."""
    module = _load_module()
    with pytest.raises(ValueError, match="omnibase_infra"):
        module.resolve_publish_ref(
            source_repo="omnibase_infra",
            source_sha=INFRA_HEAD,
            primary_ref=INFRA_HEAD,
        )


def test_primary_ref_must_be_a_commit_sha() -> None:
    """A branch ALIAS is the OMN-18122 defect class; refuse it at the producer."""
    module = _load_module()
    with pytest.raises(ValueError, match="SHA"):
        module.resolve_publish_ref(
            source_repo="omnimarket",
            source_sha=OMNIMARKET_MERGE_SHA,
            primary_ref="origin/dev",
        )


def test_github_output_records_the_sibling_repo_and_sha(tmp_path: Path) -> None:
    """The convergence job asserts the SIBLING revision, so it must be told it."""
    module = _load_module()
    out = tmp_path / "gh-output"
    import os

    previous = os.environ.get("GITHUB_OUTPUT")
    os.environ["GITHUB_OUTPUT"] = str(out)
    try:
        module.emit_github_output(
            True,
            INFRA_HEAD,
            "dev",
            source_repo="omnimarket",
            sibling_sha=OMNIMARKET_MERGE_SHA,
        )
    finally:
        if previous is None:
            os.environ.pop("GITHUB_OUTPUT", None)
        else:
            os.environ["GITHUB_OUTPUT"] = previous

    written = out.read_text(encoding="utf-8")
    assert "published=true" in written
    assert f"source_sha={INFRA_HEAD}" in written
    assert "source_repo=omnimarket" in written
    assert f"sibling_sha={OMNIMARKET_MERGE_SHA}" in written


def test_omnimarket_runtime_paths_still_classify_as_runtime() -> None:
    """The canonical classifier already owns omnimarket's runtime paths."""
    module = _load_module()
    classifier = module.load_runtime_path_classifier(
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "runtime_path_classifier.py"
    )
    matched = module.classify_runtime_paths(
        [
            "src/omnimarket/nodes/node_occ_autobind_effect/handlers/handler_autobind.py",
            "docs/plans/whatever.md",
        ],
        classifier,
    )
    assert matched


def test_omnimarket_docs_only_merge_declines() -> None:
    """The positive control for the zero: docs must not rebuild the lane."""
    module = _load_module()
    classifier = module.load_runtime_path_classifier(
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "runtime_path_classifier.py"
    )
    matched = module.classify_runtime_paths(
        ["docs/plans/whatever.md", "README.md", "CHANGELOG.md"],
        classifier,
    )
    assert not matched
    assert not module.should_trigger(matched, [])
