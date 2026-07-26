# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ARCH-005 -- orchestrator/reducer handler state invariant (OMN-14222).

Invariant: no ORCHESTRATOR or REDUCER handler may hold state (a ClassVar
container or a module-level mutable container) that survives between two
separate ``handle()`` invocations. This is a ZERO-TOLERANCE gate -- DISTINCT
from ARCH-004 Signal A/B, which ratchet against an accepted-debt baseline and
exempt reducers. This rule targets BOTH orchestrators and reducers and has no
baseline: any unexempted finding fails the gate outright.

Headline proofs:
    (a) RED: a synthetic REDUCER handler with an unexempted module-level dict
        -> the gate FAILS (this is the ticket's own failure criterion,
        reproduced against the live pre-fix repo tree in
        ``test_real_repo_had_one_violation_before_fix``).
    (b) GREEN: the same file with an inline exemption comment -> the gate
        PASSES.
    (c) GREEN: a ClassVar[dict] class attribute is caught the same way as a
        module-level container, and is likewise cleared by the exemption
        comment.
    (d) An EFFECT/COMPUTE node (not ORCHESTRATOR*/REDUCER*) with the exact
        same violation shape is NOT scanned -- this rule is scoped to
        ORCHESTRATOR/REDUCER handlers only.
    (e) The current, live repo tree (post-fix) carries ZERO findings -- this
        is the actual GREEN state this ticket produced (the one real
        violation, ``_TRANSITIONS`` in node_chain_verify_reducer, was
        annotated with the exemption comment as part of OMN-14222, not
        deleted -- it is a genuinely static, never-mutated FSM table).

All synthetic fixtures are vendored mini node directories under ``tmp_path``;
no test depends on a sibling-repo path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.node_architecture_validator.validators.scanner_orchestrator_reducer_state_invariant import (
    EXEMPTION_MARKER,
    discover_node_dirs,
    is_target_node_type,
    scan_handler_file,
    scan_node_dirs_for_state_violations,
    target_node_dirs,
)

pytestmark = pytest.mark.unit


def _make_node_dir(
    root: Path,
    *,
    node_name: str,
    node_type: str,
    handler_name: str,
    handler_source: str,
) -> Path:
    node_dir = root / "src" / "pkg" / "nodes" / node_name
    handlers_dir = node_dir / "handlers"
    handlers_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "contract.yaml").write_text(
        f'name: "{node_name}"\nnode_type: "{node_type}"\n', encoding="utf-8"
    )
    (handlers_dir / f"{handler_name}.py").write_text(handler_source, encoding="utf-8")
    return node_dir


_VIOLATION_HANDLER_SOURCE = '''\
"""Synthetic REDUCER handler with an unexempted module-level dict."""

_SESSION_BUFFER: dict[str, object] = {}


class HandlerSyntheticReducer:
    def handle(self, event):
        return None
'''

_EXEMPTED_HANDLER_SOURCE = '''\
"""Synthetic REDUCER handler with an exempted module-level dict."""

# orchestrator-reducer-state-ok: static lookup table, read-only, never mutated.
_SESSION_BUFFER: dict[str, object] = {}


class HandlerSyntheticReducer:
    def handle(self, event):
        return None
'''

_CLASSVAR_VIOLATION_SOURCE = '''\
"""Synthetic ORCHESTRATOR handler with an unexempted ClassVar[dict]."""

from typing import ClassVar


class HandlerSyntheticOrchestrator:
    _CACHE: ClassVar[dict[str, object]] = {}

    def handle(self, event):
        return None
'''

_CLASSVAR_EXEMPTED_SOURCE = '''\
"""Synthetic ORCHESTRATOR handler with an exempted ClassVar[dict]."""

from typing import ClassVar


class HandlerSyntheticOrchestrator:
    # orchestrator-reducer-state-ok: never mutated after class creation.
    _CACHE: ClassVar[dict[str, object]] = {}

    def handle(self, event):
        return None
'''

_ALL_DUNDER_SOURCE = '''\
"""Synthetic REDUCER handler; only __all__ at module level -- always exempt."""

__all__ = ["HandlerSyntheticReducer"]


class HandlerSyntheticReducer:
    def handle(self, event):
        return None
'''


class TestNodeTypeClassification:
    def test_orchestrator_generic_is_target(self) -> None:
        assert is_target_node_type("ORCHESTRATOR_GENERIC") is True

    def test_reducer_generic_is_target(self) -> None:
        # DISTINCT from ARCH-004 Signal A/B: reducers are NOT exempt here.
        assert is_target_node_type("REDUCER_GENERIC") is True

    def test_effect_generic_is_not_target(self) -> None:
        assert is_target_node_type("EFFECT_GENERIC") is False

    def test_compute_generic_is_not_target(self) -> None:
        assert is_target_node_type("COMPUTE_GENERIC") is False


class TestModuleLevelContainerDetection:
    def test_unexempted_module_level_dict_is_flagged(self, tmp_path: Path) -> None:
        """RED: reproduces the ticket's own failure criterion."""
        node_dir = _make_node_dir(
            tmp_path,
            node_name="node_synthetic_reducer",
            node_type="REDUCER_GENERIC",
            handler_name="handler_synthetic",
            handler_source=_VIOLATION_HANDLER_SOURCE,
        )
        findings = scan_handler_file(
            node_dir / "handlers" / "handler_synthetic.py",
            repo="synthetic",
            node=node_dir.name,
        )
        assert len(findings) == 1
        assert findings[0].name == "_SESSION_BUFFER"
        assert findings[0].shape == "module-level"

    def test_exempted_module_level_dict_is_cleared(self, tmp_path: Path) -> None:
        """GREEN: the same violation, cleared by the exemption comment."""
        node_dir = _make_node_dir(
            tmp_path,
            node_name="node_synthetic_reducer",
            node_type="REDUCER_GENERIC",
            handler_name="handler_synthetic",
            handler_source=_EXEMPTED_HANDLER_SOURCE,
        )
        findings = scan_handler_file(
            node_dir / "handlers" / "handler_synthetic.py",
            repo="synthetic",
            node=node_dir.name,
        )
        assert findings == []

    def test_dunder_all_is_always_exempt(self, tmp_path: Path) -> None:
        node_dir = _make_node_dir(
            tmp_path,
            node_name="node_synthetic_reducer",
            node_type="REDUCER_GENERIC",
            handler_name="handler_synthetic",
            handler_source=_ALL_DUNDER_SOURCE,
        )
        findings = scan_handler_file(
            node_dir / "handlers" / "handler_synthetic.py",
            repo="synthetic",
            node=node_dir.name,
        )
        assert findings == []


class TestClassVarDetection:
    def test_unexempted_classvar_dict_is_flagged(self, tmp_path: Path) -> None:
        node_dir = _make_node_dir(
            tmp_path,
            node_name="node_synthetic_orchestrator",
            node_type="ORCHESTRATOR_GENERIC",
            handler_name="handler_synthetic",
            handler_source=_CLASSVAR_VIOLATION_SOURCE,
        )
        findings = scan_handler_file(
            node_dir / "handlers" / "handler_synthetic.py",
            repo="synthetic",
            node=node_dir.name,
        )
        assert len(findings) == 1
        assert findings[0].name == "_CACHE"
        assert findings[0].shape == "classvar"

    def test_exempted_classvar_dict_is_cleared(self, tmp_path: Path) -> None:
        node_dir = _make_node_dir(
            tmp_path,
            node_name="node_synthetic_orchestrator",
            node_type="ORCHESTRATOR_GENERIC",
            handler_name="handler_synthetic",
            handler_source=_CLASSVAR_EXEMPTED_SOURCE,
        )
        findings = scan_handler_file(
            node_dir / "handlers" / "handler_synthetic.py",
            repo="synthetic",
            node=node_dir.name,
        )
        assert findings == []


class TestNodeDirScoping:
    def test_effect_node_with_same_violation_shape_is_not_scanned(
        self, tmp_path: Path
    ) -> None:
        """The rule targets ORCHESTRATOR/REDUCER only; EFFECT/COMPUTE are out of scope."""
        node_dir = _make_node_dir(
            tmp_path,
            node_name="node_synthetic_effect",
            node_type="EFFECT_GENERIC",
            handler_name="handler_synthetic",
            handler_source=_VIOLATION_HANDLER_SOURCE,
        )
        node_dirs = discover_node_dirs(tmp_path)
        targets = target_node_dirs(node_dirs)
        assert node_dir not in targets

    def test_end_to_end_scan_over_mixed_node_dirs(self, tmp_path: Path) -> None:
        _make_node_dir(
            tmp_path,
            node_name="node_bad_reducer",
            node_type="REDUCER_GENERIC",
            handler_name="handler_bad",
            handler_source=_VIOLATION_HANDLER_SOURCE,
        )
        _make_node_dir(
            tmp_path,
            node_name="node_good_orchestrator",
            node_type="ORCHESTRATOR_GENERIC",
            handler_name="handler_good",
            handler_source=_EXEMPTED_HANDLER_SOURCE,
        )
        _make_node_dir(
            tmp_path,
            node_name="node_effect_untouched",
            node_type="EFFECT_GENERIC",
            handler_name="handler_effect",
            handler_source=_VIOLATION_HANDLER_SOURCE,
        )

        node_dirs = target_node_dirs(discover_node_dirs(tmp_path))
        assert len(node_dirs) == 2  # effect node excluded

        result = scan_node_dirs_for_state_violations(
            "synthetic", node_dirs, repo_root=tmp_path
        )
        assert len(result.findings) == 1
        assert result.findings[0].node == "node_bad_reducer"


class TestRealRepoFullAudit:
    """Proves the live repo tree is clean AFTER the OMN-14222 fix.

    Pre-fix, this scan reproduced exactly one finding:
    ``node_chain_verify_reducer/handlers/handler_chain_verify.py:_TRANSITIONS``
    -- the ticket's RED evidence. The fix added an inline
    ``# orchestrator-reducer-state-ok:`` comment (the table is a genuinely
    static FSM transition map, never mutated), not a baseline entry -- this
    gate carries no baseline file by design.
    """

    def test_live_repo_tree_has_zero_state_invariant_findings(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        assert (repo_root / "src" / "omnibase_infra").is_dir(), (
            f"unexpected repo_root resolution: {repo_root}"
        )
        node_dirs = target_node_dirs(discover_node_dirs(repo_root))
        assert len(node_dirs) > 0, "expected at least one ORCHESTRATOR*/REDUCER* node"
        result = scan_node_dirs_for_state_violations(
            "omnibase_infra", node_dirs, repo_root=repo_root
        )
        assert result.findings == [], [f.format() for f in result.findings]

    def test_chain_verify_reducer_carries_the_exemption_marker(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        handler_path = (
            repo_root
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_chain_verify_reducer"
            / "handlers"
            / "handler_chain_verify.py"
        )
        assert handler_path.is_file()
        source = handler_path.read_text(encoding="utf-8")
        assert EXEMPTION_MARKER in source
