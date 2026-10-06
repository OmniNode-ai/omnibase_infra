# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The repository's own pre-push chain remains here; guard tests moved to omnibase_internal."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_prepush_hook_unsets_leaked_git_scoping_vars_before_running_pytest() -> None:
    """Chaining turns the pre-push hook ON for the first time on `.200`, and git
    hands hooks a `GIT_DIR` that overrides `-C`/cwd for every descendant git
    call. Without the unset, every test that builds a throwaway repository under
    `tmp_path` operates on the real worktree instead and errors at setup --
    proven live 2026-07-30: `tests/scripts/test_check_deployed_migration_tree_sync.py`
    is 9 errors with `GIT_DIR` exported and green without it, identically under
    the pre-fix and post-fix guard (so the breakage is the leak, not the guard).
    """
    hook = REPO_ROOT / "scripts" / "hooks" / "prepush_smart_tests.sh"
    text = hook.read_text(encoding="utf-8")
    unset_index = text.find("unset GIT_DIR")
    assert unset_index != -1, (
        f"{hook} must unset the repo-scoping GIT_* variables git exports into "
        "hooks before handing control to pytest"
    )
    for var in ("GIT_WORK_TREE", "GIT_INDEX_FILE"):
        assert var in text[unset_index : unset_index + 200], (
            f"{var} must be unset alongside GIT_DIR"
        )
    first_pytest = text.find("uv run pytest")
    assert first_pytest != -1, "expected a pytest invocation in the pre-push hook"
    assert unset_index < first_pytest, (
        "the unset must precede every pytest invocation, or the leak still applies"
    )
