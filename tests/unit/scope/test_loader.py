# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for overlay scope loader (OMN-9905).

8 cases per DoD:
1. No-overlay passthrough — base scope returned unchanged when no overlay file.
2. Selector match by name — exact hook/skill name applies overlay fields.
3. Selector match by glob — hook_glob pattern applies overlay to matching IDs.
4. Selector match by predicate — predicate selectors do not match (load-time only).
5. Multi-overlay merge order — multiple entries applied in declaration order; last-write wins.
6. Explicit null clears — null in overlay clears (resets to default) the named field.
7. Disabled: true short-circuits — disabled=True in overlay unconditionally suppresses.
8. Missing overlay file is non-error — non-existent path returns base scope.

Design source: kustomize strategic merge
(https://kubectl.docs.kubernetes.io/references/kustomize/kustomization/patchesstrategicmerge/).
"""

from __future__ import annotations

import textwrap
from logging import WARNING
from pathlib import Path

import pytest

from omnibase_core.models.scope import ModelArtifactEnforcement, ModelEnforcementScope
from omnibase_infra.scope.loader import ScopeCache, load_scope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_overlay(tmp_path: Path, content: str) -> Path:
    """Write an overlay YAML string to a temp file and return the path."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(textwrap.dedent(content), encoding="utf-8")
    return overlay


def fresh_cache() -> ScopeCache:
    """Return a new empty ScopeCache for test isolation."""
    return ScopeCache()


# ---------------------------------------------------------------------------
# 1. No-overlay passthrough
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_overlay_passthrough(tmp_path: Path) -> None:
    """When the overlay file is absent, base scope is returned unchanged."""
    non_existent = tmp_path / "does_not_exist.yaml"
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="warn"))
    result = load_scope(
        "pre_tool_use_bash_guard",
        base=base,
        overlay_path=non_existent,
        cache=fresh_cache(),
    )
    assert result == base
    assert result.enforcement.default == "warn"


# ---------------------------------------------------------------------------
# 2. Selector match by name (exact hook/skill name)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_selector_match_by_name_hook(tmp_path: Path) -> None:
    """Exact hook name selector applies overlay fields to the matching artifact."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: pre_tool_use_bash_guard
            set:
              enforcement:
                default: warn
        """,
    )
    result = load_scope(
        "pre_tool_use_bash_guard",
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.enforcement.default == "warn"


@pytest.mark.unit
def test_selector_match_by_name_skill(tmp_path: Path) -> None:
    """Exact skill name selector applies overlay fields to the matching artifact."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              skill: dod_verify
            set:
              enforcement:
                default: observe
        """,
    )
    result = load_scope(
        "dod_verify",
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.enforcement.default == "observe"


@pytest.mark.unit
def test_selector_name_no_match(tmp_path: Path) -> None:
    """A named selector for a different artifact does not affect the queried artifact."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: some_other_hook
            set:
              enforcement:
                default: observe
        """,
    )
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="block"))
    result = load_scope(
        "pre_tool_use_bash_guard",
        base=base,
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.enforcement.default == "block"


# ---------------------------------------------------------------------------
# 3. Selector match by glob
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_selector_match_by_glob(tmp_path: Path) -> None:
    """hook_glob pattern applies overlay to all matching artifact IDs."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook_glob: "pre_tool_use_*"
            set:
              enforcement:
                default: warn
        """,
    )
    result_guard = load_scope(
        "pre_tool_use_bash_guard",
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    result_dispatch = load_scope(
        "pre_tool_use_dispatch_guard",
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result_guard.enforcement.default == "warn"
    assert result_dispatch.enforcement.default == "warn"


@pytest.mark.unit
def test_selector_glob_no_match(tmp_path: Path) -> None:
    """hook_glob does not match artifacts outside the pattern."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook_glob: "pre_tool_use_*"
            set:
              enforcement:
                default: observe
        """,
    )
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="block"))
    result = load_scope(
        "post_tool_use_bash_guard",
        base=base,
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.enforcement.default == "block"


# ---------------------------------------------------------------------------
# 4. Selector match by predicate (non-matching at load time)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_selector_predicate_does_not_match_at_load_time(tmp_path: Path) -> None:
    """Predicate selectors do not match at plugin-load time (no runtime metadata).

    The predicate selector is reserved for future runtime evaluation. At load
    time, the loader returns the base scope unchanged when only predicate
    selectors are present.
    """
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              predicate:
                repo_kind: python
            set:
              enforcement:
                default: observe
        """,
    )
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="block"))
    result = load_scope(
        "pre_tool_use_bash_guard",
        base=base,
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.enforcement.default == "block"


# ---------------------------------------------------------------------------
# 5. Multi-overlay merge order (last-write wins per-field)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_overlay_merge_order_last_write_wins(tmp_path: Path) -> None:
    """Multiple matching overlays are applied in declaration order; last wins."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: pre_tool_use_bash_guard
            set:
              enforcement:
                default: warn
          - selector:
              hook_glob: "pre_tool_use_*"
            set:
              enforcement:
                default: observe
        """,
    )
    result = load_scope(
        "pre_tool_use_bash_guard",
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.enforcement.default == "observe"


@pytest.mark.unit
def test_multi_overlay_non_conflicting_fields_accumulate(tmp_path: Path) -> None:
    """Non-conflicting fields from multiple overlays are independently applied."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: my_hook
            set:
              enforcement:
                default: warn
          - selector:
              hook: my_hook
            set:
              enforcement:
                non_matching_scope: warn
        """,
    )
    result = load_scope(
        "my_hook",
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.enforcement.default == "warn"
    assert result.enforcement.non_matching_scope == "warn"


# ---------------------------------------------------------------------------
# 6. Explicit null clears a field (resets to default)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_explicit_null_clears_field(tmp_path: Path) -> None:
    """Explicit null in an overlay clears the field, resetting it to default."""
    overlay_content = "overlays:\n  - selector:\n      hook: my_hook\n    set:\n      enforcement: null\n"
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(overlay_content, encoding="utf-8")

    base = ModelEnforcementScope(
        enforcement=ModelArtifactEnforcement(
            default="fail-fast",
            non_matching_scope="warn",
        )
    )
    result = load_scope(
        "my_hook",
        base=base,
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    default_enforcement = ModelArtifactEnforcement()
    assert result.enforcement.default == default_enforcement.default
    assert (
        result.enforcement.non_matching_scope == default_enforcement.non_matching_scope
    )


# ---------------------------------------------------------------------------
# 7. disabled: true short-circuits absolutely
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_disabled_true_short_circuits(tmp_path: Path) -> None:
    """set: { disabled: true } maps disabled_when to a universal predicate.

    A universal disabled_when (all-empty predicate) suppresses the artifact for
    every invocation, regardless of applies_when or enforcement tier.
    """
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: pre_tool_use_bash_guard
            set:
              disabled: true
        """,
    )
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="block"))
    result = load_scope(
        "pre_tool_use_bash_guard",
        base=base,
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.applicability.disabled_when.is_universal()


@pytest.mark.unit
def test_disabled_true_stops_further_overlay_application(tmp_path: Path) -> None:
    """Once disabled: true is applied, subsequent overlays are not processed.

    The enforcement tier from after the disabled entry must not be applied.
    """
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: my_hook
            set:
              disabled: true
          - selector:
              hook: my_hook
            set:
              enforcement:
                default: observe
        """,
    )
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="block"))
    result = load_scope(
        "my_hook",
        base=base,
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result.applicability.disabled_when.is_universal()
    assert result.enforcement.default == "block"


# ---------------------------------------------------------------------------
# 8. Missing overlay file is non-error
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_overlay_file_is_non_error(tmp_path: Path) -> None:
    """A non-existent overlay file path is silently ignored; base is returned."""
    missing = tmp_path / "no_such_overlay.yaml"
    assert not missing.exists()
    base = ModelEnforcementScope()
    result = load_scope(
        "pre_tool_use_bash_guard",
        base=base,
        overlay_path=missing,
        cache=fresh_cache(),
    )
    assert result == base


@pytest.mark.unit
def test_empty_overlay_file_is_non_error(tmp_path: Path) -> None:
    """An overlay file with no overlay entries returns the base scope."""
    overlay = write_overlay(tmp_path, "overlays: []\n")
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="warn"))
    result = load_scope(
        "my_skill",
        base=base,
        overlay_path=overlay,
        cache=fresh_cache(),
    )
    assert result == base


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cache_returns_same_instance(tmp_path: Path) -> None:
    """Subsequent calls for the same artifact_id return the cached scope."""
    non_existent = tmp_path / "overlay.yaml"
    cache = fresh_cache()
    result1 = load_scope("my_hook", overlay_path=non_existent, cache=cache)
    result2 = load_scope("my_hook", overlay_path=non_existent, cache=cache)
    assert result1 is result2


@pytest.mark.unit
def test_cache_isolates_artifact_ids(tmp_path: Path) -> None:
    """Different artifact_ids have independent cache entries."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: hook_a
            set:
              enforcement:
                default: warn
          - selector:
              hook: hook_b
            set:
              enforcement:
                default: observe
        """,
    )
    cache = fresh_cache()
    result_a = load_scope("hook_a", overlay_path=overlay, cache=cache)
    result_b = load_scope("hook_b", overlay_path=overlay, cache=cache)
    assert result_a.enforcement.default == "warn"
    assert result_b.enforcement.default == "observe"
    assert len(cache) == 2


@pytest.mark.unit
def test_cache_invalidate_single(tmp_path: Path) -> None:
    """Invalidating a single artifact_id removes only that entry from cache."""
    non_existent = tmp_path / "overlay.yaml"
    cache = fresh_cache()
    load_scope("hook_a", overlay_path=non_existent, cache=cache)
    load_scope("hook_b", overlay_path=non_existent, cache=cache)
    cache.invalidate("hook_a")
    assert len(cache) == 1
    assert cache.get("hook_a") is None
    assert cache.get("hook_b") is not None


@pytest.mark.unit
def test_onex_overlay_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ONEX_OVERLAY_PATH env var is used as overlay file path when set."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - selector:
              hook: env_hook
            set:
              enforcement:
                default: warn
        """,
    )
    monkeypatch.setenv("ONEX_OVERLAY_PATH", str(overlay))
    result = load_scope("env_hook", cache=fresh_cache())
    assert result.enforcement.default == "warn"


@pytest.mark.unit
def test_malformed_overlay_yaml_returns_base(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Malformed overlays are ignored without mutating the base scope."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("overlays: [", encoding="utf-8")
    base = ModelEnforcementScope(enforcement=ModelArtifactEnforcement(default="block"))

    with caplog.at_level(WARNING, logger="omnibase_infra.scope.loader"):
        result = load_scope(
            "pre_tool_use_bash_guard",
            base=base,
            overlay_path=overlay,
            cache=fresh_cache(),
        )

    assert result == base
    assert "Failed to parse overlay file" in caplog.text


@pytest.mark.unit
def test_non_dict_overlay_item_is_ignored(tmp_path: Path) -> None:
    """Scalar/list overlay entries are ignored instead of raising."""
    overlay = write_overlay(
        tmp_path,
        """
        overlays:
          - not-a-dict
          - ["also", "not", "a", "dict"]
          - selector:
              hook: my_hook
            set:
              enforcement:
                default: warn
        """,
    )

    result = load_scope(
        "my_hook",
        overlay_path=overlay,
        cache=fresh_cache(),
    )

    assert result.enforcement.default == "warn"
