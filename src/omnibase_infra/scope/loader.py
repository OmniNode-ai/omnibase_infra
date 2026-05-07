# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Overlay loader for ONEX plugin enforcement-scope contracts.

Design source: kustomize strategic merge
(https://kubectl.docs.kubernetes.io/references/kustomize/kustomization/patchesstrategicmerge/).

The loader composes a base ``ModelEnforcementScope`` with zero or more user
overlays from the overlay file. Overlays are applied at plugin-load time; the
resolved scope is cached in-process. Per-fire overhead is a dict lookup only.

Overlay file location (in priority order):

1. ``${ONEX_OVERLAY_PATH}`` — if set, used as the absolute path.
2. ``~/.omninode/overlay.yaml`` — default user overlay location.

Missing overlay file is non-error: the loader returns the base scope unchanged.

Overlay YAML format:

.. code-block:: yaml

    overlays:
      - selector:
          skill: dod_verify
        set:
          enforcement:
            default: warn
      - selector:
          hook: pre_tool_use_bash_guard
        set:
          disabled: true
      - selector:
          hook_glob: "pre_tool_use_*"
        set:
          applicability:
            applies_when:
              cwd_in: [omninode_worktree]
      - selector:
          predicate:
            repo_kind: python
        set:
          enforcement:
            default: observe

Selector types:
    - ``skill: <name>``    — exact match on skill artifact ID.
    - ``hook: <name>``     — exact match on hook artifact ID.
    - ``hook_glob: <pat>`` — fnmatch glob match on artifact ID.
    - ``predicate: {...}`` — key/value predicate match on artifact metadata.

Merge rules:
    - ``set: { disabled: true }`` short-circuits absolutely: the artifact is
      suppressed regardless of scope or enforcement.
    - Field-level merge for ``applicability`` / ``enforcement`` /
      ``unavailable_behavior``: later overlay entry wins per-field.
    - Explicit ``null`` in an overlay clears the base field (resets to default).
    - Multiple selectors matching one artifact: applied in declaration order;
      last-write wins per field.
    - Base contract is read-only; loader returns a resolved copy, never mutates.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import cast

import yaml

from omnibase_infra.scope.models import ModelEnforcementScope
from omnibase_infra.utils import sanitize_error_message, sanitize_secret_path

_DEFAULT_OVERLAY_PATH = Path.home() / ".omninode" / "overlay.yaml"
_ONEX_OVERLAY_PATH_ENV = "ONEX_OVERLAY_PATH"
_logger = logging.getLogger(__name__)
OverlayEntry = dict[str, object]


class ScopeCache:
    """In-process cache of resolved enforcement scopes.

    Scopes are resolved once at plugin-load time and cached here. Per-fire
    overhead is a single dict lookup — no YAML parsing or overlay application
    on the hot path.

    Call ``invalidate()`` to clear the cache (e.g., in tests or after overlay
    file changes during development).
    """

    def __init__(self) -> None:
        self._cache: dict[str, ModelEnforcementScope] = {}

    def get(self, artifact_id: str) -> ModelEnforcementScope | None:
        return self._cache.get(artifact_id)

    def set(self, artifact_id: str, scope: ModelEnforcementScope) -> None:
        self._cache[artifact_id] = scope

    def invalidate(self, artifact_id: str | None = None) -> None:
        """Clear one or all cached scopes."""
        if artifact_id is None:
            self._cache.clear()
        else:
            self._cache.pop(artifact_id, None)

    def __len__(self) -> int:
        return len(self._cache)


_default_cache = ScopeCache()


def load_scope(
    artifact_id: str,
    base: ModelEnforcementScope | None = None,
    *,
    overlay_path: Path | None = None,
    cache: ScopeCache | None = None,
) -> ModelEnforcementScope:
    """Return the resolved enforcement scope for *artifact_id*.

    Reads the base scope (``base`` or a default empty scope), applies any
    matching overlays from the overlay file, caches the result, and returns
    the resolved :class:`~omnibase_core.models.scope.ModelEnforcementScope`.

    Args:
        artifact_id: Hook or skill identifier (e.g. ``"pre_tool_use_bash_guard"``).
        base:        Base scope contract. Defaults to ``ModelEnforcementScope()``
                     (fully permissive with no restrictions).
        overlay_path: Override the overlay file path. Defaults to
                      ``${ONEX_OVERLAY_PATH}`` → ``~/.omninode/overlay.yaml``.
        cache:       Scope cache to use. Defaults to the module-level cache.
                     Pass a fresh ``ScopeCache()`` in tests to isolate state.

    Returns:
        A frozen :class:`~omnibase_core.models.scope.ModelEnforcementScope`
        with all applicable overlays merged in.

    Note:
        The loader runs at plugin-load time, not per-fire. A cache hit on a
        subsequent call for the same *artifact_id* returns the cached scope
        immediately without re-reading the overlay file.
    """
    if cache is None:
        cache = _default_cache

    cached = cache.get(artifact_id)
    if cached is not None:
        return cached

    if base is None:
        base = ModelEnforcementScope()

    overlays = _load_overlay_entries(overlay_path)
    resolved = _apply_overlays(artifact_id, base, overlays)
    cache.set(artifact_id, resolved)
    return resolved


def _resolve_overlay_path(overlay_path: Path | None) -> Path:
    if overlay_path is not None:
        return overlay_path
    env_path = os.environ.get(_ONEX_OVERLAY_PATH_ENV)
    if env_path:
        return Path(env_path)
    return _DEFAULT_OVERLAY_PATH


def _load_overlay_entries(overlay_path: Path | None) -> list[OverlayEntry]:
    """Parse the overlay file and return the list of overlay entries.

    A missing file is non-error — returns an empty list.
    """
    path = _resolve_overlay_path(overlay_path)
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        _logger.warning(
            "Failed to parse overlay file %s: %s",
            sanitize_secret_path(str(path)),
            sanitize_error_message(exc),
        )
        return []

    if not isinstance(data, dict):
        return []
    entries = data.get("overlays", [])
    if not isinstance(entries, list):
        return []
    return cast("list[OverlayEntry]", entries)


def _selector_matches(artifact_id: str, selector: OverlayEntry) -> bool:
    """Return True if *selector* matches *artifact_id*."""
    if "skill" in selector:
        return str(selector["skill"]) == artifact_id
    if "hook" in selector:
        return str(selector["hook"]) == artifact_id
    if "hook_glob" in selector:
        return fnmatch.fnmatch(artifact_id, str(selector["hook_glob"]))
    if "predicate" in selector:
        # Predicate selectors match based on artifact metadata key/value pairs.
        # The loader does not have access to runtime artifact metadata — predicate
        # selectors are evaluated as non-matching at load time unless a metadata
        # registry is provided. This preserves future extensibility without
        # over-engineering the initial implementation.
        return False
    return False


def _merge_dict_field(
    base_value: OverlayEntry | None,
    overlay_value: OverlayEntry | None,
) -> OverlayEntry | None:
    """Shallow-merge dict fields using field-level last-write-wins semantics.

    Overlay values win per top-level key, ``None`` clears the field, and nested
    dicts are replaced instead of being recursively merged.
    """
    if overlay_value is None:
        return None
    if base_value is None:
        return overlay_value
    merged = dict(base_value)
    merged.update(overlay_value)
    return merged


_DISABLED_SENTINEL = "__disabled__"

_UNIVERSAL_DISABLED_WHEN: OverlayEntry = {
    "applies_when": {},
    "disabled_when": {},
}


def _apply_single_overlay(
    scope_dict: OverlayEntry,
    set_fields: OverlayEntry,
) -> OverlayEntry:
    """Apply one overlay's ``set`` block to a scope dict (last-write wins per-field).

    ``disabled: true`` short-circuits: maps to setting ``applicability.disabled_when``
    to a universal predicate (matches all invocations) and marks the dict with a
    sentinel key so the caller can break the overlay loop early.

    Explicit ``null`` in *set_fields* clears the corresponding base field by
    resetting it to its default (empty/None). Complex nested fields (dicts)
    are merged shallowly — the overlay's dict wins per-key.
    """
    if set_fields.get("disabled") is True:
        result = dict(scope_dict)
        result["applicability"] = _UNIVERSAL_DISABLED_WHEN
        result[_DISABLED_SENTINEL] = True
        return result

    result = dict(scope_dict)
    for key, value in set_fields.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict_field(
                cast("OverlayEntry", result[key]),
                cast("OverlayEntry", value),
            )
        else:
            result[key] = value
    return result


def _apply_overlays(
    artifact_id: str,
    base: ModelEnforcementScope,
    overlay_entries: list[OverlayEntry],
) -> ModelEnforcementScope:
    """Apply all matching overlays to *base* in declaration order.

    Returns the base scope unchanged if no overlays match.
    """
    if not overlay_entries:
        return base

    current = cast("OverlayEntry", base.model_dump(mode="json"))

    any_match = False
    for entry in overlay_entries:
        if not isinstance(entry, dict):
            continue
        selector = entry.get("selector", {})
        if not isinstance(selector, dict):
            continue
        selector_map = cast("OverlayEntry", selector)
        if not _selector_matches(artifact_id, selector_map):
            continue

        set_fields = entry.get("set", {})
        if not isinstance(set_fields, dict):
            continue
        set_fields_map = cast("OverlayEntry", set_fields)

        any_match = True
        current = _apply_single_overlay(current, set_fields_map)

        if current.get(_DISABLED_SENTINEL) is True:
            break

    if not any_match:
        return base

    current.pop(_DISABLED_SENTINEL, None)
    return ModelEnforcementScope.model_validate(current)


__all__ = ["ScopeCache", "load_scope"]
