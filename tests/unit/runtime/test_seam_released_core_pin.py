# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14628: seam test over the RELEASED omnibase-core pin.

Why this exists
---------------
``omnibase_infra`` consumed ``omnibase-core`` through a ``[tool.uv.sources]``
git-rev override while the delegation surface was unreleased. A git-rev pin is
not a reproducible published artifact, and nothing mechanically asserted that
the symbols and field *types* infra depends on actually survive the move to the
published release.

This test is that mechanism. It fails on **exists-but-wrong**, which is the
failure mode a plain import smoke test cannot see:

* released 0.46.7 imports ``ModelContractDodItem`` perfectly well, but that
  class has **no** ``execution_scope`` field at all;
* a future core release could keep the field but widen it back to ``str``.

Both are TYPE failures, not ``ImportError``. So this test never lets an import
failure escape: every resolution is soft, and every problem is accumulated into
a typed finding list that is asserted at the end. That keeps the RED signal
attributable to the real defect class instead of collapsing every regression
into a single opaque collection error.

The same test also freezes the ``RuntimeDelegationDispatchPort.dispatch()``
keyword surface that current OmniMarket callers rely on (OMN-14628 / PRs #2312,
#2595) — a cross-repo callable seam with no other mechanical guard.
"""

from __future__ import annotations

import importlib
import inspect
import json
import types
import typing
from pathlib import Path
from typing import Any, Final

import pytest

_FIXTURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "fixtures" / "seams" / "core_release"
)
_SYMBOLS_FIXTURE: Final[Path] = _FIXTURE_DIR / "0.46.9_expected_symbols.json"
_DISPATCH_FIXTURE: Final[Path] = _FIXTURE_DIR / "dispatch_kwargs_frozen.json"


def _load(path: Path) -> dict[str, Any]:
    """Load a seam fixture, failing loudly if the fixture itself is missing."""
    if not path.is_file():
        pytest.fail(f"seam fixture missing: {path}")
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _import_module(name: str, findings: list[str]) -> types.ModuleType | None:
    """Import ``name`` softly, recording a typed finding instead of raising."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - defect path
        findings.append(f"[missing-module] {name}: {exc}")
        return None


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner_annotation, was_optional)`` for ``X | None`` unions."""
    origin = typing.get_origin(annotation)
    if origin is not typing.Union and origin is not types.UnionType:
        return annotation, False
    args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    optional = len(args) != len(typing.get_args(annotation))
    if len(args) == 1:
        return args[0], optional
    return annotation, optional


def _annotation_name(annotation: Any) -> str:
    """Best-effort stable display name for an annotation."""
    return getattr(annotation, "__name__", None) or str(annotation)


def test_seam_released_core_pin_importable_and_typed() -> None:
    """Released core exposes the exact symbols and field TYPES infra consumes.

    Failure taxonomy in the assertion message is deliberate:
    ``[missing-module]`` / ``[missing-symbol]`` / ``[wrong-type]`` /
    ``[missing-field]`` / ``[dispatch-*]``. A RED here must be readable as
    "the seam changed shape", not "something blew up on import".
    """
    fixture = _load(_SYMBOLS_FIXTURE)
    findings: list[str] = []

    # --- 1. released core is importable and reports the pinned version -----
    core = _import_module("omnibase_core", findings)
    expected_version = fixture["core_version"]
    if core is not None:
        actual_version = getattr(core, "__version__", None)
        if actual_version is not None and actual_version != expected_version:
            findings.append(
                f"[version] omnibase_core.__version__={actual_version!r} "
                f"but fixture freezes {expected_version!r}"
            )

    # --- 2. every frozen symbol resolves off its module --------------------
    for module_name, symbols in fixture["modules"].items():
        module = _import_module(module_name, findings)
        if module is None:
            continue
        for symbol in symbols:
            if not hasattr(module, symbol):
                findings.append(f"[missing-symbol] {module_name}.{symbol}")

    # --- 3. THE point of this test: field TYPES, not mere importability ----
    for spec in fixture["typed_fields"]:
        module_name = spec["module"]
        model_name = spec["model"]
        field_name = spec["field"]
        expected_type = spec["expected_type"]

        module = _import_module(module_name, findings)
        if module is None:
            continue
        model = getattr(module, model_name, None)
        if model is None:
            findings.append(f"[missing-symbol] {module_name}.{model_name}")
            continue

        model_fields = getattr(model, "model_fields", None)
        if model_fields is None:
            findings.append(
                f"[wrong-type] {model_name} is not a pydantic model "
                f"(no model_fields); got {type(model)!r}"
            )
            continue

        if field_name not in model_fields:
            # exists-but-wrong: the class imported fine, the FIELD is absent.
            findings.append(
                f"[missing-field] {model_name}.{field_name} is absent from the "
                f"installed core; fixture requires it typed as {expected_type}. "
                f"present fields: {sorted(model_fields)}"
            )
            continue

        annotation = model_fields[field_name].annotation
        inner, was_optional = _unwrap_optional(annotation)
        actual_type = _annotation_name(inner)
        if actual_type != expected_type:
            findings.append(
                f"[wrong-type] {model_name}.{field_name} is typed "
                f"{actual_type!r}; fixture requires {expected_type!r}"
            )
        elif was_optional != bool(spec["optional"]):
            findings.append(
                f"[wrong-type] {model_name}.{field_name} optionality is "
                f"{was_optional}; fixture requires {bool(spec['optional'])}"
            )

    assert not findings, "released core pin seam drifted:\n" + "\n".join(
        f"  - {item}" for item in findings
    )


def test_seam_dispatch_kwargs_frozen() -> None:
    """The cross-repo dispatch keyword surface OmniMarket calls is frozen."""
    fixture = _load(_DISPATCH_FIXTURE)
    findings: list[str] = []

    module = _import_module(fixture["module"], findings)
    if module is None:
        pytest.fail("dispatch seam unresolvable:\n" + "\n".join(findings))

    class_name, method_name = fixture["qualname"].split(".")
    port = getattr(module, class_name, None)
    if port is None:
        pytest.fail(f"[missing-symbol] {fixture['module']}.{class_name}")
    method = getattr(port, method_name, None)
    if method is None:
        pytest.fail(f"[missing-symbol] {fixture['qualname']}")

    signature = inspect.signature(method)
    hints = typing.get_type_hints(method)

    for spec in fixture["required_keyword_only_params"]:
        name = spec["name"]
        parameter = signature.parameters.get(name)
        if parameter is None:
            findings.append(
                f"[dispatch-missing-kwarg] {fixture['qualname']}({name}=...) "
                f"is gone; OmniMarket callers pass it"
            )
            continue
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            findings.append(
                f"[dispatch-wrong-kind] {name} is {parameter.kind.name}; "
                f"callers pass it keyword-only"
            )
        if parameter.default is not spec["default"]:
            findings.append(
                f"[dispatch-wrong-default] {name} defaults to "
                f"{parameter.default!r}; fixture freezes {spec['default']!r}"
            )
        annotation = hints.get(name, parameter.annotation)
        rendered = (
            annotation if isinstance(annotation, str) else _render_union(annotation)
        )
        if rendered != spec["annotation"]:
            findings.append(
                f"[dispatch-wrong-type] {name} is typed {rendered!r}; "
                f"fixture freezes {spec['annotation']!r}"
            )

    assert not findings, "dispatch kwarg seam drifted:\n" + "\n".join(
        f"  - {item}" for item in findings
    )


def _render_union(annotation: Any) -> str:
    """Render an annotation as ``A | B`` source-ish text for comparison."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        parts = [
            "None" if arg is type(None) else _render_union(arg)
            for arg in typing.get_args(annotation)
        ]
        return " | ".join(parts)
    if origin is not None:
        args = ", ".join(_render_union(arg) for arg in typing.get_args(annotation))
        return f"{_annotation_name(origin)}[{args}]"
    return _annotation_name(annotation)
