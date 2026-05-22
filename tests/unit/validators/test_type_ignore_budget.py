# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the type-ignore budget gate (OMN-10822)."""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.validators.type_ignore_budget import main, scan_file


def _write(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


def test_flags_arg_type(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "foo.py",
        "x = foo(bar)  # type: ignore[arg-type]\n",
    )
    findings = scan_file(f)
    assert len(findings) == 1
    assert findings[0].code == "arg-type"
    assert findings[0].line == 1


def test_flags_union_attr(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "foo.py",
        "x = obj.attr  # type: ignore[union-attr]\n",
    )
    findings = scan_file(f)
    assert len(findings) == 1
    assert findings[0].code == "union-attr"


def test_substrate_allow_suppresses(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "foo.py",
        "x = foo(bar)  # type: ignore[arg-type]  # substrate-allow: legacy-api\n",
    )
    assert scan_file(f) == []


def test_onex_exclude_suppresses(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "foo.py",
        "x = foo(bar)  # type: ignore[arg-type]  # ONEX_EXCLUDE: budget\n",
    )
    assert scan_file(f) == []


def test_ignores_other_type_ignore_codes(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "foo.py",
        "x = foo()  # type: ignore[misc]\n",
    )
    assert scan_file(f) == []


def test_ignores_return_value(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "foo.py",
        "x = foo()  # type: ignore[return-value]\n",
    )
    assert scan_file(f) == []


def test_multi_code_bracket_detected(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "foo.py",
        "x = foo(bar)  # type: ignore[arg-type, misc]\n",
    )
    findings = scan_file(f)
    assert len(findings) == 1
    assert findings[0].code == "arg-type"


def test_multiple_lines_counted(tmp_path: Path) -> None:
    src = (
        "a = foo(x)  # type: ignore[arg-type]\n"
        "b = obj.attr  # type: ignore[union-attr]\n"
        "c = bar(y)  # type: ignore[arg-type]\n"
    )
    f = _write(tmp_path, "foo.py", src)
    findings = scan_file(f)
    assert len(findings) == 3


def test_main_passes_within_budget(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "foo.py",
        "x = foo(bar)  # type: ignore[arg-type]\n",
    )
    assert main(["--max-violations", "5", str(tmp_path)]) == 0


def test_main_fails_over_budget(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "foo.py",
        "x = foo(bar)  # type: ignore[arg-type]\n"
        "y = obj.attr  # type: ignore[union-attr]\n",
    )
    assert main(["--max-violations", "1", str(tmp_path)]) == 1


def test_main_passes_at_zero_when_clean(tmp_path: Path) -> None:
    _write(tmp_path, "foo.py", "x = foo()\n")
    assert main(["--max-violations", "0", str(tmp_path)]) == 0


def test_self_exclusion(tmp_path: Path) -> None:
    validators_dir = tmp_path / "validators"
    validators_dir.mkdir()
    f = validators_dir / "type_ignore_budget.py"
    f.write_text("x = foo()  # type: ignore[arg-type]\n", encoding="utf-8")
    assert scan_file(f) == []
