# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the handler Any signature gate (OMN-10820)."""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.validators.handler_any_signature import main, validate_file


def _write(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


def test_flags_handle_with_any_param(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, input_data: Any) -> None:
        pass
""",
    )
    findings = validate_file(f)
    assert len(findings) == 1
    assert "input_data" in findings[0].detail
    assert findings[0].method_name == "handle"


def test_flags_handle_with_dict_any_param(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return {}
""",
    )
    findings = validate_file(f)
    # Both param and return type flagged
    assert len(findings) >= 1
    assert any("input_data" in fd.detail for fd in findings)


def test_flags_handle_return_type_any(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, data: str) -> Any:
        return data
""",
    )
    findings = validate_file(f)
    assert len(findings) == 1
    assert "return type" in findings[0].detail


def test_flags_init_with_any_injectable(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client
""",
    )
    findings = validate_file(f)
    assert len(findings) == 1
    assert "client" in findings[0].detail


def test_flags_vararg_and_kwarg_any_annotations(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, *args: Any, **kwargs: Any) -> None:
        pass
""",
    )
    findings = validate_file(f)
    assert len(findings) == 2
    assert {finding.detail for finding in findings} == {
        "param 'args: Any'",
        "param 'kwargs: Any'",
    }


def test_allows_concrete_typed_handle(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_good.py",
        """
class ModelInput:
    pass

class ModelOutput:
    pass

class HandlerFoo:
    def handle(self, input_data: ModelInput) -> ModelOutput:
        return ModelOutput()
""",
    )
    assert validate_file(f) == []


def test_substrate_allow_suppresses(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_suppressed.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, input_data: Any) -> None:  # substrate-allow: yaml-load
        pass
""",
    )
    assert validate_file(f) == []


def test_onex_exclude_suppresses(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "handler_excluded.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, input_data: Any) -> None:  # ONEX_EXCLUDE: any_type
        pass
""",
    )
    assert validate_file(f) == []


def test_gate_is_class_name_agnostic(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "service_foo.py",
        """
from typing import Any

class ServiceFoo:
    def handle(self, data: Any) -> None:
        pass
""",
    )
    # Gate checks ANY class with handle/__init__, regardless of class name.
    findings = validate_file(f)
    assert len(findings) == 1  # gate is class-name-agnostic


def test_main_returns_zero_on_clean_dir(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "handler_clean.py",
        """
class HandlerFoo:
    def handle(self, data: str) -> str:
        return data
""",
    )
    assert main([str(tmp_path)]) == 0


def test_main_returns_one_on_violation(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, data: Any) -> None:
        pass
""",
    )
    assert main([str(tmp_path)]) == 1


def test_main_budget_mode_passes_within_budget(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, data: Any) -> None:
        pass
""",
    )
    # 1 violation, budget 5 → should pass
    assert main(["--max-violations", "5", str(tmp_path)]) == 0


def test_main_budget_mode_fails_over_budget(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "handler_bad.py",
        """
from typing import Any

class HandlerFoo:
    def handle(self, data: Any) -> None:
        pass
""",
    )
    # 1 violation, budget 0 → should fail
    assert main(["--max-violations", "0", str(tmp_path)]) == 1
