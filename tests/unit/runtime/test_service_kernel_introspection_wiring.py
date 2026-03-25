# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test: ServiceKernel must wire introspection_service into RuntimeHostProcess.

OMN-6405: When ``introspection_service`` is not passed (or is ``None``),
``RuntimeHostProcess._publish_introspection_with_jitter()`` no-ops on its first
guard, and no node ever publishes contract metadata.  All nodes then appear as
unnamed COMPUTE on the omnidash node-registry page.

This test uses source-level analysis to verify the wiring stays in place,
matching the pattern used by ``test_kernel_no_hardcoded_topics.py``.

Checked invariants:
    1. ``ServiceNodeIntrospection`` is imported in ``service_kernel.py``.
    2. ``RuntimeHostProcess(...)`` is called with ``introspection_service=`` as
       a keyword argument (not ``None`` literal).
    3. ``ServiceNodeIntrospection.from_contract_dir(`` is called to construct
       the service before it is passed.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

SERVICE_KERNEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "service_kernel.py"
)


def _read_source() -> str:
    """Read service_kernel.py source for AST analysis."""
    assert SERVICE_KERNEL_PATH.exists(), (
        f"service_kernel.py not found at {SERVICE_KERNEL_PATH}"
    )
    return SERVICE_KERNEL_PATH.read_text(encoding="utf-8")


class TestIntrospectionServiceWiring:
    """Verify ServiceKernel wires introspection into RuntimeHostProcess (OMN-6405)."""

    def test_service_node_introspection_is_imported(self) -> None:
        """ServiceNodeIntrospection must be imported in service_kernel.py.

        Without the import, the kernel cannot construct the introspection
        service, and RuntimeHostProcess receives ``None``.
        """
        source = _read_source()
        assert "ServiceNodeIntrospection" in source, (
            "service_kernel.py must import ServiceNodeIntrospection "
            "to wire introspection into RuntimeHostProcess (OMN-6405)"
        )

    def test_runtime_host_receives_introspection_service_kwarg(self) -> None:
        """RuntimeHostProcess(...) must include introspection_service= keyword.

        This verifies the keyword argument is present in the constructor call.
        If someone removes it, introspection silently breaks.
        """
        source = _read_source()
        tree = ast.parse(source)

        found_kwarg = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match RuntimeHostProcess(...) call
            func = node.func
            if isinstance(func, ast.Name) and func.id == "RuntimeHostProcess":
                for kw in node.keywords:
                    if kw.arg == "introspection_service":
                        found_kwarg = True
                        # Verify it is NOT None literal
                        if (
                            isinstance(kw.value, ast.Constant)
                            and kw.value.value is None
                        ):
                            pytest.fail(
                                "introspection_service=None is explicitly passed "
                                "to RuntimeHostProcess — this disables all "
                                "introspection publishing (OMN-6405)"
                            )
                        break

        assert found_kwarg, (
            "RuntimeHostProcess() call in service_kernel.py is missing "
            "introspection_service= keyword argument. Without it, "
            "introspection_service defaults to None and all nodes appear "
            "as unnamed COMPUTE on the dashboard (OMN-6405)."
        )

    def test_from_contract_dir_is_called(self) -> None:
        """ServiceNodeIntrospection.from_contract_dir() must be called.

        The factory method reads contract.yaml files to extract node
        descriptions, names, and types. Without it, introspection events
        have empty metadata.
        """
        source = _read_source()
        assert "ServiceNodeIntrospection.from_contract_dir(" in source, (
            "service_kernel.py must call ServiceNodeIntrospection.from_contract_dir() "
            "to construct the introspection service from contract data (OMN-6405). "
            "Without this, published introspection events lack descriptions and types."
        )

    def test_introspection_service_not_unconditionally_none(self) -> None:
        """The introspection_service variable must not be hardcoded to None.

        Specifically, verifies that there is a code path that assigns a
        non-None value to introspection_service before the RuntimeHostProcess
        constructor.
        """
        source = _read_source()
        tree = ast.parse(source)

        # Find all assignments to introspection_service
        assigns_non_none = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "introspection_service"
                    ):
                        # Check if the value is NOT a None constant
                        if not (
                            isinstance(node.value, ast.Constant)
                            and node.value.value is None
                        ):
                            assigns_non_none = True

        assert assigns_non_none, (
            "introspection_service is only ever assigned None in service_kernel.py. "
            "There must be a code path that assigns a ServiceNodeIntrospection "
            "instance (OMN-6405)."
        )
