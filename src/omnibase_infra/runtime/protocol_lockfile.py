# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol lockfile generator for snapshot testing.

This module generates a deterministic JSON lockfile from the protocol definitions
in ``omnibase_infra.protocols``. The lockfile captures protocol method signatures,
parameter types, and return types so that snapshot tests can detect accidental
breaking changes to protocol interfaces.

The lockfile is stored at ``contracts/runtime/runtime_protocol.lock.json`` and
verified by CI via ``tests/unit/contracts/test_protocol_lockfile.py``.

Architecture:
    Protocol interfaces are the primary contract surface between omnibase_infra
    and its consumers. Changes to protocol method signatures (adding/removing
    parameters, changing types, renaming methods) are breaking changes that must
    be intentional and reviewed.

    The lockfile captures:
    - Protocol class names and their ``runtime_checkable`` status
    - Method names and full signatures (parameter names, types, defaults)
    - Return type annotations
    - Envelope schema version (package version)
    - Handler protocol versions (per-protocol method count as stability metric)

Usage:
    Generate/update the lockfile::

        from omnibase_infra.runtime.protocol_lockfile import generate_lockfile
        lockfile = generate_lockfile()

    Write to disk::

        from omnibase_infra.runtime.protocol_lockfile import write_lockfile
        write_lockfile()

Related:
    - OMN-335: Add Protocol Lockfile Snapshot Tests
    - omnibase_infra.protocols: Protocol definitions
    - tests/unit/contracts/test_protocol_lockfile.py: Snapshot tests

.. versionadded:: 0.11.0
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import omnibase_infra
from omnibase_infra.protocols import __all__ as protocol_names

# Type aliases using object-based types (no Any per ONEX conventions).
# All nested dicts use dict[str, object] to avoid union proliferation.
JsonDict = dict[str, object]

# Path to the lockfile relative to the repo root
LOCKFILE_RELATIVE_PATH = Path("contracts") / "runtime" / "runtime_protocol.lock.json"


def _get_repo_root() -> Path:
    """Find the repository root by walking up from this file."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() and (current / "src").exists():
            return current
        current = current.parent
    msg = "Could not find repository root (no pyproject.toml + src/ found)"
    raise RuntimeError(msg)


def _extract_method_signature(method: object) -> JsonDict:
    """Extract a deterministic signature dict from a method.

    Args:
        method: A function/method object from a Protocol class.

    Returns:
        Dictionary with parameter details and return annotation.
    """
    sig = inspect.signature(method)  # type: ignore[arg-type]
    params: list[dict[str, str | None]] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        param_info: dict[str, str | None] = {
            "name": name,
            "kind": param.kind.name,
        }

        # Capture annotation as string
        if param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
            param_info["annotation"] = ann if isinstance(ann, str) else str(ann)
        else:
            param_info["annotation"] = None

        # Capture default value
        if param.default is not inspect.Parameter.empty:
            param_info["default"] = repr(param.default)
        else:
            param_info["default"] = None

        params.append(param_info)

    # Return annotation
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return_annotation = None
    elif isinstance(ret, str):
        return_annotation = ret
    else:
        return_annotation = str(ret)

    return {
        "parameters": params,
        "return_annotation": return_annotation,
    }


def _extract_protocol_info(protocol_cls: type) -> JsonDict:
    """Extract complete protocol information for the lockfile.

    Args:
        protocol_cls: A Protocol class to inspect.

    Returns:
        Dictionary with protocol metadata and method signatures.
    """
    # Check if the class has the runtime_checkable decorator
    # by checking for _is_runtime_protocol attribute
    is_runtime_checkable: bool = getattr(protocol_cls, "_is_runtime_protocol", False)

    methods: dict[str, JsonDict] = {}
    for name, method in inspect.getmembers(protocol_cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        methods[name] = _extract_method_signature(method)

    return {
        "runtime_checkable": is_runtime_checkable,
        "method_count": len(methods),
        "methods": dict(sorted(methods.items())),
    }


def generate_lockfile() -> JsonDict:
    """Generate the protocol lockfile data structure.

    Inspects all protocols exported from ``omnibase_infra.protocols`` and
    captures their method signatures in a deterministic format.

    Returns:
        Dictionary suitable for JSON serialization as the lockfile.
    """
    import omnibase_infra.protocols as proto_module

    protocols: dict[str, JsonDict] = {}
    for name in sorted(protocol_names):
        cls = getattr(proto_module, name)
        protocols[name] = _extract_protocol_info(cls)

    return {
        "schema_version": "1.0.0",
        "package_version": omnibase_infra.__version__,
        "protocol_count": len(protocols),
        "protocols": protocols,
    }


def write_lockfile(repo_root: Path | None = None) -> Path:
    """Generate and write the lockfile to disk.

    Args:
        repo_root: Repository root directory. If None, auto-detected.

    Returns:
        Path to the written lockfile.
    """
    if repo_root is None:
        repo_root = _get_repo_root()

    lockfile_path = repo_root / LOCKFILE_RELATIVE_PATH
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_lockfile()
    lockfile_path.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return lockfile_path


def load_lockfile(repo_root: Path | None = None) -> JsonDict:
    """Load the existing lockfile from disk.

    Args:
        repo_root: Repository root directory. If None, auto-detected.

    Returns:
        Parsed lockfile data.

    Raises:
        FileNotFoundError: If the lockfile does not exist.
    """
    if repo_root is None:
        repo_root = _get_repo_root()

    lockfile_path = repo_root / LOCKFILE_RELATIVE_PATH
    if not lockfile_path.exists():
        msg = (
            f"Protocol lockfile not found at {lockfile_path}. "
            "Run `write_lockfile()` to generate it."
        )
        raise FileNotFoundError(msg)

    result: JsonDict = json.loads(lockfile_path.read_text(encoding="utf-8"))
    return result
