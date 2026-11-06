# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Safe YAML parsing shared by contract discovery and runtime readers."""

from __future__ import annotations

from pathlib import Path

import yaml

__all__ = ["read_contract_yaml"]


def read_contract_yaml(contract_path: Path) -> object:
    """Read contract YAML safely for discovery and pre-subscription wiring.

    Cold wiring re-reads contract extensions after discovery. Keep those reads
    on the same fast safe parser so they do not restore the pure-Python parse
    delay before the first consumer bind (OMN-18843). I/O and parse errors
    propagate to the caller, which owns the field's refusal semantics.
    """
    with open(contract_path) as f:
        # OMN-18843: cold discovery gates the first consumer bind. Use the
        # safe LibYAML parser when available; the memo only helps later scans.
        # Both loaders reject Python object tags, with a portable fallback.
        loader_class = (
            yaml.CSafeLoader if hasattr(yaml, "CSafeLoader") else yaml.SafeLoader
        )
        loader = loader_class(f)
        try:
            raw = loader.get_single_data()
        finally:
            loader.dispose()

    return raw
