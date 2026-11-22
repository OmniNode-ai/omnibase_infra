# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Safe YAML parsing shared by contract discovery and runtime readers."""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.contracts.contract_read_cache import ContractReadCache

__all__ = ["read_contract_yaml"]


def read_contract_yaml(contract_path: Path) -> object:
    """Read contract YAML once per file change, as a private copy (OMN-26359).

    A file is parsed on its first read and again only when its path, size or
    modification time changes (:class:`ContractReadCache`); every read returns a
    copy the caller may mutate.
    """
    return ContractReadCache.read_parsed(contract_path, _parse_contract_yaml)


def _parse_contract_yaml(contract_path: Path) -> object:
    """Read contract YAML safely for discovery and pre-subscription wiring.

    Cold wiring re-reads contract extensions after discovery. Keep those reads
    on the same fast safe parser so they do not restore the pure-Python parse
    delay before the first consumer bind (OMN-18843). I/O and parse errors
    propagate to the caller, which owns the field's refusal semantics.
    """
    with open(contract_path, encoding="utf-8") as f:
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
