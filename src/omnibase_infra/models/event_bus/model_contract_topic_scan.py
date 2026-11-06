# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A stored contract-topic scan, read once per request and contract tree (OMN-22607).

``ContractTopicExtractor.extract_all`` parses every ``contract.yaml`` under the
contracts root and every installed package's contracts. ``TopicProvisioner`` and
``TopicStartupValidator`` each ran it on construction, so one runtime start parsed
the same contracts once per construction (the dev auto-wire test counted eleven
scans in one test; a kafka start builds a validator, an auto-create provisioner,
the kernel provisioner, the migration-executor provisioner and the Pattern B
provisioner).

:meth:`ModelContractTopicScan.read` keys each scan by its request and by a
fingerprint of the files the scan reads (path, size and modification time under the
contracts root and the skill-manifest roots) and reuses the stored scan while that
fingerprint is unchanged. Installed packages do not change inside a process, so a
runtime start reads them once. A contract written or edited after the start changes
the fingerprint, so the next reader scans again and sees it; no caller has to know
when a start began or whether a contract is new.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from omnibase_infra.tools.contract_topic_extractor import (
    ContractTopicExtractor,
    ModelContractTopicEntry,
)

_MAX_STORED = 16


def _fingerprint(roots: Sequence[Path | None]) -> int:
    """Hash of (path, size, mtime) for every source file under the given roots.

    Bytecode caches are left out: importing a module writes them, and that is not a
    change to anything a scan reads.
    """
    seen = (
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for root in roots
        if root is not None and root.is_dir()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    return hash(tuple(seen))


class ModelContractTopicScan(BaseModel):
    """The topic entries one scan found, and the request that found them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: tuple[str, ...]
    fingerprint: int
    entries: tuple[ModelContractTopicEntry, ...]

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _stored: ClassVar[dict[tuple[object, ...], ModelContractTopicScan]] = {}
    _runs: ClassVar[list[int]] = [0]

    @classmethod
    def read(
        cls,
        *,
        contracts_root: Path,
        skill_manifests_root: Path | None = None,
        skill_manifests_roots: Sequence[Path] | None = None,
        include_installed_packages: bool = True,
    ) -> list[ModelContractTopicEntry]:
        """Return the entries for this request, scanning only when it is new.

        The result is a copy, so a caller that filters it never changes what the
        next reader gets.
        """
        roots = [contracts_root, skill_manifests_root, *(skill_manifests_roots or ())]
        request = (*map(str, roots), str(include_installed_packages))
        key = (*request, _fingerprint(roots))
        with cls._lock:
            if key not in cls._stored:
                cls._runs[0] += 1
                extractor = ContractTopicExtractor(
                    include_installed_packages=include_installed_packages
                )
                found = extractor.extract_all(
                    contracts_root=contracts_root,
                    skill_manifests_root=skill_manifests_root,
                    skill_manifests_roots=list(skill_manifests_roots or ()) or None,
                )
                if len(cls._stored) >= _MAX_STORED:
                    cls._stored.clear()
                cls._stored[key] = cls(
                    request=request, fingerprint=hash(key), entries=tuple(found)
                )
            return list(cls._stored[key].entries)

    @classmethod
    def runs(cls) -> int:
        """How many scans have actually run in this process."""
        return cls._runs[0]

    @classmethod
    def clear(cls) -> None:
        """Forget every stored scan (tests that change what a scan reads)."""
        with cls._lock:
            cls._stored.clear()
