# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node runtime for the prebuilt node manifest: the image build step and the boot step 0 entry (OMN-26361).

Run at image build as
``python -m omnibase_infra.nodes.node_boot_catalog_effect.runtime_prebuilt_manifest build``.
The file is written beside this node's contract so the kernel finds it with no configuration read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from omnibase_infra.handlers import handler_package_announcer as announcer
from omnibase_infra.handlers.handler_prebuilt_manifest import (
    SCHEMA_VERSION,
    prebuilt_manifest_path,
    stamp_of,
)
from omnibase_infra.models.node_announce.model_prebuilt_node_manifest import (
    ModelPrebuiltNodeManifest,
)
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_cache_clear,
)
from omnibase_infra.utils.util_runtime_packages import get_active_runtime_packages


def build_prebuilt_manifest(path: Path, *, workload: str) -> ModelPrebuiltNodeManifest:
    """Run the announcer and discovery once and write the one stamped file."""
    active = get_active_runtime_packages()
    inventory = announcer.read_package_inventory(active)
    announced = announcer.build_catalog_direct(
        inventory, announcer.local_announce_request(active, workload=workload)
    )
    discover_contracts_cache_clear()
    manifest = discover_contracts(announced)
    hashes = tuple(
        (str(r.contract_path), r.content_hash)
        for r in inventory.records
        if r.contract_path is not None and r.content_hash is not None
    )
    prebuilt = ModelPrebuiltNodeManifest(
        schema_version=SCHEMA_VERSION,
        stamp=stamp_of(inventory.fingerprint, hashes, workload),
        announced=announced,
        manifest=manifest,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(prebuilt.model_dump_json(), encoding="utf-8")
    tmp.replace(path)
    return prebuilt


def main(argv: Sequence[str] | None = None) -> int:
    """Image build entry point: ``build`` writes the prebuilt file."""
    parser = argparse.ArgumentParser(prog="runtime_prebuilt_manifest")
    parser.add_argument("operation", choices=["build"])
    parser.add_argument("--path", type=Path, default=prebuilt_manifest_path())
    parser.add_argument("--workload", default="main")
    args = parser.parse_args(argv)
    prebuilt = build_prebuilt_manifest(args.path, workload=args.workload)
    sys.stdout.write(
        f"prebuilt node manifest: {len(prebuilt.announced.announcements)} announced, "
        f"{prebuilt.manifest.total_discovered} discovered, "
        f"{prebuilt.manifest.total_errors} errors -> {args.path}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
