# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Recurrence ratchet for OMN-15137: the bash sibling-clone manifest
(``sibling_clone_manifest.sh``) must stay in exact parity with the Python
pin-check authority (``check_sibling_lock_pins.py``'s
``DEFAULT_PACKAGE_REPO_DIRS``).

OMN-15137's root cause was two independently hardcoded repo lists (one in
ensure_runner_clones.sh, one derived ad hoc in stage_workspace.sh) silently
drifting apart -- omnibase_spi was added to one and never mirrored to the
other. The fix makes ``sibling_clone_manifest.sh`` the single bash-side
source of truth that both scripts source. This test is the cross-language
guardrail: if a future 7th sibling is added to
``DEFAULT_PACKAGE_REPO_DIRS`` (Python) but never mirrored into
``sibling_clone_manifest.sh`` (bash), or vice versa, this test fails CI
immediately instead of failing 3 deploy hops deep on a real runner.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "scripts" / "runtime_build" / "sibling_clone_manifest.sh"
CHECK_PINS = REPO_ROOT / "scripts" / "runtime_build" / "check_sibling_lock_pins.py"


def _bash_array(source: str, name: str) -> list[str]:
    match = re.search(
        rf"^{re.escape(name)}=\(\n(.*?)\n\)",
        source,
        re.DOTALL | re.MULTILINE,
    )
    assert match, f"could not find bash array {name!r} in {MANIFEST}"
    return re.findall(r'"([^"]+)"', match.group(1))


def _python_default_package_repo_dirs() -> dict[str, str]:
    """Read DEFAULT_PACKAGE_REPO_DIRS from check_sibling_lock_pins.py via a
    real subprocess import -- proves the *actual* module attribute, not a
    regex guess at Python source text."""
    result = subprocess.run(
        [
            "python3",
            "-c",
            (
                "import importlib.util, sys, json; "
                f"spec = importlib.util.spec_from_file_location('m', {str(CHECK_PINS)!r}); "
                "m = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(m); "
                "print(json.dumps(m.DEFAULT_PACKAGE_REPO_DIRS))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return json.loads(result.stdout)


def test_manifest_dirs_match_python_pin_check_authority_exactly() -> None:
    source = MANIFEST.read_text(encoding="utf-8")
    manifest_dirs = _bash_array(source, "SIBLING_CLONE_MANIFEST")
    manifest_dist_names = _bash_array(source, "SIBLING_CLONE_MANIFEST_DIST_NAMES")

    assert len(manifest_dirs) == len(manifest_dist_names), (
        "SIBLING_CLONE_MANIFEST and SIBLING_CLONE_MANIFEST_DIST_NAMES must be "
        "index-aligned and the same length"
    )

    manifest_pairs = dict(zip(manifest_dist_names, manifest_dirs, strict=True))
    python_pairs = _python_default_package_repo_dirs()

    assert manifest_pairs == python_pairs, (
        "sibling_clone_manifest.sh has drifted from "
        "check_sibling_lock_pins.py's DEFAULT_PACKAGE_REPO_DIRS -- "
        f"bash={manifest_pairs!r} python={python_pairs!r}. Update both "
        "together (OMN-15137 recurrence guard)."
    )


def test_manifest_includes_omnibase_spi() -> None:
    """The specific OMN-15137 regression: omnibase_spi must be in the set."""
    source = MANIFEST.read_text(encoding="utf-8")
    manifest_dirs = _bash_array(source, "SIBLING_CLONE_MANIFEST")
    assert "omnibase_spi" in manifest_dirs
