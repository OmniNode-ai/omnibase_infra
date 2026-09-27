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
import shlex
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


# ---------------------------------------------------------------------------
# OMN-19072: no script under scripts/runtime_build/ keeps its own sibling list.
#
# OMN-15137 closed a drift between two independently hardcoded sibling lists,
# but the guard above only compares the manifest with the Python pin authority.
# cut-lab-ref.sh then carried a third literal list with a fourth membership
# (it omitted omnibase_spi), and a comment claiming it mirrored a list it did
# not. Five more scripts in this directory kept literal lists of their own.
# Every named sibling set now lives in sibling_clone_manifest.sh, and the tests
# below keep it that way: a literal array naming a sibling repo anywhere else
# in the directory fails, and each derived set is pinned to the relation its
# manifest comment declares.
# ---------------------------------------------------------------------------

RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
COMPUTE_PROVENANCE = RUNTIME_BUILD / "compute_workspace_provenance.py"

# A bash array assignment: optional declaration keyword(s) and flags, the name,
# `=(` or `+=(`, then the body up to the first `)` that closes a line.
_BASH_ARRAY_ASSIGNMENT = re.compile(
    r"^[ \t]*(?:(?:readonly|local|declare|typeset|export)(?:[ \t]+-[A-Za-z]+)*[ \t]+)*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\+?=\((?P<body>.*?)\)[ \t]*(?:#[^\n]*)?$",
    re.MULTILINE | re.DOTALL,
)


def literal_sibling_arrays(
    source: str, sibling_names: frozenset[str]
) -> list[tuple[str, list[str]]]:
    """Return (array name, sibling tokens) for every literal bash array in
    ``source`` that names a sibling repo, in source order.

    An element spelled as an expansion (``"${SIBLING_CLONE_MANIFEST[@]}"``) is
    not a literal and never matches, so sourcing the manifest is the way out.
    """
    offenders: list[tuple[str, list[str]]] = []
    for match in _BASH_ARRAY_ASSIGNMENT.finditer(source):
        body = match.group("body")
        try:
            tokens = shlex.split(body, comments=True, posix=True)
        except ValueError:
            tokens = body.split()
        named = [token for token in tokens if token in sibling_names]
        if named:
            offenders.append((match.group("name"), named))
    return offenders


def _manifest_array(name: str) -> list[str]:
    """Resolve an array by SOURCING the manifest in bash, so derived sets are
    read as bash computes them, not as a regex guesses them."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; printf "%s\\n" "${' + name + '[@]}"',
            "_",
            str(MANIFEST),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"sourcing {MANIFEST.name} and reading {name} failed: {result.stderr}"
    )
    return result.stdout.split()


def _sibling_names() -> frozenset[str]:
    """Every repo name and uv.lock distribution name any manifest set declares."""
    return frozenset(
        _manifest_array("SIBLING_CLONE_MANIFEST")
        + _manifest_array("SIBLING_CLONE_MANIFEST_DIST_NAMES")
        + _manifest_array("SIBLING_EXTRA_TRACKED_REPOS")
    )


def test_literal_array_detector_positive_control() -> None:
    """The detector must flag a literal list and pass an expansion, or a zero
    from the tree scan below proves nothing (rule 16)."""
    names = frozenset({"omnibase_core", "omnibase_spi", "omnibase-core"})
    literal = (
        "#!/usr/bin/env bash\n"
        "readonly TAG_REPOS=(\n"
        '    "omnibase_infra"  # the build context\n'
        '    "omnibase_core"\n'
        ")\n"
        "ONE_LINE=(omnibase_spi omnimarket)\n"
        "DISTS+=(omnibase-core)\n"
    )
    assert literal_sibling_arrays(literal, names) == [
        ("TAG_REPOS", ["omnibase_core"]),
        ("ONE_LINE", ["omnibase_spi"]),
        ("DISTS", ["omnibase-core"]),
    ]
    derived = (
        'source "${SCRIPT_DIR}/sibling_clone_manifest.sh"\n'
        'readonly TAG_REPOS=("${SIBLING_LAB_TAG_REPOS[@]}")\n'
        "PLAN_ENV=(\n"
        '    "OMNI_HOME=${OMNI_HOME}"\n'
        ")\n"
        "# COMMENTED=(omnibase_core)\n"
    )
    assert literal_sibling_arrays(derived, names) == []


def test_no_runtime_build_script_hardcodes_a_sibling_repo_array() -> None:
    """OMN-19072 AC-1: sibling_clone_manifest.sh is the only file under
    scripts/runtime_build/ that spells a sibling-repo list."""
    names = _sibling_names()
    scripts = sorted(p for p in RUNTIME_BUILD.rglob("*.sh") if p != MANIFEST)
    assert len(scripts) >= 10, f"scan found too few scripts to be real: {scripts}"
    offenders = {
        str(path.relative_to(REPO_ROOT)): found
        for path in scripts
        if (found := literal_sibling_arrays(path.read_text(encoding="utf-8"), names))
    }
    assert offenders == {}, (
        "these scripts keep their own literal sibling-repo array; source "
        "sibling_clone_manifest.sh and use (or add) a named set there instead "
        f"(OMN-19072, the OMN-15137 drift class): {offenders!r}"
    )


def test_lab_tag_set_is_manifest_plus_declared_extras() -> None:
    """OMN-19072 AC-2: the lab tag set covers every repo the pin preflight
    reads, plus the declared tag-only extras, and nothing else."""
    manifest = _manifest_array("SIBLING_CLONE_MANIFEST")
    extras = _manifest_array("SIBLING_EXTRA_TRACKED_REPOS")
    assert extras == ["onex_change_control"]
    assert not set(extras) & set(manifest)
    assert _manifest_array("SIBLING_LAB_TAG_REPOS") == manifest + extras


def test_vendored_set_is_an_ordered_subset_matching_the_provenance_check() -> None:
    """The source-vendored siblings are a subset of the clone manifest, core
    first (OMN-13405), and equal the set the in-image provenance check
    verifies (compute_workspace_provenance.WORKSPACE_PACKAGES), which runs
    inside the image and so cannot source the manifest."""
    manifest = _manifest_array("SIBLING_CLONE_MANIFEST")
    vendored = _manifest_array("SIBLING_VENDORED_REPOS")
    assert vendored == [repo for repo in manifest if repo in vendored]
    assert vendored[0] == "omnibase_core"
    assert "omnibase_infra" not in vendored  # the Docker build context itself
    assert "omnibase_spi" not in vendored  # installed from the wheel

    result = subprocess.run(
        [
            "python3",
            "-c",
            (
                "import importlib.util, json; "
                f"spec = importlib.util.spec_from_file_location('m', {str(COMPUTE_PROVENANCE)!r}); "
                "m = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(m); "
                "print(json.dumps(m.WORKSPACE_PACKAGES))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    workspace_packages: dict[str, str] = json.loads(result.stdout)
    assert list(workspace_packages) == vendored
    dist_by_dir = dict(
        zip(
            _manifest_array("SIBLING_CLONE_MANIFEST"),
            _manifest_array("SIBLING_CLONE_MANIFEST_DIST_NAMES"),
            strict=True,
        )
    )
    assert workspace_packages == {repo: dist_by_dir[repo] for repo in vendored}


def test_lane_refresh_set_is_tag_set_minus_declared_exclusions() -> None:
    """The lane refresh scripts track the tag set less the declared
    exclusions -- the membership they hardcoded before OMN-19072, unchanged."""
    tag_set = _manifest_array("SIBLING_LAB_TAG_REPOS")
    excluded = _manifest_array("SIBLING_LANE_REFRESH_EXCLUDED_REPOS")
    assert excluded == ["omnibase_spi"]
    assert set(excluded) <= set(tag_set)
    assert _manifest_array("SIBLING_LANE_REFRESH_REPOS") == [
        repo for repo in tag_set if repo not in excluded
    ]
