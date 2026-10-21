# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-21256: the scripts reconcile-workspace-venvs.sh sources must load under
macOS ``/bin/bash`` 3.2 with ``set -u``.

Bash before 4.4 treats ``"${A[@]}"`` of an empty array as an unbound variable
under ``set -u``. ``SIBLING_EXTRA_TRACKED_REPOS=()`` was emptied by OMN-17427
and then expanded bare inside ``SIBLING_LAB_TAG_REPOS``, so sourcing the
manifest aborted on every stock Mac, and onboarding stopped at "build the
dispatch venv". Dev Macs and CI run bash 5, where the same line is legal, so
nothing caught it.

CI has no bash 3.2, so the guard here is static: in each sourced script, an
array that is ever assigned empty may only be expanded through the
``${A[@]+"${A[@]}"}`` form, or at a line listed in ``_GUARDED_ELSEWHERE`` with
the reason it cannot be empty there. That list is the AC2 sweep. The runtime
test runs the real ``/bin/bash`` on every platform, including 3.2 on stock
Macs. The static guard covers the empty-array rule on newer bash versions.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECONCILE = REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"
MANIFEST = REPO_ROOT / "scripts" / "runtime_build" / "sibling_clone_manifest.sh"

# Every script reconcile-workspace-venvs.sh sources, by the variable it sources
# through. The sweep test fails if the script starts sourcing anything else.
SOURCED = {
    "_PRIVILEGE_LIB": REPO_ROOT / "scripts" / "reconcile_privilege_lib.sh",
    "_CLONE_MANIFEST_SH": MANIFEST,
}

# (script name, array) -> why the unguarded expansion cannot see an empty array.
_GUARDED_ELSEWHERE = {
    ("reconcile_privilege_lib.sh", "RUN_AS"): (
        "as_owner expands it only in the else branch of `[[ ${#RUN_AS[@]} -eq 0 ]]`"
    ),
}

_EMPTY_ASSIGN = re.compile(r"^\s*(?:declare -a |local -a |local )?([A-Za-z_]\w*)=\(\)")
_EXPANSION = re.compile(r"\$\{([A-Za-z_]\w*)\[@\]\}")
_GUARDED = re.compile(r'\$\{([A-Za-z_]\w*)\[@\]\+"\$\{\1\[@\]\}"\}')


def _unguarded_empty_expansions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    empty = {m.group(1) for line in lines if (m := _EMPTY_ASSIGN.match(line))}
    found = []
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        bare = _GUARDED.sub("", line)
        for name in _EXPANSION.findall(bare):
            if name in empty and (path.name, name) not in _GUARDED_ELSEWHERE:
                found.append(f"{path.name}:{number}: {line.strip()}")
    return found


@pytest.mark.unit
def test_sourced_set_is_the_swept_set() -> None:
    source = RECONCILE.read_text(encoding="utf-8")
    sourced = re.findall(r'^\s*source "\$(\w+)"', source, re.MULTILINE)
    assert sorted(sourced) == sorted(SOURCED), (
        "reconcile-workspace-venvs.sh sources a script this sweep does not "
        f"cover; add it to SOURCED: {sourced!r}"
    )
    for var, path in SOURCED.items():
        assert path.relative_to(REPO_ROOT / "scripts").as_posix() in source, var


@pytest.mark.unit
def test_bash32_no_unguarded_expansion_of_an_empty_array() -> None:
    offenders = [
        o for path in SOURCED.values() for o in _unguarded_empty_expansions(path)
    ]
    assert not offenders, (
        "bash 3.2 (macOS /bin/bash) aborts on these under set -u when the array "
        'is empty; write ${A[@]+"${A[@]}"} instead (OMN-21256): ' + repr(offenders)
    )


@pytest.mark.unit
def test_bash32_detector_positive_control(tmp_path: Path) -> None:
    script = tmp_path / "sample.sh"
    script.write_text(
        'A=()\nB=(x "${A[@]}")\nC=(x ${A[@]+"${A[@]}"})\n# "${A[@]}"\n',
        encoding="utf-8",
    )
    assert _unguarded_empty_expansions(script) == ['sample.sh:2: B=(x "${A[@]}")']


@pytest.mark.unit
def test_bash32_manifest_sources_under_bin_bash_nounset() -> None:
    result = subprocess.run(
        ["/bin/bash", "-euc", 'source "$1"', "_", str(MANIFEST)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
