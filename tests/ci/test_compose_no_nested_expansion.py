"""CI guard: no nested variable expansion in Docker Compose files.

Nested expansion (e.g. ${OUTER:-prefix-${INNER:-default}}) silently produces
wrong values on Docker Compose < v2.20 and has caused production regressions
(OMNIINTELLIGENCE_DB_URL absent → PluginIntelligence skipped → all intents
classified as Unknown). All DSN/URL values must be set explicitly in
~/.omnibase/.env; compose files must use :? (required) not :- (fallback).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Match ${OUTER:-..${INNER..} — outer fallback containing inner expansion
_NESTED_EXPANSION_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-[^}]*\$\{[A-Za-z_]")

_DOCKER_DIR = Path(__file__).parent.parent.parent / "docker"
_COMPOSE_FILES = list(_DOCKER_DIR.glob("docker-compose*.yml"))

# Guard against silently skipped test when no compose files are found.
# If the docker/ directory is missing or renamed, _COMPOSE_FILES will be
# empty and pytest.mark.parametrize will silently collect zero test cases
# rather than failing.  This assertion catches that at collection time.
assert _COMPOSE_FILES, (
    f"No docker-compose*.yml files found in {_DOCKER_DIR}. "
    "Has the docker/ directory been moved or renamed?"
)


def _find_violations(path: Path) -> list[tuple[int, str]]:
    violations = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if _NESTED_EXPANSION_RE.search(line):
            violations.append((lineno, line.strip()))
    return violations


@pytest.mark.unit
@pytest.mark.parametrize("compose_file", _COMPOSE_FILES, ids=lambda p: p.name)
def test_no_nested_variable_expansion(compose_file: Path) -> None:
    """Fail if any compose file contains nested ${OUTER:-..${INNER..}} expansion.

    All DSN/URL values must be explicit in ~/.omnibase/.env.
    Use ${VAR:?error message} to require explicit configuration.
    """
    violations = _find_violations(compose_file)
    if violations:
        lines = "\n".join(f"  line {n}: {text}" for n, text in violations)
        raise AssertionError(
            f"{compose_file.name}: {len(violations)} nested expansion(s) found.\n"
            f"Set these vars explicitly in ~/.omnibase/.env and use ${{VAR:?}} instead:\n"
            f"{lines}"
        )
