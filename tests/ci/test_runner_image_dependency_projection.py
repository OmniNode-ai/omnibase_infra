# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner-image identity binds the dependency environment, not file bytes.

OMN-18351. ``shared_env_digest`` / ``identity_digest`` hashed ``pyproject.toml``
as raw bytes, so *any* edit to that file flipped the runner-image identity and
forced every such PR to re-commit ``docker/runners/runner-image.lock.json``.

Two things follow from that, and both were measured on live PRs:

1. The churn is spurious. The runner image bakes ``uv sync --frozen
   --all-extras --all-groups --no-install-project``. Under
   ``--no-install-project`` the project itself is never installed, so its
   version, its entry points and its tool configuration cannot change one byte
   of the image -- yet ``omnibase_infra`` ``#3487`` (one pytest marker line),
   ``#3481`` (one entry-point line) and ``#3449`` (a version bump) each had to
   re-bind the image identity.

2. The value is unknowable on a PR. CI evaluates the lock against the *merge*
   tree (``#3499`` commit ``b87f3ac1`` checked out ``refs/remotes/pull/3499/
   merge`` and failed ``assert 'c69358a5e128497de7077c0c' ==
   '66f25fdfe54991520c21a0a4'``). A merge combines dev's ``pyproject.toml``
   edit with the PR's, producing a manifest neither parent hashed, so whichever
   side of the lock the merge resolves to is wrong by construction. Nothing can
   repair that after the fact either: ``dev`` protection is ``contexts:
   ["CI Summary"]`` with ``enforce_admins: true`` and no bypass actor, so no
   post-merge job can push a corrected lock.

The fix is to bind what the image actually contains. These tests pin that
narrowing in both directions: a non-dependency edit must not move the digest
(AC1), every dependency-defining table must (AC2), and anything unparseable or
structurally unfamiliar must fail closed rather than hash a default (AC3).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
LOCK_FILE = REPO_ROOT / "docker" / "runners" / "runner-image.lock.json"

# A minimal but structurally faithful pyproject: every table the projection is
# required to bind, plus three it must ignore.
BASE_PYPROJECT = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "omnibase-infra"
version = "0.38.24"
requires-python = ">=3.12"
dependencies = [
    "omnibase-core>=0.47.12,<0.48.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[project.entry-points."onex.nodes"]
node_example = "omnibase_infra.nodes.node_example"

[dependency-groups]
test = ["pytest-cov>=5.0.0"]

[tool.uv]
package = true

[tool.uv.sources]
omnibase-core = { index = "pypi" }

[tool.ruff]
line-length = 88

[tool.pytest.ini_options]
markers = [
    "unit: unit tests",
]
"""


def _load_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_CI / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module


@pytest.fixture(scope="module")
def identity() -> Any:
    return _load_module("runner_image_identity")


@pytest.fixture(scope="module")
def env_digest() -> Any:
    return _load_module("ci_env_digest")


@pytest.fixture(scope="module")
def lock_data() -> dict[str, Any]:
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _tree(root: Path, pyproject: str, *, uv_lock: str = "version = 1\n") -> Path:
    """Materialise a tree carrying every digest input, with real repo bytes.

    The three non-manifest shared-env inputs are copied verbatim from the repo
    so the fixture exercises the production input set rather than a stand-in.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (root / "uv.lock").write_text(uv_lock, encoding="utf-8")
    for relative in (
        ".github/actions/setup-python-uv/action.yml",
        "scripts/ci/ci_env_digest.py",
        "scripts/ci/ensure_ci_env.sh",
    ):
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / relative).read_bytes())
    return root


def _digests(identity: Any, lock: dict[str, Any], root: Path) -> tuple[str, str]:
    return (
        identity.compute_shared_env_digest(root, lock),
        identity.compute_identity(root, lock),
    )


# --------------------------------------------------------------------------- #
# AC1 — a non-dependency edit must not move either digest                      #
# --------------------------------------------------------------------------- #

# Each case is a real edit shape taken from a dev commit that was forced to
# re-bind the runner image identity for no reason.
NON_DEPENDENCY_EDITS: tuple[tuple[str, str, str], ...] = (
    (
        "pytest_marker_omnibase_infra_3487",
        '    "unit: unit tests",\n',
        '    "unit: unit tests",\n    "cross_repo_consumer: added by #3487",\n',
    ),
    (
        "entry_point_omnibase_infra_3481",
        'node_example = "omnibase_infra.nodes.node_example"\n',
        'node_example = "omnibase_infra.nodes.node_example"\n'
        'node_added = "omnibase_infra.nodes.node_added"\n',
    ),
    (
        "version_bump_omnibase_infra_3449",
        'version = "0.38.24"',
        'version = "0.38.25"',
    ),
    (
        "ruff_line_length",
        "line-length = 88",
        "line-length = 100",
    ),
    (
        "comment_and_whitespace_only",
        "[tool.ruff]\n",
        "# a passing comment\n\n[tool.ruff]\n",
    ),
)


@pytest.mark.parametrize(
    ("case", "old", "new"),
    NON_DEPENDENCY_EDITS,
    ids=[case for case, _, _ in NON_DEPENDENCY_EDITS],
)
def test_non_dependency_pyproject_edit_does_not_move_the_identity(
    identity: Any,
    lock_data: dict[str, Any],
    tmp_path: Path,
    case: str,
    old: str,
    new: str,
) -> None:
    """AC1: an edit that cannot reach the baked env must not re-bind the image.

    The bake is ``--no-install-project``, so the project's own version, entry
    points and tool configuration are absent from the image. Binding them makes
    every unrelated PR re-commit the lock and chase a moving dev.
    """
    assert old in BASE_PYPROJECT, f"fixture anchor missing for {case}"
    before = _tree(tmp_path / "before", BASE_PYPROJECT)
    after = _tree(tmp_path / "after", BASE_PYPROJECT.replace(old, new, 1))
    assert (before / "pyproject.toml").read_bytes() != (
        after / "pyproject.toml"
    ).read_bytes(), f"{case}: fixture did not actually change pyproject.toml"

    assert _digests(identity, lock_data, before) == _digests(
        identity, lock_data, after
    ), (
        f"{case}: a non-dependency pyproject edit moved the runner-image "
        "identity; it cannot change what the image contains"
    )


# --------------------------------------------------------------------------- #
# AC2 — every dependency-defining table must still move both digests           #
# --------------------------------------------------------------------------- #

DEPENDENCY_EDITS: tuple[tuple[str, str, str], ...] = (
    (
        "project_dependencies",
        '    "omnibase-core>=0.47.12,<0.48.0",\n',
        '    "omnibase-core>=0.47.13,<0.48.0",\n',
    ),
    (
        "project_optional_dependencies",
        'dev = ["pytest>=8.0.0"]',
        'dev = ["pytest>=8.1.0"]',
    ),
    (
        "dependency_groups",
        'test = ["pytest-cov>=5.0.0"]',
        'test = ["pytest-cov>=6.0.0"]',
    ),
    (
        "project_requires_python",
        'requires-python = ">=3.12"',
        'requires-python = ">=3.13"',
    ),
    (
        "tool_uv",
        "[tool.uv]\npackage = true",
        "[tool.uv]\npackage = false",
    ),
    (
        "tool_uv_sources",
        'omnibase-core = { index = "pypi" }',
        'omnibase-core = { index = "internal" }',
    ),
    (
        "build_system_requires",
        'requires = ["hatchling"]',
        'requires = ["hatchling>=1.27"]',
    ),
)


@pytest.mark.parametrize(
    ("case", "old", "new"),
    DEPENDENCY_EDITS,
    ids=[case for case, _, _ in DEPENDENCY_EDITS],
)
def test_dependency_pyproject_edit_moves_both_digests(
    identity: Any,
    lock_data: dict[str, Any],
    tmp_path: Path,
    case: str,
    old: str,
    new: str,
) -> None:
    """AC2: the narrowing must not drop any table that shapes ``uv sync``.

    A projection that forgets one of these under-binds the image silently,
    which is strictly worse than the churn this ticket removes. Each table is
    asserted independently so the failure names the one that was dropped.
    """
    assert old in BASE_PYPROJECT, f"fixture anchor missing for {case}"
    before = _tree(tmp_path / "before", BASE_PYPROJECT)
    after = _tree(tmp_path / "after", BASE_PYPROJECT.replace(old, new, 1))

    env_before, id_before = _digests(identity, lock_data, before)
    env_after, id_after = _digests(identity, lock_data, after)
    assert env_before != env_after, (
        f"{case}: shared_env_digest did not change; this table shapes what "
        "uv sync installs and must participate in the binding"
    )
    assert id_before != id_after, (
        f"{case}: identity_digest did not change; this table shapes what "
        "uv sync installs and must participate in the binding"
    )


def test_uv_lock_still_binds_as_raw_bytes(
    identity: Any, lock_data: dict[str, Any], tmp_path: Path
) -> None:
    """``uv.lock`` is the resolved set; it keeps full raw-byte participation."""
    before = _tree(tmp_path / "before", BASE_PYPROJECT, uv_lock="version = 1\n")
    after = _tree(
        tmp_path / "after", BASE_PYPROJECT, uv_lock="version = 1\nrevision = 2\n"
    )
    assert _digests(identity, lock_data, before) != _digests(identity, lock_data, after)


# --------------------------------------------------------------------------- #
# AC3 — fail closed, and stay loud about unfamiliar structure                  #
# --------------------------------------------------------------------------- #


def test_unparseable_pyproject_fails_closed(env_digest: Any, tmp_path: Path) -> None:
    """A projection that cannot parse must raise, never hash a default.

    Hashing an empty projection on a parse failure would make every malformed
    tree agree with every other one, which is the silent under-binding the
    narrowing must not introduce.
    """
    import tomllib

    assert hasattr(env_digest, "pyproject_dependency_projection")
    root = _tree(tmp_path / "bad", BASE_PYPROJECT)
    (root / "pyproject.toml").write_text("[project\nname = broken", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        env_digest.pyproject_dependency_projection(root)


def test_missing_pyproject_fails_closed(env_digest: Any, tmp_path: Path) -> None:
    """An absent manifest is a broken tree, not an empty dependency set."""
    assert hasattr(env_digest, "pyproject_dependency_projection")
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        env_digest.pyproject_dependency_projection(root)


def test_new_top_level_table_is_visible_to_the_binding(
    identity: Any, lock_data: dict[str, Any], tmp_path: Path
) -> None:
    """AC3: an unfamiliar top-level table must not vanish from the binding.

    The projection allowlists the tables it understands. Without a tripwire, a
    future table that *does* shape the environment would be dropped in silence.
    Carrying the sorted top-level table names makes its arrival move the
    digest, which forces a deliberate decision instead of a silent omission.
    """
    before = _tree(tmp_path / "before", BASE_PYPROJECT)
    after = _tree(
        tmp_path / "after",
        BASE_PYPROJECT + '\n[tool.some-future-installer]\nmode = "eager"\n',
    )
    assert _digests(identity, lock_data, before) != _digests(
        identity, lock_data, after
    ), (
        "a new top-level table left the runner-image identity unchanged; the "
        "projection must carry the table-name set as a tripwire"
    )


def test_new_depth_three_table_is_visible_to_the_binding(
    identity: Any, lock_data: dict[str, Any], tmp_path: Path
) -> None:
    """A nested future-installer table must not vanish from the tripwire."""
    before = _tree(tmp_path / "before", BASE_PYPROJECT)
    after = _tree(
        tmp_path / "after",
        BASE_PYPROJECT
        + '\n[tool.some-future-installer.sources.internal]\npriority = "explicit"\n',
    )
    assert _digests(identity, lock_data, before) != _digests(
        identity, lock_data, after
    ), "a new depth-three table left the runner-image identity unchanged"


def test_absent_dependency_path_cannot_collide_with_literal_string(
    env_digest: Any, tmp_path: Path
) -> None:
    """Missing paths and attacker-controlled strings use different envelopes."""
    absent = BASE_PYPROJECT.replace(
        '\n[project.optional-dependencies]\ndev = ["pytest>=8.0.0"]\n',
        "\n",
    )
    literal = BASE_PYPROJECT.replace(
        'dev = ["pytest>=8.0.0"]',
        'dev = "<absent>"',
    )
    a = _tree(tmp_path / "absent", absent)
    b = _tree(tmp_path / "literal", literal)
    absent_projection = json.loads(
        env_digest.pyproject_dependency_projection(a).decode("utf-8")
    )
    literal_projection = json.loads(
        env_digest.pyproject_dependency_projection(b).decode("utf-8")
    )
    path = "project.optional-dependencies"
    assert absent_projection["values"][path] == {"present": False}
    assert literal_projection["values"][path] == {
        "present": True,
        "value": {
            "type": "table",
            "value": {"dev": {"type": "str", "value": "<absent>"}},
        },
    }
    assert absent_projection != literal_projection


def test_toml_datetime_dependency_value_fails_with_path_context(
    env_digest: Any, tmp_path: Path
) -> None:
    """Non-JSON TOML values fail closed with the offending dependency path."""
    root = _tree(
        tmp_path / "datetime",
        BASE_PYPROJECT.replace(
            "[tool.uv]\npackage = true",
            "[tool.uv]\npackage = true\ncache-until = 2026-09-13T12:00:00Z",
        ),
    )
    with pytest.raises(TypeError, match="tool\\.uv\\.cache-until"):
        env_digest.pyproject_dependency_projection(root)


@pytest.mark.parametrize(
    ("case", "toml_value"),
    [
        ("datetime", "2026-09-13T12:00:00Z"),
        ("date", "2026-09-13"),
        ("time", "12:00:00"),
        ("list", "[2026-09-13T12:00:00Z]"),
    ],
)
def test_non_json_toml_dependency_values_fail_with_path_context(
    env_digest: Any, tmp_path: Path, case: str, toml_value: str
) -> None:
    """Non-JSON TOML scalars fail closed, including inside lists."""
    root = _tree(
        tmp_path / case,
        BASE_PYPROJECT.replace(
            "[tool.uv]\npackage = true",
            f"[tool.uv]\npackage = true\ncache-until = {toml_value}",
        ),
    )
    with pytest.raises(TypeError, match="tool\\.uv\\.cache-until"):
        env_digest.pyproject_dependency_projection(root)


def test_non_finite_float_dependency_value_fails_with_path_context(
    env_digest: Any, tmp_path: Path
) -> None:
    """NaN/Infinity are rejected instead of serialized as non-canonical JSON."""
    root = _tree(
        tmp_path / "nan",
        BASE_PYPROJECT.replace(
            "[tool.uv]\npackage = true",
            "[tool.uv]\npackage = true\nresolution-weight = nan",
        ),
    )
    with pytest.raises(TypeError, match="tool\\.uv\\.resolution-weight"):
        env_digest.pyproject_dependency_projection(root)


def test_table_tripwire_is_bounded_after_depth_three(
    env_digest: Any, tmp_path: Path
) -> None:
    """Deep tool-internal nesting does not become an unbounded digest surface."""
    root = _tree(
        tmp_path / "deep",
        BASE_PYPROJECT
        + '\n[tool.some-future-installer.sources.internal.extra]\nmode = "ignored"\n',
    )
    projection = json.loads(
        env_digest.pyproject_dependency_projection(root).decode("utf-8")
    )
    assert "tool.some-future-installer.sources" in projection["tables"]
    assert "tool.some-future-installer.sources.internal" not in projection["tables"]


def test_projection_is_order_independent_across_tables(
    env_digest: Any, tmp_path: Path
) -> None:
    """Reordering whole tables must not move the projection.

    This is what makes the binding survive a merge that interleaves two
    independent edits: the projection is over parsed values, not file layout.
    A raw-bytes hash fails this by construction.
    """
    optional = '[project.optional-dependencies]\ndev = ["pytest>=8.0.0"]\n'
    groups = '[dependency-groups]\ntest = ["pytest-cov>=5.0.0"]\n'
    assert optional in BASE_PYPROJECT and groups in BASE_PYPROJECT
    swapped = (
        BASE_PYPROJECT.replace(optional, "\x00OPTIONAL\x00")
        .replace(groups, optional)
        .replace("\x00OPTIONAL\x00", groups)
    )
    assert swapped != BASE_PYPROJECT, "fixture did not actually reorder the tables"

    a = _tree(tmp_path / "a", BASE_PYPROJECT)
    b = _tree(tmp_path / "b", swapped)
    assert env_digest.pyproject_dependency_projection(
        a
    ) == env_digest.pyproject_dependency_projection(b)


# --------------------------------------------------------------------------- #
# AC4 — the gate is narrowed, never weakened                                   #
# --------------------------------------------------------------------------- #


def test_no_bypass_option_exists_on_the_identity_cli() -> None:
    """AC4: narrowing the inputs must not smuggle in an escape hatch.

    Asserted against the script's own source so adding a bypass flag is a red
    test rather than something a reviewer has to notice.
    """
    source = (SCRIPTS_CI / "runner_image_identity.py").read_text(encoding="utf-8")
    forbidden = ("--skip", "--force", "--no-verify", "--allow-stale", "--ignore-drift")
    present = [flag for flag in forbidden if flag in source]
    assert not present, f"identity CLI grew a bypass option: {present}"


def test_verify_still_fails_on_a_drifted_digest(
    identity: Any, lock_data: dict[str, Any], tmp_path: Path
) -> None:
    """AC4: the comparison itself is unchanged -- only its inputs narrowed."""
    drifted = dict(lock_data)
    drifted["identity_digest"] = "deadbeef" * 4
    lock_path = tmp_path / "runner-image.lock.json"
    lock_path.write_text(json.dumps(drifted), encoding="utf-8")
    assert identity.verify_lock(REPO_ROOT, lock_path) != 0
