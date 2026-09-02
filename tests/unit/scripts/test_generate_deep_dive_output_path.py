# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17235 — a generated deep dive lands in knowledge-base-internal, or nowhere.

Operator ruling, 2026-08-31 (cited by OMN-16990 DoD 8): plans for current work,
deep dives, and tracking / status / rollup artifacts go to
knowledge-base-internal; ``omni_home/docs`` keeps doctrine + architecture
reference.

``generate_deep_dive.py`` used to default its ``--out`` to
``<root>/docs/deep-dives/``, where ``--root`` defaults to ``$OMNI_HOME`` — so
the default behaviour wrote a doc-class artifact straight into the registry the
ruling excludes it from. ``omni_home`` now runs a ``kb-doc-gate`` required check
that rejects a new markdown file there, so the old default would fail its own
commit.

The two properties under test are the ones that make the repoint stick:

* the default resolves to ``$KNOWLEDGE_BASE_INTERNAL_PATH/beta/deep-dives/``,
  mirroring the layout the morning workflows already publish into; and
* an unset variable **raises**, rather than falling back to a path that happens
  to exist (CLAUDE.md Operating Rule 8). A silent default is precisely how the
  drift this ticket closes happened in the first place.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "generate_deep_dive.py"


def _module():
    spec = importlib.util.spec_from_file_location("generate_deep_dive", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_default_out_path_is_the_kb_internal_deep_dives_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module()
    monkeypatch.setenv("KNOWLEDGE_BASE_INTERNAL_PATH", str(tmp_path))
    out = mod.default_out_path(dt.date(2026, 9, 2))
    assert out.parent == tmp_path.resolve() / "beta" / "deep-dives", (
        f"deep dives belong in the kb-internal beta/deep-dives/ mirror, got {out}"
    )
    assert out.name.endswith("_DEEP_DIVE.md")


@pytest.mark.unit
def test_unset_env_raises_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-fast, not a silent fallback — CLAUDE.md Operating Rule 8."""
    mod = _module()
    monkeypatch.delenv("KNOWLEDGE_BASE_INTERNAL_PATH", raising=False)
    with pytest.raises(KeyError) as excinfo:
        mod.default_out_path(dt.date(2026, 9, 2))
    message = str(excinfo.value)
    assert "KNOWLEDGE_BASE_INTERNAL_PATH" in message, "the error must name the variable"
    assert "no default path" in message and "no omni_home fallback" in message, (
        "the error must state that there is no fallback, so a reader does not go "
        "looking for one"
    )


@pytest.mark.unit
def test_unset_env_does_not_fall_back_to_omni_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The specific regression: OMNI_HOME being set must not rescue the default.

    ``--root`` still defaults to ``$OMNI_HOME`` (the scan target is genuinely the
    registry), so it would be easy to reintroduce ``root / "docs" / "deep-dives"``
    as a fallback. That is the bug, not the fix.
    """
    mod = _module()
    monkeypatch.delenv("KNOWLEDGE_BASE_INTERNAL_PATH", raising=False)
    monkeypatch.setenv("OMNI_HOME", "/nonexistent/omni_home")
    with pytest.raises(KeyError):
        mod.default_out_path(dt.date(2026, 9, 2))


@pytest.mark.unit
def test_non_directory_env_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path that is set but wrong fails loudly too — otherwise the report is
    written into a freshly created directory nobody is watching."""
    mod = _module()
    stray = tmp_path / "not-a-clone.txt"
    stray.write_text("x", encoding="utf-8")
    monkeypatch.setenv("KNOWLEDGE_BASE_INTERNAL_PATH", str(stray))
    with pytest.raises(NotADirectoryError):
        mod.default_out_path(dt.date(2026, 9, 2))


@pytest.mark.unit
def test_script_no_longer_defaults_into_omni_home_docs() -> None:
    """Guard the literal that was the defect."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"docs" / "deep-dives"' not in source, (
        "generate_deep_dive.py still defaults into <root>/docs/deep-dives — that is "
        "an omni_home path and the kb-doc-gate rejects a new markdown file there"
    )
