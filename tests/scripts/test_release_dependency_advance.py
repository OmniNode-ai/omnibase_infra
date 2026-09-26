# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A sibling pin that runs ahead of the published release queues the release train (OMN-19655).

omnibase_infra pins its siblings exactly (``omnibase-core==X``). A core release
reaches infra's dev through the dependency cascade's bump PR, but nothing then
cuts infra: the published infra keeps pinning the old core, and every
downstream that raises its core floor is unresolvable until someone cuts by
hand. On 2026-09-25 omnibase_infra dev pinned omnibase-core==0.47.23 from
about 03:00Z while the newest published infra, 0.38.57, pinned ==0.47.22, and
omnimarket's release failed on every dev push for the whole day.

``release_dependency_advance.py`` measures exactly that state. These tests hold
its verdict and the workflow that acts on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import scripts.ci.release_dependency_advance as advance

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = (
    _REPO_ROOT / ".github" / "workflows" / "release-train-dependency-advance.yml"
)

#: omnibase-infra 0.38.57's Requires-Dist as PyPI served it on 2026-09-25.
_PUBLISHED_0_38_57 = [
    "omnibase-compat==0.5.7",
    "omnibase-core==0.47.22",
    "omnibase-spi==0.23.5",
    "pydantic<3.0.0,>=2.11.7",
    'python-snappy>=0.7; extra == "snappy"',
]


def _pyproject(tmp_path: Path, deps: list[str]) -> Path:
    body = "\n".join(f'    "{d}",' for d in deps)
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "omnibase-infra"\nversion = "0.38.58"\n'
        f"dependencies = [\n{body}\n]\n",
        encoding="utf-8",
    )
    return path


def _fetch(version: str, requires: list[str]) -> Any:
    def fetch(package: str) -> dict[str, Any]:
        assert package == "omnibase-infra"
        return {"info": {"version": version, "requires_dist": requires}}

    return fetch


@pytest.mark.unit
def test_exact_sibling_pins_reads_only_unconditional_exact_pins_on_our_packages() -> (
    None
):
    pins = advance.exact_sibling_pins(
        [
            "omnibase-core==0.47.23",
            "omnibase_spi == 0.23.5",
            "omnimarket>=0.4.0",
            "pydantic==2.11.7",
            'omnibase-compat==0.5.7; extra == "compat"',
        ]
    )
    assert pins == {"omnibase-core": "0.47.23", "omnibase-spi": "0.23.5"}


@pytest.mark.unit
def test_a_stranded_core_pin_is_an_advance(tmp_path: Path) -> None:
    """The 2026-09-25 state: dev pins core 0.47.23, published infra pins 0.47.22."""
    verdict = advance.decide(
        package="omnibase-infra",
        pyproject=_pyproject(
            tmp_path,
            [
                "omnibase-core==0.47.23",
                "omnibase-spi==0.23.5",
                "omnibase-compat==0.5.7",
            ],
        ),
        fetch=_fetch("0.38.57", _PUBLISHED_0_38_57),
    )
    assert verdict.stranded is True
    assert verdict.published_version == "0.38.57"
    assert [(a.name, a.published, a.dev) for a in verdict.advances] == [
        ("omnibase-core", "0.47.22", "0.47.23")
    ]


@pytest.mark.unit
def test_pins_in_step_with_the_published_release_are_not_an_advance(
    tmp_path: Path,
) -> None:
    """Negative control: the same comparison with dev equal to the release."""
    verdict = advance.decide(
        package="omnibase-infra",
        pyproject=_pyproject(
            tmp_path,
            [
                "omnibase-core==0.47.22",
                "omnibase-spi==0.23.5",
                "omnibase-compat==0.5.7",
            ],
        ),
        fetch=_fetch("0.38.57", _PUBLISHED_0_38_57),
    )
    assert verdict.stranded is False
    assert verdict.advances == ()


@pytest.mark.unit
def test_a_pin_behind_the_published_release_is_not_an_advance(tmp_path: Path) -> None:
    verdict = advance.decide(
        package="omnibase-infra",
        pyproject=_pyproject(tmp_path, ["omnibase-core==0.47.21"]),
        fetch=_fetch("0.38.57", _PUBLISHED_0_38_57),
    )
    assert verdict.stranded is False


@pytest.mark.unit
def test_a_new_exact_sibling_the_release_does_not_carry_is_an_advance(
    tmp_path: Path,
) -> None:
    verdict = advance.decide(
        package="omnibase-infra",
        pyproject=_pyproject(
            tmp_path, ["omnibase-core==0.47.22", "omninode-memory==0.18.2"]
        ),
        fetch=_fetch("0.38.57", _PUBLISHED_0_38_57),
    )
    assert [(a.name, a.published, a.dev) for a in verdict.advances] == [
        ("omninode-memory", "", "0.18.2")
    ]


@pytest.mark.unit
def test_main_writes_the_verdict_and_the_step_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(
        advance, "fetch_pypi_json", _fetch("0.38.57", _PUBLISHED_0_38_57)
    )
    pyproject = _pyproject(tmp_path, ["omnibase-core==0.47.23"])

    code = advance.main(["--package", "omnibase-infra", "--pyproject", str(pyproject)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stranded"] is True
    assert payload["advances"] == [
        {"name": "omnibase-core", "published": "0.47.22", "dev": "0.47.23"}
    ]
    assert "stranded=true" in output.read_text(encoding="utf-8")


@pytest.mark.unit
def test_an_unreadable_index_fails_closed_and_claims_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unread index is not "in step": exit 2, and no stranded=false output."""
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    def broken(package: str) -> dict[str, Any]:
        raise OSError("connection reset")

    monkeypatch.setattr(advance, "fetch_pypi_json", broken)
    code = advance.main(
        ["--package", "omnibase-infra", "--pyproject", str(_pyproject(tmp_path, []))]
    )
    assert code == 2
    assert not output.exists() or "stranded=" not in output.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The workflow that acts on the verdict
# ---------------------------------------------------------------------------


def _workflow() -> dict[Any, Any]:
    loaded = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.unit
def test_workflow_runs_after_every_dev_rebuild_and_hourly() -> None:
    triggers = _workflow().get("on", _workflow().get(True))
    assert isinstance(triggers, dict)
    run = triggers["workflow_run"]
    assert run["workflows"] == ["Runtime Rebuild Trigger"], (
        "a receipt for the bump merge arrives with the rebuild; judge then"
    )
    assert run["types"] == ["completed"]
    assert run["branches"] == ["dev"]
    assert triggers["schedule"], "an hourly backstop when a rebuild event is missed"
    assert "workflow_dispatch" in triggers


@pytest.mark.unit
def test_workflow_dispatches_the_train_for_infra_only_when_stranded() -> None:
    (job,) = _workflow()["jobs"].values()
    steps = job["steps"]
    measure = next(
        s for s in steps if "release_dependency_advance.py" in str(s.get("run", ""))
    )
    assert measure.get("id") == "advance"
    dispatch = next(s for s in steps if "gh workflow run" in str(s.get("run", "")))
    assert "steps.advance.outputs.stranded == 'true'" in str(dispatch.get("if", ""))
    body = str(dispatch["run"])
    assert "release-train-nightly.yml" in body
    assert "repos=omnibase_infra" in body
    assert "--ref dev" in body
    assert job["permissions"]["actions"] == "write"
    assert (
        "contents" not in job["permissions"] or job["permissions"]["contents"] == "read"
    )


@pytest.mark.unit
def test_workflow_does_not_redispatch_while_a_release_pr_is_open() -> None:
    (job,) = _workflow()["jobs"].values()
    dispatch = next(
        s for s in job["steps"] if "gh workflow run" in str(s.get("run", ""))
    )
    body = str(dispatch["run"])
    assert "gh pr list" in body
    assert body.index("gh pr list") < body.index("gh workflow run")
