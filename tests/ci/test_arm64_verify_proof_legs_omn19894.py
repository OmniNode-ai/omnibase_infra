# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The arm64 proof's legs are data, checked against the inventory (OMN-19894).

Pins that .github/actions/resolve-arm64-proof-legs refuses every way the legs
variable can disagree with itself or with the declared inventory. Each refusal
is RED, never a default. The inventory here is a fixture, so these tests do not
depend on which machines the lab has today.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "resolve-arm64-proof-legs"
_spec = importlib.util.spec_from_file_location(
    "resolve_arm64_proof_legs", ACTION_DIR / "resolve_arm64_proof_legs.py"
)
assert _spec is not None and _spec.loader is not None
legs_mod = importlib.util.module_from_spec(_spec)
sys.modules["resolve_arm64_proof_legs"] = legs_mod
_spec.loader.exec_module(legs_mod)
LegsError = legs_mod.LegsError
check_against_inventory = legs_mod.check_against_inventory
declared_host_labels = legs_mod.declared_host_labels
parse_legs = legs_mod.parse_legs

pytestmark = pytest.mark.unit


def _leg(host: str, name: str | None = None, creds: bool = True) -> dict[str, object]:
    return {
        "name": name or host,
        "runs_on": ["self-hosted", "omnibase-verify", "arch-arm64", host],
        "lab_credentials": creds,
    }


def _inventory(tmp_path: Path) -> tuple[Path, Path]:
    docker = tmp_path / "docker"
    docker.mkdir()
    fleet = tmp_path / "runner_fleet.yaml"
    fleet.write_text(
        "runner_name_prefix: primary-runner\n"
        "hosts:\n"
        "  - {host: a, arch: amd64, runner_name_prefix: primary-runner,"
        " classes: [action]}\n"
        "  - {host: b, arch: arm64, runner_name_prefix: mac-a, classes: [verify]}\n"
        "  - {host: c, arch: arm64, runner_name_prefix: mac-b, classes: [verify]}\n"
        "  - {host: d, arch: amd64, runner_name_prefix: pc-b, classes: [verify]}\n",
        encoding="utf-8",
    )
    primary = docker / "docker-compose.runners.yml"
    primary.write_text(
        "services:\n  primary-runner-1:\n    environment:\n"
        "      RUNNER_LABELS: self-hosted,omnibase-ci,host-1\n",
        encoding="utf-8",
    )
    for prefix, label, arch in (
        ("mac-a", "host-7", "arch-arm64"),
        ("mac-b", "host-8", "arch-arm64"),
        ("pc-b", "host-9", "arch-amd64"),
    ):
        (docker / f"docker-compose.runners-{prefix}.yml").write_text(
            f"services:\n  {prefix}-1:\n    environment:\n"
            f"      RUNNER_LABELS: self-hosted,omnibase-verify,{label},{arch}\n",
            encoding="utf-8",
        )
    return fleet, primary


def test_the_inventory_names_only_arm64_verify_hosts(tmp_path: Path) -> None:
    fleet, primary = _inventory(tmp_path)
    assert declared_host_labels(fleet, primary) == {"host-7", "host-8"}


def test_matching_legs_pass(tmp_path: Path) -> None:
    fleet, primary = _inventory(tmp_path)
    legs = parse_legs(json.dumps([_leg("host-7"), _leg("host-8", creds=False)]))
    check_against_inventory(legs, declared_host_labels(fleet, primary))


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_an_unset_variable_is_red(raw: str | None) -> None:
    with pytest.raises(LegsError, match="unset or empty"):
        parse_legs(raw)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("not json", "not JSON"),
        ("[]", "non-empty JSON list"),
        ('{"a": 1}', "non-empty JSON list"),
        (json.dumps([{"runs_on": [], "lab_credentials": True}]), "no name"),
        (
            json.dumps([{"name": "x", "runs_on": "host-7", "lab_credentials": True}]),
            "must be a list",
        ),
        (
            json.dumps(
                [
                    {
                        "name": "x",
                        "runs_on": ["self-hosted", "host-7"],
                        "lab_credentials": True,
                    }
                ]
            ),
            "lacks",
        ),
        (
            json.dumps(
                [
                    {
                        "name": "x",
                        "runs_on": ["self-hosted", "omnibase-verify", "arch-arm64"],
                        "lab_credentials": True,
                    }
                ]
            ),
            "exactly one host label",
        ),
        (
            json.dumps([{**_leg("host-7"), "lab_credentials": "yes"}]),
            "true or false",
        ),
    ],
)
def test_a_malformed_variable_is_red(raw: str, match: str) -> None:
    with pytest.raises(LegsError, match=match):
        parse_legs(raw)


def test_a_declared_host_with_no_leg_is_red(tmp_path: Path) -> None:
    fleet, primary = _inventory(tmp_path)
    with pytest.raises(LegsError, match=r"no leg: \['host-8'\]"):
        check_against_inventory(
            parse_legs(json.dumps([_leg("host-7")])),
            declared_host_labels(fleet, primary),
        )


def test_a_leg_for_an_undeclared_host_is_red(tmp_path: Path) -> None:
    fleet, primary = _inventory(tmp_path)
    legs = parse_legs(json.dumps([_leg("host-7"), _leg("host-8"), _leg("host-9")]))
    with pytest.raises(LegsError, match=r"no compose file registers: \['host-9'\]"):
        check_against_inventory(legs, declared_host_labels(fleet, primary))


def test_two_legs_for_one_host_is_red(tmp_path: Path) -> None:
    fleet, primary = _inventory(tmp_path)
    legs = parse_legs(
        json.dumps([_leg("host-7"), _leg("host-7", name="again"), _leg("host-8")])
    )
    with pytest.raises(LegsError, match="more than one leg"):
        check_against_inventory(legs, declared_host_labels(fleet, primary))


def test_main_writes_the_legs_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fleet, primary = _inventory(tmp_path)
    monkeypatch.setattr(legs_mod, "FLEET_CONFIG", fleet)
    monkeypatch.setattr(legs_mod, "PRIMARY_COMPOSE", primary)
    monkeypatch.setattr(
        legs_mod,
        "declared_host_labels",
        lambda: declared_host_labels(fleet, primary),
    )
    output = tmp_path / "out"
    env = {
        "GITHUB_OUTPUT": str(output),
        "ARM64_VERIFY_PROOF_LEGS_JSON": json.dumps([_leg("host-7"), _leg("host-8")]),
    }
    assert legs_mod.main(env) == 0
    written = output.read_text(encoding="utf-8").strip()
    assert written.startswith("legs=")
    assert [leg["name"] for leg in json.loads(written[5:])] == ["host-7", "host-8"]

    assert legs_mod.main({"GITHUB_OUTPUT": str(output)}) == 1
    assert "::error::ARM64_VERIFY_PROOF_LEGS_JSON is unset" in capsys.readouterr().out
