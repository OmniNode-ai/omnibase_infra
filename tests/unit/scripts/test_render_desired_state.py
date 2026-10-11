# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19413: generator conformance and byte-identical replay of a fixture tag."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/lab_sync"))
from desired_state import parse_desired_state, render_desired_state

pytestmark = pytest.mark.unit


def inputs() -> dict:
    return {
        "target_ref": {
            "kind": "release",
            "repo": "OmniNode-ai/omnibase_infra",
            "ref": "v0.0.0-fixture",
            "commit": "a" * 40,
            "composition": [],
        },
        "host": "lab-201",
        "lane": "dev",
        "manifest_yaml": "hosts: {lab-201: {aliases: [omninode-pc]}}\nlanes:\n  dev:\n    hosts: [lab-201]\n    compose_project: omnibase-infra\n    services:\n      - {name: runtime, kind: service}\n      - {name: broker, kind: service}\n",
        "lock_toml": '[[package]]\nname = "omnibase-infra"\nversion = "1.2.3"\n[[package]]\nname = "omnibase-core"\nversion = "4.5.6"\n',
        "compose_json": json.dumps(
            {
                "services": {
                    "kernel": {
                        "container_name": "runtime",
                        "build": {"context": ".", "dockerfile": "Dockerfile.runtime"},
                        "healthcheck": {"test": ["CMD", "true"]},
                    },
                    "redpanda": {
                        "container_name": "broker",
                        "image": "redpanda:1",
                        "command": ["redpanda", "start", "--memory", "2G"],
                    },
                }
            }
        ),
        "compose_hashes": f"kernel {'b' * 64}\nredpanda {'c' * 64}\n",
        "broker_yaml": "segment_fallocation_step: 1048576\nlog_segment_ms: 86400000\nretention_bytes: 1073741824\n",
    }


def test_fixture_tag_renders_byte_identical() -> None:
    first = render_desired_state(inputs())
    second = render_desired_state(inputs())
    assert first == second
    state = parse_desired_state(first.decode())
    assert state.document["containers"][1]["packages"] == {
        "omnibase-infra": "1.2.3",
        "omnibase-core": "4.5.6",
    }
    assert state.document["containers"][0]["revision"] is None
    assert state.document["broker"]["memory"] == "2G"


def test_timestamp_is_refused() -> None:
    data = inputs()
    data["generated_at"] = "2026-09-24T17:05:00Z"
    with pytest.raises(ValueError):
        render_desired_state(data)


def test_byte_identical_after_source_order_changes() -> None:
    data = inputs()
    expected = render_desired_state(data)
    data["compose_hashes"] = "\n".join(reversed(data["compose_hashes"].splitlines()))
    compose = json.loads(data["compose_json"])
    compose["services"] = dict(reversed(list(compose["services"].items())))
    data["compose_json"] = json.dumps(compose)
    data["manifest_yaml"] = data["manifest_yaml"].replace(
        "      - {name: runtime, kind: service}\n      - {name: broker, kind: service}",
        "      - {name: broker, kind: service}\n      - {name: runtime, kind: service}",
    )
    assert render_desired_state(data) == expected


@pytest.mark.parametrize(
    "case",
    [
        "empty_hashes",
        "missing_hash",
        "bad_hash",
        "duplicate_hash",
        "missing_service",
        "empty_lane",
        "wrong_project",
        "wrong_host",
        "bad_ref",
        "conflicting_broker",
        "ambiguous_lock",
    ],
)
def test_refuses_incomplete_or_ambiguous_sources(case: str) -> None:
    data = inputs()
    if case == "empty_hashes":
        data["compose_hashes"] = ""
    elif case == "missing_hash":
        data["compose_hashes"] = data["compose_hashes"].splitlines()[1]
    elif case == "bad_hash":
        data["compose_hashes"] = "kernel not-a-hash"
    elif case == "duplicate_hash":
        data["compose_hashes"] += data["compose_hashes"]
    elif case == "empty_lane":
        data["manifest_yaml"] = (
            data["manifest_yaml"].split("    services:")[0] + "    services: []\n"
        )
    elif case == "wrong_host":
        data["host"] = "lab-999"
    elif case == "bad_ref":
        data["target_ref"]["commit"] = "abcdef"
    elif case == "ambiguous_lock":
        data["lock_toml"] += '[[package]]\nname = "omnibase-core"\nversion = "9.0"\n'
    else:
        compose = json.loads(data["compose_json"])
        if case == "missing_service":
            del compose["services"]["kernel"]
        elif case == "conflicting_broker":
            compose["services"]["init-a"] = {
                "command": ["rpk cluster config set retention_bytes 1"]
            }
            compose["services"]["init-b"] = {
                "command": ["rpk cluster config set retention_bytes 2"]
            }
        else:
            compose["name"] = "wrong-project"
        data["compose_json"] = json.dumps(compose)
    with pytest.raises((ValueError, KeyError)):
        render_desired_state(data)


def test_config_change_moves_hash_and_cannot_be_hidden_by_ordering() -> None:
    data = inputs()
    before = parse_desired_state(render_desired_state(data).decode())
    data["compose_hashes"] = data["compose_hashes"].replace("b" * 64, "d" * 64)
    after = parse_desired_state(render_desired_state(data).decode())
    assert after.desired_state_sha256 != before.desired_state_sha256
    assert after.document["containers"][1]["config_hash"] == "d" * 64


def test_broker_profile_reads_declared_one_shot_and_health_disable() -> None:
    data = inputs()
    compose = json.loads(data["compose_json"])
    compose["services"]["init"] = {
        "image": "redpanda:1",
        "command": ["rpk cluster config set retention_bytes 9876"],
    }
    compose["services"]["kernel"]["healthcheck"]["disable"] = True
    data["compose_json"] = json.dumps(compose)
    doc = parse_desired_state(render_desired_state(data).decode()).document
    assert doc["broker"]["cluster_config"]["retention_bytes"] == "9876"
    assert doc["containers"][1]["health_required"] is False


def test_declared_profile_gated_container_is_not_expected() -> None:
    data = inputs()
    data["manifest_yaml"] += "      - {name: disabled, kind: profile_gated}\n"
    doc = parse_desired_state(render_desired_state(data).decode()).document
    assert [r["name"] for r in doc["containers"]] == ["broker", "runtime"]


def test_cli_replay_preserves_file_mtime_and_refuses_secret_input(
    tmp_path: Path,
) -> None:
    import subprocess

    source = tmp_path / "source.json"
    output = tmp_path / "desired.json"
    source.write_text(json.dumps(inputs()))
    script = Path(__file__).resolve().parents[3] / "scripts/lab_sync/desired_state.py"
    command = [
        sys.executable,
        str(script),
        "render",
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    stamp = output.stat().st_mtime_ns
    second = subprocess.run(command, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    assert output.stat().st_mtime_ns == stamp
    data = inputs()
    data["secret_unknown_field"] = "must-never-appear-in-errors"
    source.write_text(json.dumps(data))
    refused = subprocess.run(command, capture_output=True, check=False)
    assert refused.returncode == 1
    assert b"must-never-appear-in-errors" not in refused.stderr + refused.stdout
    assert output.stat().st_mtime_ns == stamp


def test_host_surface_uses_the_manifest_allowance_list() -> None:
    data = inputs()
    data["surface_kind"] = "host"
    data["manifest_yaml"] += (
        "allowed_undeclared:\n  - name: cache\n    match: {compose_project: cache}\n    owner: infrastructure\n    reason: pull-through-cache\n"
    )
    doc = parse_desired_state(render_desired_state(data).decode()).document
    assert doc["surface"]["id"] == "lab-201/host"
    assert doc["allowed_undeclared"][0]["name"] == "cache"
    assert doc["containers"] == []


def test_runner_counts_come_from_fleet_config_and_hashes_from_compose() -> None:
    data = inputs()
    data.update(
        surface_kind="runner_fleet",
        runner_host_address="runner-host",
        runner_workdir="/srv/runners/docker",
        fleet_yaml="""version: '1.0'
github_org: OmniNode-ai
runner_host: runner-host
runner_group: omnibase-ci
runner_name_prefix: runner
expected_count: 2
hosts:
  - host: runner-host
    arch: amd64
    runner_name_prefix: runner
    expected_count: 2
    classes: [action]
""",
    )
    data["compose_json"] = json.dumps(
        {
            "name": "runners",
            "services": {
                f"runner-{i}": {
                    "container_name": f"runner-{i}",
                    "environment": {"LABELS": "omnibase-ci,arch-amd64"},
                }
                for i in (1, 2)
            },
        }
    )
    data["compose_hashes"] = f"runner-1 {'a' * 64}\nrunner-2 {'b' * 64}"
    doc = parse_desired_state(render_desired_state(data).decode()).document
    assert doc["runners"]["pools"] == [
        {
            "name": "runner",
            "expected_count": 2,
            "labels": ["arch-amd64", "omnibase-ci", "self-hosted"],
            "workdir": "/srv/runners/docker",
        }
    ]
    assert doc["runners"]["containers"][1]["config_hash"] == "b" * 64
    data["fleet_yaml"] = data["fleet_yaml"].replace(
        "expected_count: 2", "expected_count: 3"
    )
    with pytest.raises(ValueError, match="render refused"):
        render_desired_state(data)


def test_compose_null_command_inherits_image_default() -> None:
    data = inputs()
    compose = json.loads(data["compose_json"])
    compose["services"]["kernel"]["command"] = None
    data["compose_json"] = json.dumps(compose)
    assert (
        parse_desired_state(render_desired_state(data).decode()).document["containers"][
            1
        ]["revision"]
        == "a" * 40
    )


def test_broker_init_script_is_not_parsed_as_broker_argv() -> None:
    data = inputs()
    compose = json.loads(data["compose_json"])
    compose["services"]["init"] = {
        "image": "redpanda:1",
        "command": ["echo 'literal-script-fragment"],
    }
    data["compose_json"] = json.dumps(compose)
    doc = parse_desired_state(render_desired_state(data).decode()).document
    assert doc["broker"]["service"] == "broker"


def test_image_only_worker_inherits_the_declared_build_revision() -> None:
    data = inputs()
    compose = json.loads(data["compose_json"])
    compose["services"]["kernel"]["image"] = "infra-runtime:fixture"
    compose["services"]["worker"] = {
        "image": "infra-runtime:fixture",
        "container_name": "worker",
    }
    data["compose_json"] = json.dumps(compose)
    data["manifest_yaml"] += "      - {name: worker, kind: service}\n"
    data["compose_hashes"] += f"worker {'d' * 64}\n"
    doc = parse_desired_state(render_desired_state(data).decode()).document
    worker = next(row for row in doc["containers"] if row["name"] == "worker")
    assert worker["revision"] == "a" * 40
    assert worker["packages"]["omnibase-infra"] == "1.2.3"


@pytest.mark.parametrize("case", ["missing_route", "duplicate_route", "wrong_strategy"])
def test_render_refuses_an_unresolvable_contract_route(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import desired_state
    import yaml

    relative = Path(
        "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml"
    )
    contract = yaml.safe_load((desired_state._REPO_ROOT / relative).read_text())
    routing = contract["handler_routing"]
    entry = next(
        e for e in routing["handlers"] if e["operation"] == "lab_desired_state.render"
    )
    if case == "missing_route":
        routing["handlers"].remove(entry)
    elif case == "duplicate_route":
        routing["handlers"].append(entry.copy())
    else:
        routing["routing_strategy"] = "payload_type_match"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(desired_state, "_REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="exactly one route"):
        render_desired_state(inputs())


def test_bus_refusal_does_not_log_private_source_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
) -> None:
    from desired_state import main

    data = inputs()
    secret = "private-fixture-credential"
    data["broker_yaml"] = f"retention_bytes: [{secret}\n"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))

    assert main(["render", "--input", str(path)]) == 1
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert not captured.out
    assert secret not in captured.err + caplog.text
