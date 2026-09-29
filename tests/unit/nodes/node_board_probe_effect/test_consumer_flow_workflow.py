# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The C28 workflow delegates measurement and grading to the declared node."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]


def test_workflow_measurement_is_only_a_node_invocation() -> None:
    wf = yaml.safe_load(
        (ROOT / ".github/workflows/chain-canary-c28-consumer-flow.yml").read_text()
    )
    assert "push" not in wf.get("on", wf.get(True))
    steps = wf["jobs"]["c28-consumer-flow"]["steps"]
    measure = next(
        s for s in steps if s["name"] == "Re-measure C28 on the dev lane and grade it"
    )
    command = measure["run"].replace("\\\n", " ")
    assert command.strip().startswith(
        "uv run --frozen onex node node_board_probe_effect "
    )
    assert (
        "--contract" in command
        and "--input" in command
        and "--output receipt" in command
    )
    assert ">" not in command
    assert "||" not in command and "jq" not in command and ";" not in command
    assert "c28_consumer_flow_probe.py" not in command
    prep = next(
        s
        for s in steps[: steps.index(measure)]
        if s["name"] == "Prepare the consumer-flow node request"
    )
    assert "board_probe.consumer_flow" in prep["run"]
    assert "subject_lane" in prep["run"]
    assert 'record: "c28-consumer-flow.json"' in prep["run"]


@pytest.mark.parametrize(
    ("case", "exit_code"), [("pass", 0), ("fail", 1), ("unreadable", 1)]
)
def test_real_node_cli_writes_record_and_propagates_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, exit_code: int
) -> None:
    import json

    from click.testing import CliRunner

    from omnibase_infra.cli import cli_node
    from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
        HandlerDockerConsumerFlowTarget,
    )
    from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_observation import (
        ModelConsumerFlowObservation,
    )

    wf = yaml.safe_load(
        (ROOT / ".github/workflows/chain-canary-c28-consumer-flow.yml").read_text()
    )
    prep = next(
        s
        for s in wf["jobs"]["c28-consumer-flow"]["steps"]
        if s["name"] == "Prepare the consumer-flow node request"
    )["run"]
    script = prep.split("<<'PYCODE'\n")[1].rsplit("PYCODE", 1)[0]
    contract_rel = Path(
        "src/omnibase_infra/nodes/node_board_probe_effect/contract.yaml"
    )
    copied = tmp_path / contract_rel
    copied.parent.mkdir(parents=True)
    copied.write_text((ROOT / contract_rel).read_text())
    monkeypatch.chdir(tmp_path)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
    observed = json.loads(
        (Path(__file__).parent / "fixtures/consumer_flow_recorded.json").read_text()
    )
    if case == "fail":
        observed["cursor"]["walked_groups"] = []
    observation = ModelConsumerFlowObservation(
        read_ok=case != "unreadable",
        read_error="unreachable" if case == "unreadable" else "",
        **observed,
    )

    async def observe(self, request):
        return observation

    monkeypatch.setattr(HandlerDockerConsumerFlowTarget, "observe", observe)
    monkeypatch.setattr(cli_node, "check_omnimarket_drift", lambda **kwargs: None)
    record = tmp_path / "c28-consumer-flow.json"
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"subject_lane": "dev", "record": str(record)}))
    result = CliRunner().invoke(
        cli_node.run_node_by_name,
        [
            "node_board_probe_effect",
            "--contract",
            str(tmp_path / "c28-contract.yaml"),
            "--input",
            str(request),
            "--state-root",
            str(tmp_path / "state"),
            "--timeout",
            "10",
            "--output",
            "receipt",
            "--emit-socket",
            str(tmp_path / "no-socket"),
        ],
    )
    assert result.exit_code == exit_code, result.output + str(result.exception)
    body = json.loads(record.read_text())
    assert body["result"]["check_id"] == "consumer_flow"
    assert (
        body["result"]["outcome"]
        == {"pass": "PASS", "fail": "FAIL", "unreadable": "INDETERMINATE"}[case]
    )
