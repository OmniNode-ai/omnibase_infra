# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The sweep and deploy preflights resolve evidence in the owning repository."""

import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from scripts import apply_governed_compose_config as compose
from scripts import preflight_lane_deploy_attribution as attribution

ROOT = Path(__file__).resolve().parents[2]


def test_sweep_defaults_to_product_evidence() -> None:
    request = ModelEvidenceAutocloseSweepRequest(correlation_id=uuid4())
    assert request.evidence_repo == "OmniNode-ai/omnibase_infra"


def test_grants_default_to_release_repository(monkeypatch) -> None:
    monkeypatch.setenv("OMNI_HOME", "/registry")
    assert compose._grants_repo("") == Path("/registry/omnibase_infra")
    assert attribution.default_grants_repo({"OMNI_HOME": "/registry"}) == Path(
        "/registry/omnibase_infra"
    )
    assert compose.ANCHOR_REF == attribution.GRANTS_REF == "main"


def test_sweep_checks_out_product_contracts() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/evidence-autoclose-sweep.yml").read_text()
    )
    steps = workflow["jobs"]["evidence-autoclose-sweep"]["steps"]
    checkouts = [
        step.get("with", {})
        for step in steps
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert not any(
        item.get("repository") == "OmniNode-ai/onex_change_control"
        for item in checkouts
    )
    seed = next(step for step in steps if "CONTRACT_REPOS" in step.get("env", {}))
    assert "omnibase_infra" in seed["env"]["CONTRACT_REPOS"].split()
    assert not any("ONEX_CC_REPO_PATH" in step.get("env", {}) for step in steps)
    run = next(
        step["run"]
        for step in steps
        if "Run evidence autoclose sweep" in str(step.get("name", ""))
    )
    assert '--clone-root "${RUNNER_TEMP}/contract-roots"' in run
    assert "--product-repos" in run


@pytest.mark.asyncio
async def test_verifier_receives_owning_repository_contract(
    tmp_path, monkeypatch
) -> None:
    from omnibase_infra.handlers import done_write_receipt_guard as guard
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        HandlerEvidenceAutocloseSweep,
    )

    root = tmp_path / "product"
    (root / "contracts").mkdir(parents=True)
    contract = root / "contracts/OMN-17427.yaml"
    contract.write_text("owner: product\n")

    # Exercise the real subprocess boundary with a verifier that reports the
    # exact path and bytes it was given, without a credential or external I/O.
    def argv(ticket_id, *, contract_path=""):
        return [
            sys.executable,
            "-c",
            "import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); "
            "print(json.dumps({'contract_path': str(p), 'content': p.read_text()}))",
            contract_path,
        ]

    monkeypatch.setattr(guard, "dod_verify_argv", argv)
    handler = HandlerEvidenceAutocloseSweep(autoclose_disabled=False)
    result, code, error = await handler._run_dod_verify_command_real(
        "OMN-17427", str(root), 5
    )
    assert (code, error) == (0, "")
    assert result == {"contract_path": str(contract), "content": "owner: product\n"}
    missing, code, error = await handler._run_dod_verify_command_real(
        "OMN-17427", str(tmp_path / "missing"), 5
    )
    assert missing is None and code == -1
    assert "Product contract is unreadable" in error


def test_product_offer_binds_through_changed_contract() -> None:
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        HandlerEvidenceAutocloseSweep,
    )

    product = {
        "number": 1,
        "title": "fix(OMN-17427): retire reads",
        "merged_at": "2026-10-18T04:00:00Z",
        "files": ["contracts/OMN-17427.yaml"],
    }
    assert HandlerEvidenceAutocloseSweep._bind_offer_candidates(
        [product], "OMN-17427"
    ) == [product]
    assert (
        HandlerEvidenceAutocloseSweep._bind_offer_candidates(
            [{**product, "files": ["contracts/OMN-99999.yaml"]}], "OMN-17427"
        )
        == []
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/autoclose_contract_source_omn17427.json"
)
def test_product_checkout_replaces_recorded_occ_source(recorded_response) -> None:
    response = recorded_response["response"]
    assert (
        hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
        == recorded_response["_provenance"]["source_sha256"]
    )
    assert len(response["head"]) == 40
    assert any(
        item.get("repository") == "OmniNode-ai/onex_change_control"
        for item in response["checkouts"]
    )
    test_sweep_checks_out_product_contracts()


@pytest.mark.parametrize("field", ["evidence_repo", "product_repos"])
def test_repository_inputs_cannot_escape_the_contract_root(field) -> None:
    value = "OmniNode-ai/../../other"
    with pytest.raises(ValidationError):
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4(),
            **{field: (value,) if field == "product_repos" else value},
        )


def test_contract_sources_match_the_existing_product_fleet() -> None:
    def repos(name):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        return next(
            set(step["env"]["CONTRACT_REPOS"].split())
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if "CONTRACT_REPOS" in step.get("env", {})
        )

    assert repos("evidence-autoclose-sweep.yml") == repos(
        "in-progress-probe-hygiene.yml"
    )
    assert "onex_change_control" not in repos("evidence-autoclose-sweep.yml")
