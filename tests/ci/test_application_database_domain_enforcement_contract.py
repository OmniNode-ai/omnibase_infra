# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mandatory-source and blocked-deployment ratchet for OMN-15361."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from omnibase_infra.validation.application_database_red_control_registry import (
    APPLICATION_DATABASE_RED_CONTROL_REGISTRY,
    ApplicationDatabaseRedControlBinding,
)
from omnibase_infra.validation.application_relation_ownership import (
    load_service_ownership_manifest,
)
from omnibase_infra.validation.enums.enum_application_database_enforcement_gate import (
    EnumApplicationDatabaseEnforcementGate,
)
from omnibase_infra.validation.enums.enum_application_database_object_kind import (
    EnumApplicationDatabaseObjectKind,
)
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)
from omnibase_infra.validation.models.model_application_database_enforcement_contract import (
    ModelApplicationDatabaseEnforcementContract,
)
from omnibase_infra.validation.models.model_database_object_evidence import (
    ModelDatabaseObjectEvidence,
)
from omnibase_infra.validation.models.model_relation_evidence import (
    ModelRelationEvidence,
)

_ROOT = Path(__file__).parents[2]
_CONTRACT = _ROOT / "config" / "application_database_domain_enforcement.yaml"
_OWNERSHIP = _ROOT / "config" / "application_database_domain_proof_ownership.yaml"
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_DOMAIN_PROOF = (
    _ROOT / "scripts" / "ci" / "prove_application_database_domain_enforcement.py"
)
_ACL_PROOF = _ROOT / "scripts" / "ci" / "prove_application_database_acl.py"
_ACL_POSTGRES16_INVENTORY = (
    _ROOT
    / "tests"
    / "fixtures"
    / "application_database_acl"
    / "principal-inventory-postgres16.yaml"
)
_ACL_FIXTURES = _ACL_POSTGRES16_INVENTORY.parent
_ACL_PRECHANGE = (
    _ROOT
    / "docker"
    / "application-acl-proof"
    / "generated"
    / "prechange-fixture-acl.json"
)


def _contract() -> ModelApplicationDatabaseEnforcementContract:
    return ModelApplicationDatabaseEnforcementContract.model_validate(
        yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    )


def test_acl_postgres16_evidence_preserves_range_type_identities() -> None:
    inventory = yaml.safe_load(_ACL_POSTGRES16_INVENTORY.read_text(encoding="utf-8"))
    inventory_kinds = {
        item["catalog_kind"]
        for item in inventory["observed_objects"]
        if item["object_ref"] == "account_id_span"
        and item.get("function_signature") is None
    }
    assert inventory_kinds == {"range_type"}
    multirange_inventory_kinds = {
        item["catalog_kind"]
        for item in inventory["observed_objects"]
        if item["object_ref"] == "account_id_span_set"
        and item.get("function_signature") is None
    }
    assert multirange_inventory_kinds == {"multirange_type"}

    prechange = json.loads(_ACL_PRECHANGE.read_text(encoding="utf-8"))
    for section in ("object_acl", "object_owners"):
        object_kinds = {
            item["object_name"]: item["catalog_kind"]
            for item in prechange[section]
            if item["object_type"] == "TYPE"
        }
        assert object_kinds["account_id_span"] == "range_type", section
        assert object_kinds["account_id_span_set"] == "multirange_type", section


def test_acl_prechange_artifact_rows_are_canonically_sorted() -> None:
    prechange = json.loads(_ACL_PRECHANGE.read_text(encoding="utf-8"))
    for section, rows in prechange.items():
        if not isinstance(rows, list) or not rows:
            continue
        assert rows == sorted(
            rows,
            key=lambda row: json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ), section


def test_acl_live_snapshot_classifies_postgres_type_subkinds_exactly() -> None:
    source = _ACL_PROOF.read_text(encoding="utf-8")
    exact_catalog_kind_case = "\n".join(
        (
            "CASE type.typtype",
            "                 WHEN 'b' THEN 'base_type'",
            "                 WHEN 'r' THEN 'range_type'",
            "                 WHEN 'm' THEN 'multirange_type'",
            "                 ELSE 'type'",
            "               END AS catalog_kind",
        )
    )

    assert source.count(exact_catalog_kind_case) == 2
    assert "'type' AS catalog_kind, 'TYPE' AS owner_keyword" not in source
    assert "'TYPE' AS object_type, 'type' AS catalog_kind" not in source


def test_acl_postgres16_fixture_matrix_is_ready_and_rollback_keywords_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_DSN", "dbname=unused")
    monkeypatch.setenv("ACL_FIXTURES", str(_ACL_FIXTURES))
    proof = runpy.run_path(str(_ACL_PROOF), run_name="acl_proof_contract")

    matrix = proof["_fixture_matrix"]()
    assert matrix.status == "READY", matrix.blockers
    typed_objects = {
        obj.catalog_kind: obj
        for obj in matrix.objects
        if obj.object_ref in {"account_id_span", "account_id_span_set"}
        and obj.function_signature is None
    }
    assert set(typed_objects) == {"range_type", "multirange_type"}
    assert proof["_acl_object_keyword"](typed_objects["range_type"]) == "TYPE"
    assert proof["_acl_object_keyword"](typed_objects["multirange_type"]) == "TYPE"


def test_every_domain_gate_is_mandatory_in_source_and_has_red_green_proof() -> None:
    contract = _contract()

    assert set(contract.gates) == set(EnumApplicationDatabaseEnforcementGate)
    for gate, state in contract.gates.items():
        assert state.source_enforcement == "mandatory", gate
        assert state.source_proofs, gate
        assert state.seeded_red_controls, gate
        for proof_path in state.source_proof_paths:
            assert (_ROOT / proof_path).is_file(), (gate, proof_path)


def test_seeded_red_control_registry_rejects_a_phantom_control() -> None:
    source = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    source["gates"]["classification"]["seeded_red_controls"].append("phantom-control")

    with pytest.raises(ValueError, match="RED control registry"):
        ModelApplicationDatabaseEnforcementContract.model_validate(source)


def test_seeded_red_control_registry_rejects_a_renamed_control() -> None:
    source = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    source["gates"]["classification"]["seeded_red_controls"][0] = "renamed-control"

    with pytest.raises(ValueError, match="RED control registry"):
        ModelApplicationDatabaseEnforcementContract.model_validate(source)


@pytest.mark.parametrize(
    ("control_id", "mismatched_node_id"),
    [
        (
            "unknown-topology-schema",
            "tests/unit/validation/"
            "test_application_database_domain_enforcement.py::"
            "test_topology_schema_domain_drift_fails_closed",
        ),
        (
            "identity-root-unproven-enumeration",
            "tests/unit/validation/"
            "test_application_database_domain_enforcement.py::"
            "test_tenant_identity_root_requires_closed_contract_relation_and_primary_key",
        ),
    ],
)
def test_red_control_binding_rejects_a_semantically_mismatched_node(
    control_id: str,
    mismatched_node_id: str,
) -> None:
    with pytest.raises(ValueError, match="exact semantic pytest case"):
        ApplicationDatabaseRedControlBinding(
            control_id=control_id,
            pytest_node_id=mismatched_node_id,
        )


def test_red_control_registry_is_deeply_immutable_and_one_to_one() -> None:
    gate = EnumApplicationDatabaseEnforcementGate.CLASSIFICATION
    controls = APPLICATION_DATABASE_RED_CONTROL_REGISTRY[gate]
    binding = controls["missing-owner"]
    all_bindings = tuple(
        item
        for gate_bindings in APPLICATION_DATABASE_RED_CONTROL_REGISTRY.values()
        for item in gate_bindings.values()
    )

    assert len({item.pytest_node_id for item in all_bindings}) == len(all_bindings)
    with pytest.raises(TypeError, match="does not support item assignment"):
        controls["missing-owner"] = binding  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        binding.pytest_node_id = "tests/phantom.py::test_phantom"  # type: ignore[misc]


def test_seeded_red_controls_execute_exactly_and_pass_in_the_required_ci_path() -> None:
    contract = _contract()
    bindings = tuple(
        binding
        for proofs in APPLICATION_DATABASE_RED_CONTROL_REGISTRY.values()
        for binding in proofs.values()
    )
    node_ids = tuple(binding.pytest_node_id for binding in bindings)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *node_ids],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(node_ids)} passed" in result.stdout, result.stdout + result.stderr
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    required_step = workflow.split(
        "- name: Execute mandatory source assertions and seeded RED controls",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]

    for gate, proofs in APPLICATION_DATABASE_RED_CONTROL_REGISTRY.items():
        state = contract.gates[gate]
        assert set(proofs) == set(state.seeded_red_controls), gate
        for control, binding in proofs.items():
            assert binding.control_id == control
            node_id = binding.pytest_node_id
            proof_path = node_id.partition("::")[0]
            assert proof_path in state.source_proof_paths, (gate, control, proof_path)
            assert proof_path in required_step, (gate, control, proof_path)


def test_docker_backed_red_controls_bind_to_exact_emitted_pass_results() -> None:
    proof_source = _DOMAIN_PROOF.read_text(encoding="utf-8")

    for gate, proofs in APPLICATION_DATABASE_RED_CONTROL_REGISTRY.items():
        for control, binding in proofs.items():
            if binding.docker_result is None:
                continue
            assert binding.docker_result == f"domain_control={control} status=PASS"
            assert f'"{control}"' in proof_source, (gate, control)


def test_unmet_deployment_preconditions_are_explicit_blockers_not_green_claims() -> (
    None
):
    contract = _contract()

    assert all(
        state.deployment_enforcement == "blocked" for state in contract.gates.values()
    )
    blocker_text = "\n".join(
        blocker
        for state in contract.gates.values()
        for blocker in state.deployment_blockers
    )
    assert "OMN-15423" in blocker_text
    assert "OMN-15358" in blocker_text
    assert "OMN-15416" in blocker_text
    assert "OMN-15424" in blocker_text
    assert "OMN-15425" in blocker_text
    assert "OMN-15426" in blocker_text
    assert "full-day" in blocker_text
    assert "secret" in blocker_text
    assert "deploy" in blocker_text


def test_kubernetes_parity_remains_a_typed_blocker_not_a_source_green_claim() -> None:
    contract = _contract()
    topology_gate = contract.gates[
        EnumApplicationDatabaseEnforcementGate.TOPOLOGY_PARITY
    ]

    source_text = "\n".join(topology_gate.source_proofs).lower()
    assert "kubernetes" not in source_text or "blocked" in source_text
    assert any("Kubernetes" in blocker for blocker in topology_gate.deployment_blockers)


def test_exact_predecessor_pins_are_immutable_and_complete() -> None:
    contract = _contract()

    assert contract.predecessor_pins == {
        "omnibase_core#1529": "1f4549d71d4d39560ac5a162ac1d39e54d86e688",
        "omnibase_infra#2547": "95351f5d8e806fcf7fa2c276d9065df93ccf92b9",
        "omnibase_infra#2548": "7228ce0c0934ae096dd6effd0f84ff1913fec6c0",
        "omnibase_infra#2558": "2a2cfb275810b34197d4d5baf55bdcddc443e6dc",
        "omnimarket#1956": "4637e625c99ef17c190aa471a5e51b7f646c6dfd",
        "omninode_infra#771": "39033d55147ef22a061b665345b506246d3aa543",
    }


def test_exact_domain_adapter_predecessor_is_checked_out_and_executed() -> None:
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")

    assert "ref: 2a2cfb275810b34197d4d5baf55bdcddc443e6dc" in workflow
    assert "path: .proof-dependencies/domain-adapter" in workflow
    assert "working-directory: .proof-dependencies/domain-adapter" in workflow


def test_private_ownership_pin_is_pat_authenticated_and_fork_fail_closed() -> None:
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")

    # OMN-16373: CROSS_REPO_PAT retired in favor of a minted onexbot-occ-writer
    # App installation token (steps.app-token-proof), same private-repo
    # authentication requirement -- an empty App token on a fork PR fails
    # the same way an empty PAT did.
    assert "token: ${{ steps.app-token-proof.outputs.token }}" in workflow
    assert "Enforce schema qualification in changed SQL (trusted)" in workflow
    assert (
        "Enforce schema qualification in changed SQL (public fork, fail closed)"
        in workflow
    )
    fork_step = workflow.split(
        "- name: Enforce schema qualification in changed SQL (public fork, fail closed)",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]
    assert ".proof-dependencies/omninode-infra" not in fork_step
    assert "config/application_database_domain_proof_ownership.yaml" not in fork_step


def test_cross_repo_pat_lane_is_dependabot_fail_closed_too() -> None:
    """OMN-16152: GitHub withholds ALL repo secrets from dependabot-triggered
    `pull_request` runs, regardless of whether the head branch is a fork. The
    "trusted" lane's `if:` only tested for fork-vs-same-repo, so a dependabot
    PR (same-repo branch, not a fork) was routed into the CROSS_REPO_PAT
    checkout with an empty token -- actions/checkout then fails hard with
    "Input required and not supplied: token" instead of falling back to
    github.token, because an explicitly-passed empty string is not "unset".

    Dependabot must take the same narrow, already-proven fail-closed lane as
    a fork PR: skip the CROSS_REPO_PAT-gated omninode_infra checkout and
    validate against only the public omnimarket ownership manifest.
    """
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")

    checkout_step = workflow.split(
        "- name: Checkout exact service ownership dependency",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]
    assert "github.actor != 'dependabot[bot]'" in checkout_step, (
        "the CROSS_REPO_PAT-gated omninode_infra checkout must not run for "
        "dependabot[bot] -- GitHub withholds secrets.CROSS_REPO_PAT from "
        "dependabot pull_request runs even on a same-repo (non-fork) branch"
    )

    trusted_step = workflow.split(
        "- name: Enforce schema qualification in changed SQL (trusted)",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]
    assert "github.actor != 'dependabot[bot]'" in trusted_step, (
        "the trusted SQL-enforcement step depends on the omninode_infra "
        "checkout above and must not run for dependabot[bot] either"
    )

    fork_step = workflow.split(
        "- name: Enforce schema qualification in changed SQL (public fork, fail closed)",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]
    assert "github.actor == 'dependabot[bot]'" in fork_step, (
        "the fail-closed lane must explicitly cover dependabot[bot] too, "
        "not just the fork condition, so dependabot PRs still get a real "
        "SQL-enforcement pass instead of silently skipping both lanes"
    )


def test_live_omnimarket_head_resolution_survives_fork_prs_without_org_secrets() -> (
    None
):
    """OMN-15703 forkfix: the live-resolve step must not hard-depend on
    CROSS_REPO_PAT. Fork-triggered pull_request runs receive no org secrets,
    so a bare `secrets.CROSS_REPO_PAT` GH_TOKEN resolves empty, `gh api`
    fails unauthenticated, and `set -euo pipefail` aborts the job before its
    dedicated fork-lane proof step
    ("Enforce schema qualification in changed SQL (public fork, fail closed)")
    ever runs. omnimarket is a public repo, so github.token (always present,
    including on fork PRs) is sufficient to read its commits API -- the step
    must fall back to it rather than failing closed on token absence alone.
    """
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")

    live_resolve_step = workflow.split(
        "- name: Live-resolve omnimarket dev HEAD",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]

    # Must not hard-depend on the minted App token alone (OMN-16373:
    # CROSS_REPO_PAT retired) -- a fork PR run has no org secrets, so a bare
    # `${{ steps.app-token-proof.outputs.token }}` here means an empty
    # GH_TOKEN and an unauthenticated `gh api` failure under
    # `set -euo pipefail`.
    assert (
        "GH_TOKEN: ${{ steps.app-token-proof.outputs.token }}" not in live_resolve_step
    )
    # Must fall back to the always-present github.token so the fork lane
    # keeps resolving (omnimarket is public; no elevated scope is needed).
    assert (
        "GH_TOKEN: ${{ steps.app-token-proof.outputs.token || github.token }}"
        in live_resolve_step
    )
    # Fail-closed behavior is preserved: no mutable-tag/latest/dev-ref
    # fallback, hard exit on invalid resolution, still gated to the
    # not-yet-pinned ("dev") case only.
    assert "set -euo pipefail" in live_resolve_step
    assert "exit 1" in live_resolve_step
    assert "if: steps.resolve-omnimarket-ref.outputs.ref == 'dev'" in live_resolve_step


def test_predecessor_pins_reject_in_place_mutation() -> None:
    contract = _contract()

    with pytest.raises(TypeError, match="does not support item assignment"):
        # Intentional runtime mutation probe against a statically read-only Mapping.
        contract.predecessor_pins["omnibase_core#1529"] = "0" * 40  # type: ignore[index]


def test_gates_reject_in_place_mutation() -> None:
    contract = _contract()
    gate = EnumApplicationDatabaseEnforcementGate.CLASSIFICATION

    with pytest.raises(TypeError, match="does not support item assignment"):
        # Intentional runtime mutation probe against a statically read-only Mapping.
        contract.gates[gate] = contract.gates[gate]  # type: ignore[index]


def test_yaml_contract_remains_json_serializable_without_shape_drift() -> None:
    source = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    contract = ModelApplicationDatabaseEnforcementContract.model_validate(source)

    assert contract.model_dump(mode="json") == source
    assert json.loads(contract.model_dump_json()) == source


def test_source_tenant_provenance_is_declared_by_typed_ownership_evidence() -> None:
    manifest = load_service_ownership_manifest(_OWNERSHIP)
    matches = tuple(
        evidence
        for evidence in manifest.relation_evidence
        if evidence.database_ref == "application"
        and evidence.schema == "omninode_internal"
        and evidence.name == "runtime_state"
    )

    assert len(matches) == 1
    assert (
        matches[0].source_tenant_provenance_contract == "non_authoritative_provenance"
    )
    assert matches[0].deduplication_key_columns == ("state_id",)
    assert matches[0].authorization_dependency_columns == ()
    assert matches[0].write_eligibility_dependency_columns == ()


def test_tenant_identity_and_function_audit_are_checked_manifest_authority() -> None:
    manifest = load_service_ownership_manifest(_OWNERSHIP)
    identity_root = tuple(
        evidence
        for evidence in manifest.relation_evidence
        if evidence.database_ref == "application"
        and evidence.schema == "tenant"
        and evidence.name == "tenants"
    )
    tenant = tuple(
        evidence
        for evidence in manifest.relation_evidence
        if evidence.database_ref == "application"
        and evidence.schema == "tenant"
        and evidence.name == "events"
    )
    function = tuple(
        database_object
        for database_object in manifest.database_objects
        if database_object.database_ref == "application"
        and database_object.schema == "tenant"
        and database_object.name == "safe_report"
    )

    assert len(identity_root) == 1
    assert identity_root[0].identity_root_control_role == "tenant_control_admin"
    assert tuple(
        operation.value
        for operation in identity_root[0].identity_root_control_operations
    ) == ("tenant_creation", "cross_tenant_enumeration")
    assert len(tenant) == 1
    assert tenant[0].tenant_identity_column == "tenant_id"
    assert tenant[0].identity_root_contract is None
    assert tenant[0].canonical_policy_name == "tenant_isolation"
    assert len(function) == 1
    expected_hash = "58b47971e3234c0117f153a4d3d7c7d0efdfb611804ba729153dfac19e503cfe"
    assert function[0].function_signature == "()"
    assert function[0].definition_sha256 == expected_hash
    assert function[0].audit_id == f"OMN-15361:tenant.safe_report:{expected_hash}"


def test_owner_manifest_can_express_every_observed_application_object_kind() -> None:
    assert EnumApplicationRelationKind.FOREIGN_TABLE.value == "foreign_table"
    assert {kind.value for kind in EnumApplicationDatabaseObjectKind}.issuperset(
        {
            "function",
            "aggregate",
            "window_function",
            "procedure",
            "sequence",
            "extension",
            "type",
            "base_type",
            "range_type",
            "multirange_type",
        }
    )
    assert {
        "tenant_identity_column",
        "identity_root_contract",
        "identity_root_control_role",
        "identity_root_control_operations",
        "canonical_policy_name",
        "deduplication_key_columns",
        "authorization_dependency_columns",
        "write_eligibility_dependency_columns",
    }.issubset(ModelRelationEvidence.model_fields)
    assert {"audit_id", "definition_sha256"}.issubset(
        ModelDatabaseObjectEvidence.model_fields
    )


def test_contract_derived_table_grant_gate_is_wired_and_strict() -> None:
    """OMN-15656: the TABLE-grant derivation gate must run in the STRICT job.

    ``deploy-gate`` never saw the 43/43 strict-wiring failure because nothing
    resolved real node contracts against the real shipped topology. This asserts
    the gate exists, consumes the pinned cross-repo checkout, and runs both
    directions of the drift check.
    """
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")

    assert (
        "- name: Enforce contract-derived application-database TABLE grants" in workflow
    )
    step = workflow.split(
        "- name: Enforce contract-derived application-database TABLE grants",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]
    assert "scripts/generate_application_database_table_grants.py" in step
    assert ".proof-dependencies/omnimarket/src/omnimarket/nodes" in step
    # --check catches grant/contract drift; --prove catches a grant that exists
    # but does not actually satisfy the wiring validator.
    assert "--check --prove" in step

    gate_module = (_ROOT / "scripts" / "ci" / "ci_summary_gate.py").read_text(
        encoding="utf-8"
    )
    assert '"Application Database Domain Enforcement (OMN-15361)"' in gate_module
