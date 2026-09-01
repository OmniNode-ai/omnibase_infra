# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex identity`` (OMN-17310 / OMN-17312, epic OMN-17306).

The headline case replays the real 2026-08-31T08:10:54Z probe (correlation
``b9cd305c-8f31-497a-b404-b75b45b98341``): a run whose orchestrator resolved out
of the operator's local venv, read as a statement about the ``.201`` dev lane.
The assertion must refuse it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_core.enums.enum_execution_locus_kind import EnumExecutionLocusKind
from omnibase_core.enums.enum_package_source_kind import EnumPackageSourceKind
from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_core.models.runtime.model_package_identity import ModelPackageIdentity
from omnibase_core.models.runtime.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra.cli.cli_identity import identity_group

_LANE_SHA = "2f123b4c01eabd2c51f7d703491e9cdf36f89bcd"
_LOCAL_SHA = "66b7131a3508bd2c51f7d703491e9cdf36f89bcd"


def _lane_manifest(tmp_path: Path, *, dirty: bool = False) -> Path:
    """The shape read live from omninode-runtime on .201, 2026-08-31."""
    path = tmp_path / "build-provenance.json"
    path.write_text(
        json.dumps(
            {
                "build_source": "workspace",
                "per_repo_vcs_provenance": {
                    "siblings": {
                        "omnimarket": {"vcs_ref": _LANE_SHA, "vcs_dirty": dirty}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _receipt(tmp_path: Path, *, commit: str | None) -> Path:
    identity = ModelRuntimeIdentity(
        host="operator-laptop",
        locus_kind=EnumExecutionLocusKind.VENV,
        execution_locus="/venvs/omnibase_infra",
        interpreter="/venvs/omnibase_infra/bin/python3.12",
        packages={
            "omnimarket": ModelPackageIdentity(
                name="omnimarket",
                version="0.4.11",
                commit=commit,
                source=(
                    EnumPackageSourceKind.VCS
                    if commit
                    else EnumPackageSourceKind.REGISTRY
                ),
            )
        },
        stamped_at=datetime(2026, 8, 31, 8, 10, 54, tzinfo=UTC),
    )
    receipt = ModelSkillResult[dict[str, str]](
        skill_name="delegate",
        node_name="node_delegate_skill_orchestrator",
        status=EnumSkillResultStatus.SUCCESS,
        correlation_id=uuid4(),
        run_id=uuid4(),
        exit_code=0,
        duration_ms=5120,
        result={"answer": "alive"},
        result_model="builtins.dict",
        runtime_identity=identity,
    )
    path = tmp_path / "receipt.json"
    path.write_text(receipt.model_dump_json(), encoding="utf-8")
    return path


@pytest.mark.unit
class TestStamp:
    def test_one_line_by_default(self) -> None:
        result = CliRunner().invoke(identity_group, ["stamp"])
        assert result.exit_code == 0
        assert result.output.startswith("identity: ")

    def test_json_round_trips_through_the_core_model(self) -> None:
        result = CliRunner().invoke(identity_group, ["stamp", "--json"])
        assert result.exit_code == 0
        ModelRuntimeIdentity.model_validate_json(result.output)


@pytest.mark.unit
class TestAssertTarget:
    def test_local_venv_receipt_against_the_lane_is_refused(
        self, tmp_path: Path
    ) -> None:
        """THE case: a laptop-local run claiming to prove the .201 dev lane."""
        result = CliRunner().invoke(
            identity_group,
            [
                "assert-target",
                "--from-build-provenance",
                str(_lane_manifest(tmp_path)),
                "--target-name",
                "dev lane (.201)",
                "--receipt",
                str(_receipt(tmp_path, commit=_LOCAL_SHA)),
            ],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output

    def test_receipt_with_no_commit_is_refused_as_unknown(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            identity_group,
            [
                "assert-target",
                "--from-build-provenance",
                str(_lane_manifest(tmp_path)),
                "--target-name",
                "dev lane (.201)",
                "--receipt",
                str(_receipt(tmp_path, commit=None)),
            ],
        )
        assert result.exit_code == 1
        assert "unknown" in result.output
        assert "not evidence of content" in result.output

    def test_matching_receipt_passes_and_names_what_was_compared(
        self, tmp_path: Path
    ) -> None:
        result = CliRunner().invoke(
            identity_group,
            [
                "assert-target",
                "--from-build-provenance",
                str(_lane_manifest(tmp_path)),
                "--target-name",
                "dev lane (.201)",
                "--receipt",
                str(_receipt(tmp_path, commit=_LANE_SHA)),
            ],
        )
        assert result.exit_code == 0
        assert "package:omnimarket" in result.output

    def test_all_dirty_manifest_is_refused_before_any_comparison(
        self, tmp_path: Path
    ) -> None:
        """A dirty tree cannot declare a commit, so it declares nothing."""
        result = CliRunner().invoke(
            identity_group,
            [
                "assert-target",
                "--from-build-provenance",
                str(_lane_manifest(tmp_path, dirty=True)),
                "--target-name",
                "dev lane (.201)",
                "--receipt",
                str(_receipt(tmp_path, commit=_LANE_SHA)),
            ],
        )
        assert result.exit_code != 0
        assert "declares no clean sibling commits" in result.output

    def test_unstamped_receipt_cannot_prove_a_target(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.json"
        legacy.write_text(
            json.dumps(
                {
                    "skill_name": "delegate",
                    "node_name": "node_delegate_skill_orchestrator",
                    "status": "success",
                    "correlation_id": str(uuid4()),
                    "run_id": str(uuid4()),
                    "exit_code": 0,
                    "duration_ms": 1,
                    "result": {},
                    "result_model": "builtins.dict",
                    "schema_version": {"major": 1, "minor": 0, "patch": 0},
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            identity_group,
            [
                "assert-target",
                "--from-build-provenance",
                str(_lane_manifest(tmp_path)),
                "--target-name",
                "dev lane (.201)",
                "--receipt",
                str(legacy),
            ],
        )
        assert result.exit_code != 0
        assert "carries no runtime_identity block" in result.output

    def test_exactly_one_declaration_source_is_required(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(identity_group, ["assert-target"])
        assert result.exit_code != 0
        assert "exactly one of --declared or --from-build-provenance" in result.output
