# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden and error chains of node_venv_atomic_install through its typed handler (OMN-22712).

The chains run on a real ``uv`` venv with wheels built in the test, so nothing
needs a network: stage the moved package, exchange it in, finalize (golden), and
a refused downgrade or an unresolvable requirement that leaves the live venv
byte for byte as it was (error).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.handlers.venv_atomic_install.handler_venv_atomic_install import (
    HandlerVenvAtomicInstall,
)
from omnibase_infra.models.model_venv_atomic_install_request import (
    ModelVenvAtomicInstallRequest,
)
from tests.scripts.test_venv_atomic_install_omn22712 import _Venv

pytestmark = pytest.mark.integration


@pytest.fixture
def venv(tmp_path: Path) -> _Venv:
    return _Venv(tmp_path)


def _request(
    venv: _Venv, command: Any, *requirements: str
) -> ModelVenvAtomicInstallRequest:
    return ModelVenvAtomicInstallRequest(
        command=command,
        python_bin=Path(venv.python),
        stage_dir=venv.stage,
        requirements=requirements,
        no_deps=True,
    )


def test_golden_chain_stage_commit_finalize_moves_only_the_moved_package(
    venv: _Venv,
) -> None:
    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    before = venv.snapshot()
    assert venv.marker() == "old"

    staged = handler.handle(_request(venv, "stage", str(venv.new)))
    assert staged.status == "ok", staged
    assert venv.snapshot() == before, "staging wrote into the live venv"

    committed = handler.handle(_request(venv, "commit"))
    assert committed.status == "ok", committed
    assert venv.marker() == "new"
    assert venv.script() == "new"

    after = venv.snapshot()
    untouched = [
        p
        for p in before
        if p.startswith(("lib/python3", "bin")) and "moved_pkg" not in p
    ]
    assert untouched, "the venv listing held no bystander paths to compare"
    assert all(after.get(p) == before[p] for p in untouched if p in after), (
        "a package that did not move changed"
    )

    done = handler.handle(_request(venv, "finalize"))
    assert done.status == "ok" and not venv.stage.exists(), done


def test_error_chain_a_downgrade_is_refused_and_the_venv_is_unchanged(
    venv: _Venv,
) -> None:
    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    assert handler.handle(_request(venv, "stage", str(venv.new))).status == "ok"
    assert handler.handle(_request(venv, "commit")).status == "ok"
    handler.handle(_request(venv, "finalize"))
    before = venv.snapshot()

    refused = handler.handle(_request(venv, "stage", str(venv.old)))

    assert refused.status in {"refused", "failed"}, refused
    assert refused.exit_code != 0
    assert venv.snapshot() == before, "a refused downgrade changed the venv"
    assert venv.marker() == "new"


def test_error_chain_an_unresolvable_requirement_stages_nothing(venv: _Venv) -> None:
    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    before = venv.snapshot()

    failed = handler.handle(
        _request(venv, "stage", str(venv.wheels / "no_such_pkg-9.9-py3-none-any.whl"))
    )

    assert failed.status in {"refused", "failed"}, failed
    assert venv.snapshot() == before, "a failed stage changed the venv"
    assert venv.marker() == "old"
