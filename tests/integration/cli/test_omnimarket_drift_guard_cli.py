# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI integration coverage for the OMN-18280 drift diagnostic boundary."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click
import pytest

from omnibase_infra.cli.cli_node import run_node_by_name

_FAKE_SHA = "a" * 40


def test_node_cli_keeps_drift_refusal_when_path_resolution_fails(
    tmp_path: Path,
) -> None:
    """The CLI surfaces the guard refusal before resolving a node contract."""
    with (
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
            return_value=None,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
            return_value=_FAKE_SHA,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.shutil.which",
            side_effect=OSError("unreadable PATH"),
        ),
    ):
        with pytest.raises(click.ClickException, match="PATH did not resolve"):
            run_node_by_name.callback(
                "not-reached",
                contract_path=None,
                input_path=None,
                state_root=tmp_path,
                backend=(),
                timeout=1,
                verbose=False,
                output_mode="default",
                emit_socket=None,
                omni_home=tmp_path,
                allow_omnimarket_drift=False,
            )
