# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14108: omnibase_infra's fleet lists no longer target omnibase_compat.

omnibase_compat is retired (RULING 2026-10-22T21:43:04Z). These repository
lists named it as one more repository to clone, sweep or audit:

* ``scripts/pull-all.sh`` and ``scripts/lib/merge_sweep_common.sh`` ``REPOS``
* ``scripts/audit-merge-method-on-main.sh`` and
  ``scripts/verify-merge-queue-no-bypass.sh`` ``REPOS``

The CI mirror list (``docker/runners/git-mirror-refresh.sh``) keeps it until
the workflows that still check compat out stop doing so.

Falsifier: put the repository back in any one of them and the matching test
names it.
"""

from __future__ import annotations

import pytest

from tests.ci.test_no_occ_fleet_lists_omn17427 import _bash_array

pytestmark = pytest.mark.unit

COMPAT = "omnibase_compat"


@pytest.mark.parametrize(
    "relpath",
    [
        "scripts/pull-all.sh",
        "scripts/lib/merge_sweep_common.sh",
        "scripts/audit-merge-method-on-main.sh",
        "scripts/verify-merge-queue-no-bypass.sh",
    ],
)
def test_fleet_repo_list_omits_compat(relpath: str) -> None:
    repos = _bash_array(relpath, "REPOS")
    assert repos, f"{relpath}: REPOS is empty, so the check would prove nothing"
    assert COMPAT not in repos, f"{relpath} REPOS still names {COMPAT}"
