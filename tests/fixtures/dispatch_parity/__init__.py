# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12548 dispatch-selection parity fixtures + regeneration harness.

This package holds the CI-consumed committed fixture (``baseline-selection-v2.json``)
and the harness (:mod:`harness`) that regenerates it. The canonical location for
OMN-12525 evidence is ``$OMNI_HOME/docs/evidence/OMN-12548/``; the fixture lives
here (inside ``omnibase_infra``) so the parity test can read it at CI time without
a cross-repo path. A pointer file in the omni_home evidence dir cites this copy.
"""
