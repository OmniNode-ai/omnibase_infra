# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Expose the host-port suite at the OMN-19497 acceptance-command path.

Keep the original module available for contracts that already cite it. Explicit
re-exports let pytest collect the same assertions and unit marker at either path.
"""

from tests.unit.infra.test_catalog_duplicate_host_ports import (
    pytestmark,
    test_bundle_list_is_not_empty,
    test_duplicate_host_port_cli_generate_refuses_before_writing,
    test_duplicate_host_port_distinct_ports_pass,
    test_duplicate_host_port_planted_clash_is_reported,
    test_no_duplicate_host_port_in_resolved_bundle,
)

__all__ = [
    "pytestmark",
    "test_bundle_list_is_not_empty",
    "test_duplicate_host_port_cli_generate_refuses_before_writing",
    "test_duplicate_host_port_distinct_ports_pass",
    "test_duplicate_host_port_planted_clash_is_reported",
    "test_no_duplicate_host_port_in_resolved_bundle",
]
