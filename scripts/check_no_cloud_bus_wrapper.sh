#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

if [[ -z "${OMNI_HOME:-}" ]]; then
  echo "SKIP: OMNI_HOME is not set" >&2
  exit 0
fi

CHECK_SCRIPT="$OMNI_HOME/scripts/check_no_cloud_bus.sh"
if [[ ! -f "$CHECK_SCRIPT" ]]; then
  echo "SKIP: check_no_cloud_bus.sh not found at OMNI_HOME=$OMNI_HOME" >&2
  exit 0
fi
exec bash "$CHECK_SCRIPT" "$PWD"
