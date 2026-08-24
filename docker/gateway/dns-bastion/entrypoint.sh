#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-16449: substitutes the operator-supplied GATEWAY_BASTION_IP into a
# runtime-only copy of dnsmasq.conf (the tracked file never carries the
# literal), then execs dnsmasq in the foreground.
set -eu

if [ -z "${GATEWAY_BASTION_IP:-}" ]; then
    echo "[dns-bastion] FATAL: GATEWAY_BASTION_IP is not set. Refusing to start" \
        "with no bastion target -- this sidecar exists to resolve exactly one" \
        "address and has no safe default." >&2
    exit 1
fi

# Basic IPv4-shape sanity check, no network validation -- fail loud on an
# obviously wrong value (empty, hostname, typo) rather than let dnsmasq start
# with a config that resolves the MSK brokers nowhere useful.
case "$GATEWAY_BASTION_IP" in
    [0-9]*.[0-9]*.[0-9]*.[0-9]*) ;;
    *)
        echo "[dns-bastion] FATAL: GATEWAY_BASTION_IP ('${GATEWAY_BASTION_IP}')" \
            "does not look like an IPv4 address." >&2
        exit 1
        ;;
esac

sed "s/__GATEWAY_BASTION_IP__/${GATEWAY_BASTION_IP}/g" \
    /etc/dnsmasq.conf.template > /etc/dnsmasq.conf

exec dnsmasq --keep-in-foreground --conf-file=/etc/dnsmasq.conf
