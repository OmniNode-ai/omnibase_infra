# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-16449: scoped MSK-bastion DNS-override sidecar for the .201
# gateway-forwarder container. See dnsmasq.conf for the full rationale and the
# operator ruling (2026-08-24) that authorizes this as an interim bridge with
# an explicit sunset -- delete this whole directory + its compose wiring once
# the forwarder's direct-MSK leg is replaced by the HTTPS-ingest path.
#
# Purpose-built rather than a third-party dnsmasq image: this repo already
# has a policy of controlling its own runtime images end to end (see
# docker/dns-cache/Dockerfile's precedent), and this sidecar's whole job is
# resolving 6 hostnames to one operator-supplied IP -- no benefit to trusting
# an external image for that.

FROM alpine:3.20

RUN apk add --no-cache dnsmasq bind-tools

COPY dnsmasq.conf /etc/dnsmasq.conf.template
COPY entrypoint.sh /usr/local/bin/dns-bastion-entrypoint.sh
RUN chmod +x /usr/local/bin/dns-bastion-entrypoint.sh

EXPOSE 53/udp 53/tcp

ENTRYPOINT ["/usr/local/bin/dns-bastion-entrypoint.sh"]
