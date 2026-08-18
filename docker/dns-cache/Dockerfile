# Local caching DNS resolver (unbound) for the self-hosted runner fleet.
# Ticket: OMN-15736 (scaling precondition #1) | Capacity probe: [mergesweep-0808-capacity]
#
# STATUS: SHOVEL-READY / INERT. Nothing builds this image until an operator runs
# `docker compose -f docker/docker-compose.dns-cache.yml up -d` at rollout, which
# is operator-gated exactly like the OMN-14027 pypi-cache pattern. See
# docs/runbooks/runner-dns-cache-rollout.md.
#
# Why unbound (not dnsmasq): the fleet's DNS failure class (OMN-15733,
# `files.pythonhosted.org` resolution failures under a 64+ concurrent cold-start
# burst) needs fine-grained negative-caching control — a short NXDOMAIN/SERVFAIL
# TTL floor that absorbs retry storms without masking a real upstream outage for
# longer than that floor. unbound exposes `cache-max-negative-ttl` and
# `cache-min-ttl` directly in unbound.conf; dnsmasq's negative-caching is
# all-or-nothing (`no-negcache`) with no TTL-floor knob. unbound also ships a
# built-in remote-control stats interface (`unbound-control stats_noreset`) that
# yields a real hit/miss counter without an extra sidecar — dnsmasq has no
# built-in stats output at all (metrics require external log-scraping of every
# query). Both are widely used, actively maintained, permissively licensed
# (BSD); unbound is the more "production resolver" of the two and matches the
# devpi pull-through cache's posture of being a real server, not a shell script
# wrapping a simpler tool.

FROM alpine:3.20

# Pin the package index snapshot implicitly via the base image tag; freeze to a
# `alpine:3.20@sha256:...` digest at rollout (same reproducibility discipline as
# docker/pypi-cache/Dockerfile) once the target digest is confirmed against the
# rollout host's package mirror.
RUN apk add --no-cache unbound bind-tools \
    && mkdir -p /etc/unbound /var/log/unbound \
    && chown -R unbound:unbound /var/log/unbound

COPY unbound.conf /etc/unbound/unbound.conf
COPY entrypoint.sh /usr/local/bin/dns-cache-entrypoint.sh
RUN chmod +x /usr/local/bin/dns-cache-entrypoint.sh

EXPOSE 53/udp 53/tcp
# unbound-control remote interface (loopback-only inside the container; not
# published to the host — see docker-compose.dns-cache.yml).
EXPOSE 8953/tcp

ENTRYPOINT ["/usr/local/bin/dns-cache-entrypoint.sh"]
