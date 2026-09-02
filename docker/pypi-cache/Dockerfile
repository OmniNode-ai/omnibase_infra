# PyPI pull-through cache (devpi-server) for the self-hosted runner fleet.
# Ticket: OMN-14027 (Layer-B egress caching) | Parent: OMN-13932
#
# STATUS: SHOVEL-READY / INERT. Nothing builds this image until an operator runs
# `docker compose -f docker/docker-compose.pypi-cache.yml up -d` at rollout, which
# is soak-gated (OMN-14027 is DESIGN-ONLY until the execution gate trips). See
# knowledge-base:runbooks/pypi-cache-egress-rollout.md.
#
# devpi's built-in `root/pypi` index is an on-demand pull-through mirror: the first
# runner to request a wheel fetches it from PyPI once; the cache then serves all 64
# runners from the LAN. This removes the redundant per-runner PyPI egress that
# saturates the single home uplink under concurrent merge_group load (OMN-14017).

# Base image FROZEN to a digest at rollout (runbook step 1), matching the
# runner-image reproducibility discipline. Digest resolved from the registry
# 2026-08-14 for tag python:3.12-slim.
#
# HONESTY NOTE: the cache image currently running on the runner host was built
# 2026-07-09, BEFORE this pin, and its base layer has since been pruned from the
# host — so the digest that image was actually built from is unrecoverable. This
# pin therefore governs the NEXT rebuild, and adopting it is a base-image change,
# not a no-op re-pin. Do not rebuild the cache purely to apply this pin: a
# rebuild is only safe alongside the warmed-corpus preservation described in the
# runbook's rollback section (the named volume survives a recreate; the image
# does not carry the corpus).
FROM python:3.12-slim@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65

# devpi versions FROZEN to the exact versions proven live in the running cache
# (readback 2026-08-14: `docker exec omninode-pypi-cache pip list` ->
# devpi-server 6.20.3, devpi-web 4.3.0, devpi-common 4.1.1). These are the
# versions that produced the 22G warmed corpus and the canary hit evidence, so
# pinning them exactly makes a rebuild reproduce a proven-good server rather
# than whatever the old range resolves to today.
ARG DEVPI_SERVER_SPEC="devpi-server==6.20.3"
ARG DEVPI_WEB_SPEC="devpi-web==4.3.0"

ENV DEVPI_SERVERDIR=/devpi/server \
    DEVPI_HOST=0.0.0.0 \
    DEVPI_PORT=3141

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "${DEVPI_SERVER_SPEC}" "${DEVPI_WEB_SPEC}"

RUN useradd --create-home --uid 1002 devpi \
    && mkdir -p /devpi \
    && chown -R devpi:devpi /devpi

COPY entrypoint.sh /usr/local/bin/pypi-cache-entrypoint.sh
RUN chmod +x /usr/local/bin/pypi-cache-entrypoint.sh

USER devpi
WORKDIR /devpi
EXPOSE 3141

ENTRYPOINT ["/usr/local/bin/pypi-cache-entrypoint.sh"]
