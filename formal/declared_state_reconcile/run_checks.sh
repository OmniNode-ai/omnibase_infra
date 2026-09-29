#!/usr/bin/env bash
# Run TLC on the model and every mutation. Usage: run_checks.sh <outdir>
# Needs TLA2TOOLS_JAR (path to tla2tools.jar) and either `java` or docker (lab: .201).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: run_checks.sh <outdir>}"
JAR="${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to tla2tools.jar}"
mkdir -p "$OUT"
run_tlc() {
  local cfg="$1"
  if command -v java >/dev/null 2>&1; then
    (cd "$HERE" && java -cp "$JAR" tlc2.TLC -config "$cfg" -workers 2 -metadir "$(mktemp -d)" DeclaredStateReconcile.tla)
  else
    docker run --rm -v "$HERE":/m:ro -v "$JAR":/tla2tools.jar:ro -w /m eclipse-temurin:17-jre-alpine \
      java -cp /tla2tools.jar tlc2.TLC -config "$cfg" -workers 2 -metadir /tmp/tlcmeta DeclaredStateReconcile.tla
  fi
}
for cfg in Model.cfg mut_*.cfg; do
  run_tlc "$cfg" > "$OUT/${cfg%.cfg}.out" 2>&1
  echo "$cfg exit=$?"
done
( cd "$HERE" && cat DeclaredStateReconcile.tla *.cfg | shasum -a 256 | cut -d' ' -f1 ) > "$OUT/model.sha256"
