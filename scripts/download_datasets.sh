#!/usr/bin/env bash
#
# Download all Phase 1 datasets via the Kaggle CLI.
#
# Auth: relies on ~/.kaggle/access_token (new format) or ~/.kaggle/kaggle.json (legacy).
# Skips datasets that already have content under data/raw/<slug>/.
#
# Usage:
#   bash scripts/download_datasets.sh                   # primary datasets only
#   STANFORD_CARS=1 bash scripts/download_datasets.sh   # also pull Stanford Cars (Phase 1.5)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/data/raw"
LOG="$REPO_ROOT/data/raw/_download.log"
mkdir -p "$DEST"
: >"$LOG"

SLUGS=(
  # Primary damage-recognition training corpus (CarDD: 6 damage types, COCO seg)
  "nasimetemadi/car-damage-detection"
  # Auxiliary head: front/rear x condition labels
  "samwash94/comprehensive-car-damage-detection"
  # Car-metadata distribution source (cost columns are paywalled in free sample)
  "rebrowser/iaai-dataset"
  # Phase 1.5 make/model/year identifier training (Krause et al. 2013, 196 classes)
  "eduardo4jesus/stanford-cars-dataset"
)

# NOTE: ganeshsura/car-damage-detection-and-cost-estimation is intentionally
# excluded. Its CSV/image join is broken (hashed Roboflow filenames vs imgNNN
# CSV keys) and est_cost is combinatorially synthetic. See CITATIONS.md.

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# "Extracted" = the dir holds something other than a leftover *.zip or _* log file.
# (Kaggle's --unzip occasionally leaves the archive un-extracted — e.g. if it was
# interrupted — and the old presence check would then skip it forever.)
has_extracted() {
  [[ -d "$1" ]] && [[ -n "$(ls -A "$1" 2>/dev/null | grep -vE '^_|\.zip$' || true)" ]]
}

# Extract any leftover *.zip in $dir, then delete the archive on success.
extract_leftover_zips() {
  local dir="$1" z
  shopt -s nullglob
  for z in "$dir"/*.zip; do
    log "unzip  $(basename "$z")  (extracting manually)"
    if command -v unzip >/dev/null 2>&1 && unzip -q -o "$z" -d "$dir" >>"$LOG" 2>&1; then
      rm -f "$z"
    else
      log "FAIL   unzip $(basename "$z") -- see $LOG (is 'unzip' installed?)"
    fi
  done
  shopt -u nullglob
}

for slug in "${SLUGS[@]}"; do
  name="${slug##*/}"
  dir="$DEST/$name"
  if has_extracted "$dir"; then
    log "skip   $slug  (already extracted at $dir)"
    continue
  fi
  mkdir -p "$dir"
  log "fetch  $slug  ->  $dir"
  kaggle datasets download -d "$slug" -p "$dir" --unzip >>"$LOG" 2>&1 || log "warn   kaggle download returned non-zero for $slug"
  # Safety net: if --unzip left the archive behind, extract it ourselves.
  extract_leftover_zips "$dir"
  if has_extracted "$dir"; then
    log "ok     $slug  ($(du -sh "$dir" | cut -f1))"
  else
    log "FAIL   $slug  -- nothing extracted; see $LOG"
  fi
done

log "done.  totals:"
du -sh "$DEST"/*/ 2>/dev/null | tee -a "$LOG"
