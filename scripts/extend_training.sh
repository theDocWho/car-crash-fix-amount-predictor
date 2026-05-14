#!/usr/bin/env bash
#
# Extend identifier + detector training without interruption.
#
# Stage 1: resume identifier ~10 more epochs at lower LR  (~30 min)
# Stage 2: fresh detector v2, full 50 epochs uninterrupted (~120 min)
# Stage 3: re-extract bbox features via the new detector
# Stage 4: retrain xgb_b on the new bbox features
# Stage 5: promote everything that improved
#
# Total wall-clock target: ~3 hours.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export DYLD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"

mkdir -p logs
TS="$(date -u +%Y-%m-%dT%H-%M-%S)"
LOG="logs/extend_${TS}.log"
LOG_ID="logs/extend_identifier_${TS}.log"
LOG_DET="logs/extend_detector_${TS}.log"

banner() {
    echo
    echo "=========================================================="
    echo "[$(date -u +%H:%M:%S)] $*"
    echo "=========================================================="
}

latest_run_id() {
    local variant="$1"
    local dir
    dir="$(ls -td "checkpoints/${variant}"/run_* 2>/dev/null | head -1)"
    [[ -n "$dir" ]] || { echo "no runs for $variant" >&2; return 1; }
    basename "$dir" | sed 's/^run_//'
}

# ---- Stage 1: resume identifier with lower LR -----------------------------
banner "Stage 1/5  Resume identifier (10 more epochs, lower LR)" | tee -a "$LOG"
ID_LAST="checkpoints/identifier/run_2026-05-12T20-20-52_identifier_v1/last.pt"
if [[ ! -f "$ID_LAST" ]]; then
    echo "ERROR: identifier last.pt missing at $ID_LAST" | tee -a "$LOG"
    exit 1
fi
# epochs_stage1=3 + epochs_stage2=25 -> 28 total; resume picks up at epoch 16
ccdp train identifier \
    --epochs-stage1 3 --epochs-stage2 25 \
    --lr-stage2 5e-5 \
    --batch-size 32 --num-workers 4 \
    --resume "$ID_LAST" \
    2>&1 | tee "$LOG_ID" >>"$LOG"

# ---- Stage 2: full detector v2 (50 epochs uninterrupted) ------------------
banner "Stage 2/5  Detector v2  (50 epochs, imgsz 640, batch 16)" | tee -a "$LOG"
ccdp train detector \
    --epochs 50 --batch 16 --imgsz 640 --workers 4 \
    --tag yolov8n_v2 \
    2>&1 | tee "$LOG_DET" >>"$LOG"

# ---- Stage 3: promote both ------------------------------------------------
banner "Stage 3/5  Promote identifier + detector v2" | tee -a "$LOG"
ID_RUN=$(latest_run_id identifier)
DET_RUN=$(latest_run_id detector)
echo "  identifier -> $ID_RUN"  | tee -a "$LOG"
echo "  detector   -> $DET_RUN" | tee -a "$LOG"
ccdp registry promote "$ID_RUN"  identifier 2>&1 | tee -a "$LOG"
ccdp registry promote "$DET_RUN" detector   2>&1 | tee -a "$LOG"

# ---- Stage 4: re-extract bbox features with the new detector --------------
banner "Stage 4/5  Re-extract bbox features via new detector" | tee -a "$LOG"
ccdp train extract-bbox-features 2>&1 | tee -a "$LOG"

# ---- Stage 5: retrain xgb_b + promote -------------------------------------
banner "Stage 5/5  Retrain xgb_b on new bbox features" | tee -a "$LOG"
ccdp train xgb --variant b \
    --n-estimators 600 --max-depth 7 --learning-rate 0.05 \
    --tag xgb_b_v2 \
    2>&1 | tee -a "$LOG"
XGB_B_RUN=$(latest_run_id xgb_b)
echo "  xgb_b -> $XGB_B_RUN" | tee -a "$LOG"
ccdp registry promote "$XGB_B_RUN" xgb_b 2>&1 | tee -a "$LOG"

banner "DONE  see registry: ccdp registry list" | tee -a "$LOG"
ccdp registry list 2>&1 | tee -a "$LOG"
