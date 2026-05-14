#!/usr/bin/env bash
#
# Extend identifier v2 + train YOLOv8s detector + retrain xgb_b downstream,
# all uninterrupted. Total wall-clock: ~3.5–4.5 hrs.
#
# Stage 1: Resume identifier v2, +10 more stage-2 epochs       (~25 min)
# Stage 2: Train YOLOv8s detector from scratch (50 epochs)     (~180 min)
# Stage 3: Promote identifier v3 + detector (if best.pt valid)  (seconds)
# Stage 4: Re-extract bbox features via YOLOv8s                 (~10 min)
# Stage 5: Retrain xgb_b on the new bbox features + promote     (~3 min)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export DYLD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"

mkdir -p logs
TS="$(date -u +%Y-%m-%dT%H-%M-%S)"
LOG="logs/extend2_${TS}.log"
LOG_ID="logs/identifier_v3_${TS}.log"
LOG_DET="logs/yolov8s_${TS}.log"

banner() {
    echo
    echo "=========================================================="
    echo "[$(date -u +%H:%M:%S)] $*"
    echo "=========================================================="
}

latest_run_id() {
    local variant="$1" suffix="${2:-}"
    local dir
    dir="$(ls -td "checkpoints/${variant}"/run_*"${suffix}" 2>/dev/null | head -1)"
    [[ -n "$dir" ]] || { echo "no runs for $variant$suffix" >&2; return 1; }
    basename "$dir" | sed 's/^run_//'
}

# ---- Stage 1: Extend identifier v2 with +10 more stage-2 epochs ---------
banner "Stage 1/5  Resume identifier v2 (epochs_stage2 12 -> 22)" | tee -a "$LOG"
ID_LAST="checkpoints/identifier/run_2026-05-13T14-30-04_identifier_v2/last.pt"
if [[ ! -f "$ID_LAST" ]]; then
    echo "ERROR: identifier v2 last.pt missing at $ID_LAST" | tee -a "$LOG"
    exit 1
fi
# v2 had 3+12 epochs; extend to 3+22 = 25 total (10 more stage-2 epochs)
ccdp train identifier \
    --epochs-stage1 3 --epochs-stage2 22 \
    --batch-size 32 --num-workers 4 \
    --resume "$ID_LAST" \
    2>&1 | tee "$LOG_ID" >>"$LOG"

# ---- Stage 2: YOLOv8s detector (full 50 epochs) -------------------------
banner "Stage 2/5  YOLOv8s detector (50 epochs, imgsz 640)" | tee -a "$LOG"
# batch=12 is the safe value for v8s on a 16GB MPS box; v8n fit at batch=16
ccdp train detector \
    --model yolov8s.pt \
    --epochs 50 --batch 12 --imgsz 640 --workers 4 \
    --tag yolov8s_v1 \
    2>&1 | tee "$LOG_DET" >>"$LOG"

# ---- Stage 3: Promote identifier + detector ------------------------------
banner "Stage 3/5  Promote identifier + detector" | tee -a "$LOG"
ID_RUN="$(latest_run_id identifier _identifier_v2)"  # resume writes back to same run dir
DET_RUN="$(latest_run_id detector _yolov8s_v1)"

# Ensure YOLOv8s best.pt symlink exists (resume defends against earlier bug)
DET_DIR="checkpoints/detector/run_${DET_RUN}"
if [[ ! -e "$DET_DIR/best.pt" ]] && [[ -e "$DET_DIR/ultralytics/weights/best.pt" ]]; then
    ln -sf ultralytics/weights/best.pt "$DET_DIR/best.pt"
    ln -sf ultralytics/weights/last.pt "$DET_DIR/last.pt"
fi

echo "  identifier -> $ID_RUN"  | tee -a "$LOG"
echo "  detector   -> $DET_RUN" | tee -a "$LOG"
ccdp registry promote "$ID_RUN"  identifier 2>&1 | tee -a "$LOG"
ccdp registry promote "$DET_RUN" detector   2>&1 | tee -a "$LOG"

# ---- Stage 4: Re-extract bbox features via YOLOv8s ----------------------
banner "Stage 4/5  Re-extract bbox features via YOLOv8s" | tee -a "$LOG"
ccdp train extract-bbox-features 2>&1 | tee -a "$LOG"

# ---- Stage 5: Retrain xgb_b on new bbox features + promote --------------
banner "Stage 5/5  Retrain xgb_b on YOLOv8s bbox features" | tee -a "$LOG"
ccdp train xgb --variant b \
    --n-estimators 600 --max-depth 7 --learning-rate 0.05 \
    --tag xgb_b_v3 \
    2>&1 | tee -a "$LOG"
XGB_B_RUN="$(latest_run_id xgb_b _xgb_b_v3)"
echo "  xgb_b -> $XGB_B_RUN" | tee -a "$LOG"
ccdp registry promote "$XGB_B_RUN" xgb_b 2>&1 | tee -a "$LOG"

banner "DONE  see registry: ccdp registry list" | tee -a "$LOG"
ccdp registry list 2>&1 | tee -a "$LOG"
