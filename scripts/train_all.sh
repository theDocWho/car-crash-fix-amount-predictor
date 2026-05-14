#!/usr/bin/env bash
#
# Full Variant A + B training sequence, ~3.5–4 hrs unattended.
#
# Steps (sequential on a single MPS device):
#   1. Stanford Cars identifier (~50 min)
#   2. CarDD damage classifier (~25 min)
#   3. YOLOv8n detector       (~90–120 min)
#   4. Promote classifier + detector
#   5. Extract features + synth targets + bbox features
#   6. Train XGBoost(A) and XGBoost(B), promote both
#
# All logs written to logs/train_all_<ts>.log; resumable runs land in
# checkpoints/<variant>/run_*  (see registry.json).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

# macOS environment essentials
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export DYLD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"

mkdir -p logs
TS="$(date -u +%Y-%m-%dT%H-%M-%S)"
LOG="logs/train_all_${TS}.log"

# Per-stage logs (easier to tail than the consolidated one)
LOG_ID="logs/train_identifier_${TS}.log"
LOG_CLS="logs/train_classifier_${TS}.log"
LOG_DET="logs/train_detector_${TS}.log"

banner() {
    echo
    echo "=========================================================="
    echo "[$(date -u +%H:%M:%S)] $*"
    echo "=========================================================="
}

# ---- Stage 1: Stanford Cars identifier (~50 min) ---------------------
banner "Stage 1/6  Stanford Cars identifier" | tee -a "$LOG"
ccdp train identifier \
    --epochs-stage1 3 --epochs-stage2 12 \
    --batch-size 32 --num-workers 4 \
    --tag identifier_v1 2>&1 | tee "$LOG_ID" >>"$LOG"

# ---- Stage 2: CarDD damage classifier (~25 min) ----------------------
banner "Stage 2/6  CarDD damage classifier" | tee -a "$LOG"
ccdp train classifier \
    --epochs-stage1 3 --epochs-stage2 12 \
    --batch-size 32 --num-workers 4 \
    --tag classifier_v1 2>&1 | tee "$LOG_CLS" >>"$LOG"

# ---- Stage 3: YOLOv8n detector (~90–120 min) -------------------------
banner "Stage 3/6  YOLOv8n detector" | tee -a "$LOG"
ccdp train detector \
    --epochs 50 --batch 16 --imgsz 640 --workers 4 \
    --tag yolov8n_v1 2>&1 | tee "$LOG_DET" >>"$LOG"

# ---- Stage 4: Promote latest classifier + detector + identifier ------
banner "Stage 4/6  Promoting identifier / classifier / detector" | tee -a "$LOG"
latest_run_id() {
    # newest run dir under checkpoints/<variant>/  (strip the run_ prefix)
    local variant="$1"
    local dir
    dir="$(ls -td "checkpoints/${variant}"/run_* 2>/dev/null | head -1)"
    [[ -n "$dir" ]] || { echo "no runs for $variant" >&2; return 1; }
    basename "$dir" | sed 's/^run_//'
}

ID_RUN=$(latest_run_id identifier)
CLS_RUN=$(latest_run_id classifier)
DET_RUN=$(latest_run_id detector)

echo "  identifier -> $ID_RUN"   | tee -a "$LOG"
echo "  classifier -> $CLS_RUN"  | tee -a "$LOG"
echo "  detector   -> $DET_RUN"  | tee -a "$LOG"
ccdp registry promote "$ID_RUN"  identifier 2>&1 | tee -a "$LOG"
ccdp registry promote "$CLS_RUN" classifier 2>&1 | tee -a "$LOG"
ccdp registry promote "$DET_RUN" detector   2>&1 | tee -a "$LOG"

# ---- Stage 5: Feature extraction + cost targets + bbox features ------
banner "Stage 5/6  Features, targets, bbox features" | tee -a "$LOG"
ccdp train extract-features                                 2>&1 | tee -a "$LOG"
ccdp train synth-targets                                    2>&1 | tee -a "$LOG"
ccdp train extract-bbox-features                            2>&1 | tee -a "$LOG"

# ---- Stage 6: XGBoost(A) + XGBoost(B), promote both ------------------
banner "Stage 6/6  XGBoost A & B" | tee -a "$LOG"
ccdp train xgb --variant a --n-estimators 600 --max-depth 7 \
    --learning-rate 0.05 --tag xgb_a_v1 2>&1 | tee -a "$LOG"
ccdp train xgb --variant b --n-estimators 600 --max-depth 7 \
    --learning-rate 0.05 --tag xgb_b_v1 2>&1 | tee -a "$LOG"

XGB_A_RUN=$(latest_run_id xgb_a)
XGB_B_RUN=$(latest_run_id xgb_b)
echo "  xgb_a -> $XGB_A_RUN"  | tee -a "$LOG"
echo "  xgb_b -> $XGB_B_RUN"  | tee -a "$LOG"
ccdp registry promote "$XGB_A_RUN" xgb_a 2>&1 | tee -a "$LOG"
ccdp registry promote "$XGB_B_RUN" xgb_b 2>&1 | tee -a "$LOG"

banner "DONE  see registry: ccdp registry list" | tee -a "$LOG"
ccdp registry list 2>&1 | tee -a "$LOG"
