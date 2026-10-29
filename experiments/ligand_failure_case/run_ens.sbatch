#!/bin/bash
#SBATCH --job-name=lig-ens
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ens_%j.out
#
# Ensemble loss: one loss on the map averaged over the 5 samples (w_L = w_P = 1) vs independent losses.
#   OUT_DIR, WANDB_GROUP and CONDS are set below.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_env.sh"
cd "$REPO"
echo "=== $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) MODE=${MODE:-run} LAMBDA=${LAMBDA:-1} ==="
export OUT_DIR=outputs_whole_blur_ens WANDB_GROUP=8a2b_whole_blur_ens
export CONDS=${CONDS:-whole_dB150_wL1.0,whole_dB150_ens_wL1.0}
NRUNS=$(echo "$CONDS" | tr "," "\n" | grep -c .)
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$EXP/wandb_offline_ens
mkdir -p "$WANDB_DIR"
$PYTHON -u "$EXP/run_whole_blur.py"
if [ "${MODE:-run}" = run ] && [ "$WANDB_MODE" = offline ]; then
  export WANDB_API_KEY=$(wandb_key)
  $WANDB_PY -m wandb sync --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" $(ls -d $WANDB_DIR/wandb/offline-run-* | tail -$NRUNS)
  unset WANDB_API_KEY
fi
echo "=== done ==="
