#!/bin/bash
#SBATCH --job-name=lig-wholeblur
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wholeblur_%j.out
#
# Whole-structure guidance with blur (8a2b + KY9): uniform dB=150, two-zone, baselines.
#   calibration:  MODE=calibrate sbatch --export=ALL run_whole_blur.sbatch
#   runs:         MODE=run LAMBDA=<x|balance> sbatch --export=ALL run_whole_blur.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_env.sh"
cd "$REPO"
echo "=== $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) MODE=${MODE:-run} LAMBDA=${LAMBDA:-1} ==="
NRUNS=4
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$EXP/wandb_offline_wholeblur
mkdir -p "$WANDB_DIR"
$PYTHON -u "$EXP/run_whole_blur.py"
if [ "${MODE:-run}" = run ] && [ "$WANDB_MODE" = offline ]; then
  export WANDB_API_KEY=$(wandb_key)
  $WANDB_PY -m wandb sync --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" $(ls -d $WANDB_DIR/wandb/offline-run-* | tail -$NRUNS)
  unset WANDB_API_KEY
fi
echo "=== done ==="
