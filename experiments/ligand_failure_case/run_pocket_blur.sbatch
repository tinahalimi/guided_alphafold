#!/bin/bash
#SBATCH --job-name=lig-blur150
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/blur_%j.out
#
# Step-2 check: blurred (dB=150) pocket guidance of 8a2b + KY9.
#   calibration:  sbatch --export=ALL,MODE=calibrate run_pocket_blur.sbatch
#   runs:         sbatch --export=ALL,MODE=run,BLUR_SCALE=<s> run_pocket_blur.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_env.sh"
cd "$REPO"
echo "=== $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) MODE=${MODE:-run} BLUR_SCALE=${BLUR_SCALE:-1.0} ==="
NRUNS=$((3 + $(echo "${EXTRA_SCALES:-}" | tr "," "\n" | grep -c .)))
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$EXP/wandb_offline_blur150
mkdir -p "$WANDB_DIR"
$PYTHON -u "$EXP/run_pocket_blur.py"
if [ "${MODE:-run}" = run ] && [ "$WANDB_MODE" = offline ]; then
  export WANDB_API_KEY=$(wandb_key)
  $WANDB_PY -m wandb sync --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" $(ls -d $WANDB_DIR/wandb/offline-run-* | tail -$NRUNS)
  unset WANDB_API_KEY
fi
echo "=== done ==="
