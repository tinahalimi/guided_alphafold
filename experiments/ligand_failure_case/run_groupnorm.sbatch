#!/bin/bash
#SBATCH --job-name=lig-groupnorm
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/groupnorm_%j.out
#
# Per-group (KY9 vs protein) gradient normalisation, whole-structure dB=150, alpha sweep.
#   OUT_DIR, WANDB_GROUP and CONDS are set below.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_env.sh"
cd "$REPO"
echo "=== $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) MODE=${MODE:-run} LAMBDA=${LAMBDA:-1} ==="
export OUT_DIR=outputs_whole_blur_groupnorm WANDB_GROUP=8a2b_whole_blur_groupnorm
export CONDS=${CONDS:-whole_dB150,whole_dB150_alpha0.3,whole_dB150_alpha0.5,whole_dB150_alpha0.7,whole_dB150_alpha0.9}
NRUNS=$(echo "$CONDS" | tr "," "\n" | grep -c .)
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$EXP/wandb_offline_groupnorm
mkdir -p "$WANDB_DIR"
$PYTHON -u "$EXP/run_whole_blur.py"
if [ "${MODE:-run}" = run ] && [ "$WANDB_MODE" = offline ]; then
  export WANDB_API_KEY=$(wandb_key)
  $WANDB_PY -m wandb sync --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" $(ls -d $WANDB_DIR/wandb/offline-run-* | tail -$NRUNS)
  unset WANDB_API_KEY
fi
echo "=== done ==="
