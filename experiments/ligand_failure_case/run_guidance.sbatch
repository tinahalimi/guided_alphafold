#!/bin/bash
#SBATCH --job-name=lig-fail-8a2b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_%j.out
#
# Density-guided vs unguided sampling of 8a2b + KY9. Smoke test:
#   sbatch --export=ALL,SMOKE_N=3,WANDB_MODE=disabled run_guidance.sbatch
# Relative paths in the config (checkpoint, CCD, alignment_dir, msa_cache) resolve against
# the repo root, so this runs from there.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_env.sh"
cd "$REPO"
echo "=== $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
# log offline, then upload with wandb 0.31 (protenix env): guided_af3's wandb 0.19.4 cannot use
# the account's 86-char wandb_v1 key (WANDB_PY in slurm_env.local.sh). The key is never printed.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$EXP/wandb_offline
mkdir -p "$WANDB_DIR"
$PYTHON -u "$EXP/run_guidance.py"
if [ "$WANDB_MODE" = offline ]; then
  export WANDB_API_KEY=$(wandb_key)
  $WANDB_PY -m wandb sync --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" $(ls -d $WANDB_DIR/wandb/offline-run-* | tail -2)
  WANDB_MODE=online $WANDB_PY "$EXP/log_summary_wandb.py"
  unset WANDB_API_KEY
fi
echo "=== done ==="
