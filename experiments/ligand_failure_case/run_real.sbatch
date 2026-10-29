#!/bin/bash
#SBATCH --job-name=lig-realmap
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/real_%x_%j.out
#
# Whole-structure guidance on the REAL 2mFo-DFc map (prepare_real_maps.py), no blur.
#   REAL_MAP=holo sbatch --export=ALL run_real.sbatch      (8a2b map)
#   REAL_MAP=apo  sbatch --export=ALL run_real.sbatch      (4i20 map, control)
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_env.sh"
cd "$REPO"
echo "=== $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) REAL_MAP=$REAL_MAP ==="
export REAL_MAP OUT_DIR=outputs_real_$REAL_MAP WANDB_GROUP=8a2b_real_$REAL_MAP
export CONDS=${CONDS:-unguided,whole_dB0,whole_dB0_wL1.0}
NRUNS=$(echo "$CONDS" | tr "," "\n" | grep -c .)
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$EXP/wandb_offline_real_$REAL_MAP
mkdir -p "$WANDB_DIR"
$PYTHON -u "$EXP/run_whole_blur.py"
if [ "$WANDB_MODE" = offline ]; then
  export WANDB_API_KEY=$(wandb_key)
  $WANDB_PY -m wandb sync --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" $(ls -d $WANDB_DIR/wandb/offline-run-* | tail -$NRUNS)
  unset WANDB_API_KEY
fi
echo "=== done ==="
