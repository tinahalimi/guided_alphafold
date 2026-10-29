# Sourced by every run_*.sbatch. Submit from this folder:  cd <this folder> && sbatch run_x.sbatch
# Override anything below from the environment (sbatch exports it), or put your settings in
# slurm_env.local.sh next to this file (git-ignored, sourced first).
EXP=${EXP:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "$EXP/forward_model.py" ] || { echo "run sbatch from the ligand_failure_case folder (or set EXP)" >&2; exit 1; }
[ -f "$EXP/slurm_env.local.sh" ] && source "$EXP/slurm_env.local.sh"

# guided_alphafold checkout: $GUIDED_AF_REPO, else <repo>/experiments/ligand_failure_case, else a sibling checkout
REPO=""
for c in "${GUIDED_AF_REPO:-}" "$EXP/../.." "$EXP/../guided_alphafold/guided_alphafold"; do
  if [ -n "$c" ] && [ -f "$c/experiment_manager.py" ]; then REPO=$(cd "$c" && pwd); break; fi
done
[ -n "$REPO" ] || { echo "guided_alphafold repo not found; set GUIDED_AF_REPO" >&2; exit 1; }
export GUIDED_AF_REPO=$REPO

PYTHON=${PYTHON:-python}              # env with the guided_alphafold requirements (torch, gemmi, wandb, ...)
WANDB_PY=${WANDB_PY:-$PYTHON}         # env whose wandb uploads the offline runs (needs wandb >= 0.22 for v1 keys)
export WANDB_ENTITY=${WANDB_ENTITY:-tina-halimi-institute-of-science-and-technology-austria}
export WANDB_PROJECT=${WANDB_PROJECT:-ligand_failure_case}
export WANDB_NETRC_HOST=${WANDB_NETRC_HOST:-api.wandb.ai}

# W&B key: $WANDB_API_KEY if set, else ~/.netrc entry for $WANDB_NETRC_HOST. Never printed.
wandb_key() {
  if [ -n "${WANDB_API_KEY:-}" ]; then echo "$WANDB_API_KEY"
  else $WANDB_PY -c "import netrc,sys;print(netrc.netrc().authenticators(sys.argv[1])[2])" "$WANDB_NETRC_HOST"; fi
}
