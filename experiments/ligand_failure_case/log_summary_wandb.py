#!/usr/bin/env python
"""
Upload the final comparison table and the two figures as the wandb "summary" run.

Runs in the protenix env (wandb 0.31). guided_af3's wandb 0.19.4 cannot authenticate with the
account's 86-char wandb_v1 key, and even offline it tries to upload when a Table is logged.
The key comes via WANDB_API_KEY, set by the sbatch (slurm_env.sh: wandb_key).

    python log_summary_wandb.py [existing_run_id]
    SUMMARY_OUT=outputs_pocket_focus SUMMARY_GROUP=8a2b_pocket_focus python log_summary_wandb.py
"""
import json
import os
import sys

import wandb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("SUMMARY_OUT", "outputs"))
ENTITY = os.environ.get("WANDB_ENTITY", "tina-halimi-institute-of-science-and-technology-austria")
PROJECT = os.environ.get("WANDB_PROJECT", "ligand_failure_case")
GROUP = os.environ.get("SUMMARY_GROUP", "8a2b_holo_xray_baseline")

m = json.load(open(os.path.join(OUT, "metrics.json")))
TAGS = list(m["final_summary"])
run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP, name="summary", job_type="summary",
                 config=m["config"], id=sys.argv[1] if len(sys.argv) > 1 else None,
                 resume="allow")
tbl = wandb.Table(columns=["metric", "statistic", *TAGS])
for k in m["final_summary"][TAGS[0]]:
    for st in ("mean", "min", "max"):
        tbl.add_data(k, st, *(m["final_summary"][t].get(k, {}).get(st) for t in TAGS))
run.log({"final_comparison": tbl,
         "plots/trajectories": wandb.Image(os.path.join(OUT, "trajectories.png")),
         "plots/final_comparison": wandb.Image(os.path.join(OUT, "final_comparison.png"))})
run.summary.update({f"final/{t}/{k}": v["mean"] for t in TAGS
                    for k, v in m["final_summary"][t].items()})
run.summary.update({f"floor/{k}": v for k, v in m["floor"].items()})
print(run.url)
run.finish()
