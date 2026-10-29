# Ligand failure case: 8a2b (EGFR kinase + KY9) holo, 4i20 apo

**Running.** Submit every `run_*.sbatch` from this folder (`cd ligand_failure_case && sbatch run_x.sbatch`);
logs go to `logs/`. `slurm_env.sh` finds the guided_alphafold checkout (`$GUIDED_AF_REPO`, else this folder at
`<repo>/experiments/ligand_failure_case`, else a sibling `../guided_alphafold/guided_alphafold`) and reads
`PYTHON`, `WANDB_PY`, `WANDB_NETRC_HOST`, `WANDB_ENTITY`, `WANDB_PROJECT` from the environment or from a
git-ignored `slurm_env.local.sh` next to it.

1. `prepare_gt.py` builds `gt/`: the GT density (repo forward model applied to the deposited 8a2b protein + KY9;
   no waters, no H, altlocs at deposited occupancy, deposited B, RMAX 3.5 Å, spacing 1.69/3 Å),
   the GT model, KY9 alone, and 4i20 superposed into the 8a2b frame.
2. `sbatch run_guidance.sbatch` runs `run_guidance.py`: 5 samples guided + 5 unguided from identical
   latents and noise (repo Protenix sampler, 200 steps, seed 101). Guidance uses the repo loss form
   `sum 0.5*|z(fo)-z(fc)|` over voxels within RMAX of the GT atoms, with the repo's X-ray strength
   (step_size 0.1 x scale 1.0). Runs in about 6 min on an L40S.

Everything (PDBs and CCP4 maps) is in the 8a2b crystal frame, and every map is on the GT grid.
- `outputs/final/`: final structures and maps
- `outputs/checkpoints/step{000,050,100,150,199}/`: snapshots of x0_hat
- `outputs/protenix_v1_unguided/`: the earlier Protenix-v1 run, re-scored the same way
- `outputs/metrics.json`, `trace_*.csv`, `*.png`; `load_in_pymol.pml`
- `outputs_scale0.3_diverged/`: the first attempt at strength 0.3, which blew up (CA RMSD about 50 Å)

In the PDBs, occupancy 0 marks model atoms with no crystal counterpart (the missing loop). These atoms are not in the density.
wandb: entity `tina-halimi-institute-of-science-and-technology-austria`, project `ligand_failure_case`, group `8a2b_holo_xray_baseline` (key read from ~/.netrc).

## Pocket-focused run (`run_pocket_guidance.py`, `sbatch run_pocket_guidance.sbatch`)
The protein outside the pocket is frozen at the 8a2b crystal (the repo's `replace_non_residue_range_atoms`).
Only the 25 pocket residues (within 5 Å of KY9) and KY9 itself are sampled and guided, with the loss computed only around the GT pocket + KY9.
Control: `frozen_unguided`, which has the same freezing but no guidance.
Outputs are in `outputs_pocket_focus/` (same layout as `outputs/`). wandb group: `8a2b_pocket_focus`.

## Blurred pocket guidance (step 2): `run_pocket_blur.py` -> `outputs_pocket_blur150/`
Same frozen pocket-guided setup; the guidance loss uses target and model densities with
B + 150 A^2 (cutoff/zone radius 6.76 A, as the dB=150 basin scan), for all 200 steps. Metrics are
on the unblurred map. Scale chosen by `MODE=calibrate`: w = step_size*scale = 0.1 is the largest
strength for which one guidance step lowers the blurred loss at every probed noise level
(0.15 already overshoots at steps 90-110), so scale stays 1.0; scale 1.5 was run as a sensitivity check.
Result: 4/5 samples end in the crystal (allosteric) pocket at 0.5-1.1 A RMSD; one stops in between.
dB=0 and unguided stay in the ATP site. Conditions rerun from identical latents on an H100: frozen
unguided reproduces the earlier run exactly; dB=0 guided differs per step (GPU numerics amplified by
guidance) but ends the same. wandb group 8a2b_pocket_blur150.

## Whole-structure guidance with blur: `run_whole_blur.py` -> `outputs_whole_blur/`
Protein not frozen, superposition on all protein atoms, scale 1.0, same latents. Conditions:
whole_dB0 (zone within 3.5 A of any GT atom, 338k voxels), whole_dB150 (within 6.76 A of any GT
atom, dB 150 on target and model, 579k voxels), two_zone (LIG = within 6.76 A of GT pocket + KY9
atoms, dB 150, 117k voxels; PROT = whole_dB0 zone minus LIG, dB 0, 244k voxels; disjoint; all model
atoms contribute to each zone with that zone's B/cutoff; loss PROT + 1.0*LIG), and the frozen
pocket dB 150 reference. lambda = 1 (voxel-equal); calibration (`MODE=calibrate`) showed per-atom
|dL/dx| on KY9 ~24-26 vs protein ~27-32 and that norm balancing (lambda ~1.6) changes the KY9 share
of the guidance direction by <0.01. Result: KY9 reaches the allosteric site in 2/5 (whole_dB150),
2/5 (two_zone), 0/5 (whole_dB0), 5/5 (frozen reference). Site labels: `site_hinge_rule` in summary.json.

## Per-group gradient normalisation: `run_whole_blur.py` (`_alpha<a>` conditions) -> `outputs_whole_blur_groupnorm/`
Whole-structure dB=150 (variant 1), same latents. Stage 1: step = a*unit(dL/dx on KY9) + (1-a)*unit(dL/dx on
the rest), per sample (`group_direction`). Stage 2 (repo, x ||delta|| over all atoms) unchanged: one scalar per
sample, so it preserves the split. Sweep a = 0.3/0.5/0.7/0.9 + global-norm baseline (`run_groupnorm.sbatch`,
wandb group 8a2b_whole_blur_groupnorm). KY9 at the crystal pose (<2 A): 2/5 global, 1/5 a=0.3, 4/5 a=0.5,
4/5 a=0.7 (+1 flipped 180 deg in the site), 3/5 a=0.9 (+2 in the site at ~4.8 A). The protein loses guidance
as a rises (CA RMSD 2.06 -> 3.07 A, i.e. back to unguided at a=0.9).

## Independent group weights: `_wL<w>` conditions -> `outputs_whole_blur_wL/`
step = w_L*unit(g_KY9) + w_P*unit(g_protein), w_P = 1 (`W_PROT`), stage 2 unchanged (`run_wL.sbatch`, wandb
group 8a2b_whole_blur_wL). Protein guidance/sampler step stays 0.10 for every w_L (= global baseline); KY9's is
~0.69*w_L. w_L=1.0: 4/5 at the crystal pose with CA 2.09 A (baseline 2.06); w_L=0.5: 3/5, CA 2.40;
w_L>=1.5 (KY9 step >= 1x its sampler step): ligand blown out of the site in 5/5 (up to 108 A).
Baselines (global, alpha 0.5) re-run in the same job and identical to outputs_whole_blur_groupnorm.

## Ensemble loss: `_ens` conditions -> `outputs_whole_blur_ens/`
One loss on the map averaged over the 5 samples (whole_dB150, w_L = w_P = 1, per-sample per-group
normalisation), vs the same setup with independent losses (`run_ens.sbatch`, wandb group 8a2b_whole_blur_ens).
Every run logs the averaged-map loss next to the mean individual loss and the KY9 pose spread.
Result: 3/5 at the crystal pose (independent: 4/5). The ensemble fits the averaged map (final loss 26.5k vs
28.3k), but the individual structures fit worse (mean individual loss 45.7k vs 32.9k; per-sample CC
0.53-0.82 vs 0.72-0.90), and the protein gets worse (CA 2.47 vs 2.09 A). Averaged maps: *_avgmap.ccp4.
Blurred (dB 150, cutoff 6.76 A) maps for viewing: `write_blurred_maps.py` -> gt/8a2b_gt_density_dB150.ccp4 and
outputs_whole_blur_ens/final_blur150/ (5 samples + averaged map, ensemble and independent runs). Same grid as all other maps.

## Real maps (2mFo-DFc from the deposited MTZs), whole-structure guidance, no blur

`prepare_real_maps.py` -> `real_maps/`: FWT/PHWT via phenix.f000 + CCP4 fft (VF000, absolute e/A^3), the repo's
X-ray recipe; unmeasured reflections (12,672 of 41,167 in 8a2b, DFc-filled by PDB-REDO) left out. Full-cell maps
(`<pdb>_2fofc_cell.ccp4`, crystal frame) and maps on the GT box grid (`8a2b_2fofc_box.ccp4`,
`4i20_2fofc_box_8a2b_frame.ccp4`). Real holo vs synthetic GT map: CC 0.71, real ~ 0.31 * synthetic + 0.56.

`run_real.sbatch` (`REAL_MAP=holo|apo`) -> `outputs_real_holo/`, `outputs_real_apo/` (wandb 8a2b_real_holo / _apo):
`run_whole_blur.py` with fo = real map (clip 0) on WHOLE0, fc incl. crystal-symmetry images; unguided,
whole_dB0 (global norm), whole_dB0_wL1.0 (per group, w_L = w_P = 1).

Holo (means of 5): KY9 RMSD 9.30 / 8.58 / 8.35 A, CA 3.21 / 1.95 / 2.13 A, pocket 2.51 / 2.58 / 2.38 A,
whole-map CC 0.46 / 0.76 / 0.75, KY9 CC 0.24 / 0.42 / 0.50. KY9 reaches the allosteric site in 0/5 for every
condition (still in the ATP site; w_L = 1 gives 3/5 "mixed" with E758/A763 contacts).
Apo (control): protein moves AWAY from 4i20 (CA vs 4i20 2.76 -> 4.76 A) -- the sampler is superposed on the
8a2b crystal, not on 4i20, so the apo map is compared in an offset frame; not a valid control as run.
