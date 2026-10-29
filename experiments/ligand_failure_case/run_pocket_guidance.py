#!/usr/bin/env python
"""
Pocket-focused density guidance of 8a2b + KY9: protein frozen, only the pocket and KY9 guided.

Same sampler, forward model, loss form, strength (0.1) and latents as run_guidance.py, with the
repo's region-of-interest treatment (DensityGuidanceLossFunction.replace_non_residue_range_atoms):

  ROI     = the 25 pocket residues (any atom within 5 A of KY9 in 8a2b, gt/meta.json) + KY9
  frozen  = every other protein atom that exists in the crystal. At each step, after the
            superposition, these are REPLACED by the 8a2b crystal coordinates, both in the
            structure whose density is computed and in the x0_hat handed to the sampler update.
            They therefore carry no gradient and the rest of the protein is held at the crystal.
            Model atoms with no crystal counterpart (missing loop) are left as sampled.
  align   = superposition on the frozen atoms only
  loss    = sum 0.5*|z(fo)-z(fc)| over voxels within RMAX of the GT pocket + KY9 atoms,
            fo standardised over that zone (repo form); fc includes the frozen protein's density

Runs, from identical latents and noise:
  pocket_guided    frozen protein + density guidance on the ROI
  frozen_unguided  frozen protein, no guidance (control: what freezing alone does)
The earlier whole-structure guided / plain unguided results (outputs/metrics.json) are read for
comparison, not re-run.

Outputs: outputs_pocket_focus/ (same layout as outputs/). wandb group 8a2b_pocket_focus.
"""
import csv
import json
import os
import sys
import time

import gemmi
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from forward_model import REPO, AtomDensityEstimator, sample_fc  # noqa: E402
from prepare_gt import read_atoms, grid_points, write_ccp4, kabsch, LIGAND  # noqa: E402

sys.path.insert(0, REPO)
from src.utils.io import load_config, query_msa_server  # noqa: E402
from src.utils.non_diffusion_model_manager import ProtenixModelManager  # noqa: E402
from src.protenix.metrics.rmsd import partially_aligned_rmsd  # noqa: E402
from experiment_manager import ExperimentManager  # noqa: E402

CONFIG = os.path.join(HERE, "configs", "8a2b_holo.yaml")
GT = os.path.join(HERE, "gt")
OUT = os.path.join(HERE, "outputs_pocket_focus")
PREV = os.path.join(HERE, "outputs", "metrics.json")   # earlier whole-structure runs
RES_OFFSET = 701            # crystal residue number = sampler res_id + 701 (checked below)
CHECKPOINTS = [0, 50, 100, 150, 199]   # 199 = final structure
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ligand_failure_case")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "tina-halimi-institute-of-science-and-technology-austria")
WANDB_GROUP = "8a2b_pocket_focus"
WANDB_NETRC_HOST = os.environ.get("WANDB_NETRC_HOST", "api.wandb.ai")   # ~/.netrc entry with the W&B key (slurm_env.local.sh)
SMOKE_N = int(os.environ.get("SMOKE_N", "0"))


# ---------------------------------------------------------------------------------------
def build_generator(cfg, device):
    """ProtenixModelManager exactly as ExperimentManager._get_model_manager builds it.

    Built directly so the repo's own loss objects (which need a reference structure in the
    repo's layout) are not constructed; the MSA directory is resolved the same way.
    """
    ExperimentManager.seed_experiment(cfg.general.seed)
    msa_dir = os.path.join(cfg.model_manager.msa_save_dir, cfg.protein.pdb_id)
    query_msa_server(msa_dir, cfg.protein.sequences)   # cache hit: fetched on the head node
    mm = cfg.model_manager
    return ProtenixModelManager(
        sequences_dictionary=cfg.protein.sequences, pdb_id=cfg.protein.pdb_id,
        assembly_identifier=cfg.protein.assembly_identifier,
        chains_to_read=cfg.protein.chains_to_use, ROI_residues=None,
        should_align_to_chains=cfg.protein.should_align_to_chains,
        reference_pdb=cfg.protein.reference_pdb,
        pdb_contains_missing_atoms=cfg.protein.contains_missing_atoms,
        N_cycle=mm.N_cycle, chunk_size=mm.chunk_size, diffusion_N=mm.diffusion_N,
        gamma0=mm.gamma0, gamma_min=mm.gamma_min, noise_scale_lambda=mm.noise_scale_lambda,
        step_scale_eta=mm.step_scale_eta, dtype="fp32", use_deepspeed_evo_attention=False,
        msa_save_dir=msa_dir, msa_embedding_cache_dir=mm.msa_embedding_cache_dir,
        model_checkpoint_path=mm.model_checkpoint_path, dump_dir=mm.dump_dir,
        use_msa=mm.use_msa, batch_size=cfg.general.batch_size, device=device,
        pairformer_mixed_precision=mm.pairformer_mixed_precision)


def guidance_direction(grad, scale):
    """Per-sample unit direction x scale (normalize_gradients=true).

    The repo skips normalisation when every element is < 1e-6 and sets norms < 1e-4 to 1
    (divide-by-zero guards). Those guards can silently turn guidance off when gradients are
    small (seen in the 5QU work), so normalise whenever the norm is nonzero; the update is then
    scale * ||delta|| * unit_direction, as the repo intends.
    """
    n = grad.flatten(1, -1).norm(dim=-1)
    n = torch.where(n > 0, n, torch.ones_like(n))
    return grad / n[:, None, None] * scale


def ligand_metrics(P, Q):
    P, Q = P.astype(np.float64), Q.astype(np.float64)
    R, t = kabsch(P, Q)
    return dict(lig_rmsd=float(np.sqrt(((P - Q) ** 2).sum(1).mean())),
                lig_centroid=float(np.linalg.norm(P.mean(0) - Q.mean(0))),
                lig_internal=float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean())),
                lig_orient_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))


def pearson(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)


# ---------------------------------------------------------------------------------------
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_config(CONFIG)
    gcfg = cfg.diffusion_process.guidance
    SCALE = float(gcfg.guidance_direction_scale_factor)
    START_FROM = int(gcfg.start_guidance_from)
    B = int(cfg.general.batch_size)
    for d in ("checkpoints", "final"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    # ---- ground truth ---------------------------------------------------------------
    meta = json.load(open(os.path.join(GT, "meta.json")))
    g = np.load(os.path.join(GT, "grid.npz"))
    lo, n, s, RMAX = g["lo"], g["n"], float(g["spacing"]), float(g["rmax"])
    pts = grid_points(lo, n, s)
    gt_map = gemmi.read_ccp4_map(os.path.join(GT, "8a2b_gt_density.ccp4"))
    gt_vals = np.array(gt_map.grid, copy=False).ravel().astype(np.float32)
    assert gt_vals.size == len(pts), "GT map does not match grid.npz"
    st_gt, gt_atoms = read_atoms(os.path.join(GT, "8a2b_gt_model.pdb"))
    cell, sg = st_gt.cell, st_gt.spacegroup_hm
    # one record per (residue, atom name): first conformer for altlocs
    prot_key, lig_key = {}, {}
    for a in gt_atoms:
        (lig_key.setdefault(a["name"], a) if a["is_ligand"]
         else prot_key.setdefault((a["resnum"], a["name"]), a))
    pocket = set(meta["pocket_residues"])
    _, apo_atoms = read_atoms(os.path.join(GT, "4i20_apo_in_8a2b_frame.pdb"))
    apo_ca = {a["resnum"]: a["xyz"] for a in apo_atoms if a["name"] == "CA" and a["altloc"] in ("", "A")}

    # ---- generator --------------------------------------------------------------------
    t0 = time.time()
    mgr = build_generator(cfg, device)
    N = int(cfg.model_manager.diffusion_N)
    assert len(mgr.noise_schedule) >= N + 1
    n_steps = SMOKE_N or N
    print(f"generator built in {time.time()-t0:.0f}s, {mgr.atom_array.shape[0]} atoms, "
          f"{N} diffusion steps{f' (SMOKE: {n_steps})' if SMOKE_N else ''}, batch {B}", flush=True)

    aa = mgr.atom_array
    res_id, res_name = np.asarray(aa.res_id), np.asarray(aa.res_name)
    atom_name, element = np.asarray(aa.atom_name), np.asarray(aa.element)
    is_lig = res_name == LIGAND
    gen_prot, gen_lig, recs = [], [], []
    for i in range(len(aa)):
        if is_lig[i]:
            r = lig_key.get(str(atom_name[i]))
            if r is not None:
                gen_lig.append(i); recs.append(r)
        else:
            r = prot_key.get((int(res_id[i]) + RES_OFFSET, str(atom_name[i])))
            if r is not None:
                assert r["resname"] == res_name[i], f"residue mismatch at {res_id[i]}"
                gen_prot.append(i); recs.append(r)
    assert len(gen_lig) == len(lig_key) == 48, f"ligand mapping {len(gen_lig)}/{len(lig_key)}"
    mapped = gen_prot + gen_lig
    print(f"mapped {len(gen_prot)}/{int((~is_lig).sum())} protein atoms and {len(gen_lig)}/"
          f"{int(is_lig.sum())} {LIGAND} atoms onto the crystal; "
          f"{len(prot_key) - len(gen_prot)} crystal protein atoms have no model counterpart")

    T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=device)
    true_xyz = T([r["xyz"] for r in recs])                  # mapped atoms, crystal coords
    n_p = len(gen_prot)
    ca_idx = [k for k, i in enumerate(gen_prot) if atom_name[i] == "CA"]
    pocket_idx = [k for k, r in enumerate(recs[:n_p]) if r["resnum"] in pocket]
    pocket_ca = [k for k in ca_idx if recs[k]["resnum"] in pocket and recs[k]["resnum"] in apo_ca]
    apo_pocket_ca = T([apo_ca[recs[k]["resnum"]] for k in pocket_ca])
    lig_true = true_xyz[n_p:].cpu().numpy()

    # forward model inputs for the model: mapped atoms only, occupancy 1, crystal B
    est = AtomDensityEstimator(device)
    occ = torch.ones(B, len(mapped), device=device)
    bfac = T([r["b"] for r in recs])[None].repeat(B, 1)
    el = est.get_element_indexes(np.array([r["element"] for r in recs]))[None].repeat(B, 1)
    full_grid = T(pts)[None]

    # zone of interest: voxels within RMAX of any GT atom (repo: within rmax of the zone atoms)
    gt_xyz_all = T([a["xyz"] for a in gt_atoms])
    def near(points, atoms, r):
        out = torch.zeros(len(points), dtype=torch.bool, device=device)
        for c in range(0, len(points), 200000):
            out[c:c + 200000] = torch.cdist(points[c:c + 200000], atoms).min(1)[0] < r
        return out
    roi_gt = T([a["xyz"] for a in gt_atoms if a["is_ligand"] or a["resnum"] in pocket])
    zone = near(full_grid[0], roi_gt, RMAX)      # loss zone: around the GT pocket + KY9
    zone_pts = full_grid[:, zone]
    fo = T(gt_vals)[zone]
    mu, sd = fo.mean(), fo.std() + 1e-6
    lig_vox = near(zone_pts[0], T(lig_true), RMAX)
    pocket_vox = near(zone_pts[0], true_xyz[pocket_idx], RMAX) | lig_vox
    print(f"pocket+ligand loss zone {int(zone.sum()):,} voxels ({100*zone.float().mean():.1f}% of the box); "
          f"ligand region {int(lig_vox.sum()):,}, pocket+ligand region {int(pocket_vox.sum()):,}")

    tgt = torch.zeros(B, len(aa), 3, device=device)
    msk = torch.zeros(B, len(aa), device=device)
    roi = np.zeros(len(mapped), bool)
    roi[pocket_idx] = True
    roi[n_p:] = True                                   # KY9
    frozen = torch.tensor(~roi, device=device)
    gen_frozen = [mapped[k] for k in np.flatnonzero(~roi)]
    tgt[:, gen_prot] = true_xyz[:n_p][None].expand(B, -1, -1).clone()   # explicit: deterministic index_put does not broadcast
    msk[:, gen_frozen] = 1.0      # superpose on the frozen protein atoms only
    print(f"ROI: {len(pocket)} pocket residues ({len(pocket_idx)} atoms) + {len(mapped)-n_p} {LIGAND} atoms; "
          f"{len(gen_frozen)} protein atoms frozen at the crystal", flush=True)
    mapped_t = torch.tensor(mapped, device=device)

    def freeze(al):
        """Replace the frozen atoms by the crystal (repo replace_non_residue_range_atoms)."""
        c = torch.where(frozen[None, :, None], true_xyz[None], al[:, mapped_t])
        al = al.index_copy(1, mapped_t, c)
        return al, c

    def fc_of(coords, grid):
        return sample_fc(est, coords, occ[:coords.shape[0]], el[:coords.shape[0]],
                         bfac[:coords.shape[0]], grid.expand(coords.shape[0], -1, -1), RMAX)

    def density_loss(fc):                       # per sample, the repo's form
        return (0.5 * ((fo - mu) / sd - (fc - mu) / sd).abs()).sum(-1)

    def metrics(c, fc):
        """c: [b, n_mapped, 3] aligned coords (crystal frame); fc: [b, zone] density."""
        with torch.no_grad():
            d2 = ((c - true_xyz) ** 2).sum(-1)
            rm = lambda idx: d2[:, idx].mean(-1).sqrt()
            m = dict(loss=density_loss(fc), mse=(fc - fo).pow(2).mean(-1),
                     cc=pearson(fc, fo), cc_ligand=pearson(fc[:, lig_vox], fo[lig_vox]),
                     cc_pocket=pearson(fc[:, pocket_vox], fo[pocket_vox]),
                     rmsd_ca=rm(ca_idx), rmsd_protein=rm(list(range(n_p))),
                     rmsd_roi=rm(list(np.flatnonzero(roi))),
                     rmsd_pocket=rm(pocket_idx), rmsd_pocket_ca=rm(pocket_ca),
                     rmsd_pocket_ca_vs_apo=((c[:, pocket_ca] - apo_pocket_ca) ** 2)
                     .sum(-1).mean(-1).sqrt())
            m = {k: v.cpu().numpy().astype(float) for k, v in m.items()}
            cl = c[:, n_p:].cpu().numpy()
            lm = [ligand_metrics(cl[b], lig_true) for b in range(c.shape[0])]
            for k in lm[0]:
                m[k] = np.array([x[k] for x in lm])
        return m

    # floor: the crystal model itself scored like a prediction (one conformer, occ 1)
    with torch.no_grad():
        fc_true = fc_of(true_xyz[None], zone_pts)
        floor = {k: float(v[0]) for k, v in metrics(true_xyz[None], fc_true).items()}
    print(f"floor (crystal model as a prediction): loss {floor['loss']:.1f}, CC {floor['cc']:.4f}, "
          f"CC ligand {floor['cc_ligand']:.4f}  -- below this is unreachable", flush=True)

    # ---- writers ------------------------------------------------------------------------
    b_out = np.zeros(len(aa)); occ_out = np.zeros(len(aa))
    b_out[mapped] = [r["b"] for r in recs]; occ_out[mapped] = 1.0

    def write_model(path, xyz):
        """All sampler atoms, crystal numbering; chain A protein, chain B KY9.
        occupancy 1 = contributes to the density, 0 = no crystal counterpart."""
        stw = gemmi.Structure()
        stw.cell, stw.spacegroup_hm = cell, sg
        model = gemmi.Model("1")
        chains = {"A": gemmi.Chain("A"), "B": gemmi.Chain("B")}
        cur, key = None, None
        for i in range(len(aa)):
            lig = bool(is_lig[i])
            k = (lig, int(res_id[i]))
            if k != key:
                if cur is not None:
                    chains[ch].add_residue(cur)
                cur, key, ch = gemmi.Residue(), k, "B" if lig else "A"
                cur.name = str(res_name[i])
                cur.seqid = gemmi.SeqId(1101 if lig else int(res_id[i]) + RES_OFFSET, " ")
                cur.het_flag = "H" if lig else "A"
            at = gemmi.Atom()
            at.name, at.element = str(atom_name[i]), gemmi.Element(str(element[i]))
            at.pos = gemmi.Position(*map(float, xyz[i]))
            at.occ, at.b_iso = float(occ_out[i]), float(b_out[i])
            cur.add_atom(at)
        chains[ch].add_residue(cur)
        for c in chains.values():
            model.add_chain(c)
        stw.add_model(model)
        stw.setup_entities()
        stw.write_pdb(path)

    def write_map(path, coords_mapped):
        with torch.no_grad():
            v = fc_of(coords_mapped[None], full_grid)[0].cpu().numpy()
        write_ccp4(path, v, lo, n, s)

    # ---- wandb ------------------------------------------------------------------------
    import wandb
    wb_mode = os.environ.get("WANDB_MODE", "online")
    # guided_af3 has wandb 0.19.4, which rejects the 86-char wandb_v1 keys, so the sbatch runs
    # offline (no login needed) and uploads afterwards with a newer wandb (protenix env).
    if wb_mode == "online":
        import netrc
        wandb.login(key=netrc.netrc().authenticators(WANDB_NETRC_HOST)[2], relogin=True)
    run_cfg = dict(pdb="8a2b", ligand=LIGAND, batch=B, diffusion_N=N, n_steps=n_steps,
                   guidance_scale=SCALE, step_size=float(gcfg.step_size),
                   effective_strength=SCALE * float(gcfg.step_size), start_guidance_from=START_FROM, rmax=RMAX, spacing=s, seed=cfg.general.seed,
                   loss="sum over pocket+KY9 zone 0.5*|z(fo)-z(fc)| (repo form)", roi="pocket (5 A) + KY9", frozen_protein="crystal 8a2b", checkpoints=CHECKPOINTS,
                   n_mapped_protein=n_p, n_ligand=len(gen_lig), floor=floor)

    # ---- identical starting point for both runs ---------------------------------------
    lat0 = mgr.get_x_noisy(mgr.get_x_start(B)).detach().clone()
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None

    def run(guided):
        tag = "pocket_guided" if guided else "frozen_unguided"
        wr = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=WANDB_GROUP, name=tag, job_type=tag,
                        config=run_cfg, mode=wb_mode, reinit=True)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        torch.set_rng_state(rng_cpu)
        lat = lat0.clone()
        trace = []
        t1 = time.time()
        for i in range(n_steps):
            guide = guided and i > START_FROM and i != N - 1   # as the repo's loop
            if guide:
                lat.requires_grad_(True)
            with (torch.enable_grad() if guide else torch.no_grad()):
                x0 = mgr.get_x_0_hat_from_x_noisy(lat, start_index=i, inplace_safe=False)
                out = partially_aligned_rmsd(pred_pose=x0, true_pose=tgt, align_mask=msk,
                                             atom_mask=msk, reduce=True)
                al, R, Tr = out[2], out[3], out[4]
                al, c = freeze(al)                 # frozen atoms -> crystal, no gradient
                fc = fc_of(c, zone_pts)
                loss = density_loss(fc)
            gd, gnorm = None, np.full(B, np.nan)
            if guide:
                loss.sum().backward()          # samples are independent: one grad each
                grad = lat.grad
                assert grad is not None and float(grad.abs().max()) > 0, "guidance is dead"
                gnorm = grad.flatten(1, -1).norm(dim=-1).cpu().numpy()
                with torch.no_grad():
                    gd = guidance_direction(grad, SCALE)
                if i == START_FROM + 1:
                    gn = gd.flatten(1, -1).norm(dim=-1)
                    assert torch.allclose(gn, torch.full_like(gn, SCALE), rtol=1e-3), gn
                    print(f"  [{tag}] guidance check OK: |direction| = {SCALE} per sample, "
                          f"raw |grad| {gnorm.round(3)}", flush=True)
                lat.grad = None
            with torch.no_grad():
                m = metrics(c.detach(), fc.detach())
                t_hat = float(mgr.get_t_hat(i))
                row = dict(step=i, t_hat=t_hat, **{f"{k}/s{b}": float(v[b]) for k, v in m.items()
                                                   for b in range(B)},
                           **{f"{k}/mean": float(v.mean()) for k, v in m.items()})
                if guide:
                    row.update({f"grad_norm/s{b}": float(gnorm[b]) for b in range(B)})
                    row["grad_norm/mean"] = float(gnorm.mean())
                trace.append(row)
                wr.log(row, step=i)
                if i in CHECKPOINTS or i == n_steps - 1:
                    d = os.path.join(OUT, "checkpoints", f"step{i:03d}")
                    os.makedirs(d, exist_ok=True)
                    for b in range(B):
                        write_model(os.path.join(d, f"{tag}_sample{b}.pdb"),
                                    al[b].detach().cpu().numpy())
                        write_map(os.path.join(d, f"{tag}_sample{b}.ccp4"), c[b].detach())
                nx = (R.permute(0, 2, 1)[:, None] @ (al.detach() - Tr)[..., None]).squeeze(-1)
            if i % 10 == 0 or i == n_steps - 1:
                print(f"  {tag[0].upper()} {i:3d} t {t_hat:7.2f}  loss {m['loss'].mean():9.1f}  "
                      f"CC {m['cc'].mean():.3f}  CClig {m['cc_ligand'].mean():.3f}  "
                      f"CA {m['rmsd_ca'].mean():.2f}  pocket {m['rmsd_pocket'].mean():.2f}  "
                      f"lig {m['lig_rmsd'].mean():.2f} A", flush=True)
            lat = mgr.get_x_t_from_x_0_hat(lat.detach(), nx, i, i + 1, guidance_direction=gd,
                                           step_size=float(gcfg.step_size),
                                           normalize_gradients=bool(gcfg.normalize_gradients))
            if i < N - 1:
                lat = mgr.get_x_noisy(lat, i, i + 1)
            lat = lat.detach().clone()
        secs = time.time() - t1
        final = {k: v for k, v in trace[-1].items()}
        wr.summary.update(dict(seconds=secs, **{f"final/{k}": v for k, v in final.items()}))
        wr.finish()
        print(f"  [{tag}] {n_steps} steps in {secs:.0f}s", flush=True)
        return trace, al.detach(), c.detach()

    print("\nPOCKET-GUIDED (protein frozen)", flush=True)
    G, G_al, G_c = run(True)
    print("\nFROZEN, UNGUIDED (same latents, same noise sequence)", flush=True)
    U, U_al, U_c = run(False)
    last = f"step{n_steps-1:03d}"
    for tag in ("pocket_guided", "frozen_unguided"):
        for b in range(B):
            for ext in ("pdb", "ccp4"):
                src = os.path.join(OUT, "checkpoints", last, f"{tag}_sample{b}.{ext}")
                dst = os.path.join(OUT, "final", f"{tag}_sample{b}.{ext}")
                if os.path.exists(dst):
                    os.remove(dst)
                os.link(src, dst)

    # ---- earlier whole-structure runs, for comparison (pocket/ligand metrics share definitions)
    prev = json.load(open(PREV))

    # ---- tables, plots, wandb summary ---------------------------------------------------
    for tag, tr in (("pocket_guided", G), ("frozen_unguided", U)):
        keys = sorted({k for r in tr for k in r})
        with open(os.path.join(OUT, f"trace_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(tr)
    KEYS = ["loss", "cc", "cc_pocket", "cc_ligand", "rmsd_ca", "rmsd_protein", "rmsd_roi", "rmsd_pocket",
            "rmsd_pocket_ca", "rmsd_pocket_ca_vs_apo", "lig_rmsd", "lig_centroid",
            "lig_orient_deg", "lig_internal"]
    per_sample = {tag: {k: [tr[-1][f"{k}/s{b}"] for b in range(B)] for k in KEYS}
                  for tag, tr in (("pocket_guided", G), ("frozen_unguided", U))}
    for t_new, t_old in (("wholestructure_guided", "guided"), ("plain_unguided", "unguided"),
                         ("protenix_v1_unguided", "protenix_v1_unguided")):
        # loss/cc/rmsd_roi are not comparable (different zone / not computed): pocket+ligand only
        per_sample[t_new] = {k: v for k, v in prev["final_per_sample"][t_old].items()
                             if k not in ("loss", "cc")}
    summary = {tag: {k: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))
                     for k, v in d.items()} for tag, d in per_sample.items()}
    TAGS = list(per_sample)
    json.dump(dict(config=run_cfg, floor=floor, gt_meta=meta, final_per_sample=per_sample,
                   final_summary=summary), open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    figs = make_plots(G, U, summary, floor, B, meta)
    write_pml(B, n_steps)

    print("\nFINAL (mean over 5 samples)")
    print("  " + f"{'metric':<22}" + "".join(f"{t:>23}" for t in TAGS))
    for k in KEYS:
        print("  " + f"{k:<22}" + "".join(f"{summary[t][k]['mean']:>23.3f}" if k in summary[t]
                                          else f"{'-':>23}" for t in TAGS))
    print(f"  floor: loss {floor['loss']:.1f}, CC {floor['cc']:.4f}, CC ligand {floor['cc_ligand']:.4f}")

    # the "summary" wandb run (table + figures) is uploaded by log_summary_wandb.py
    print(f"\nwrote outputs to {OUT}")


# ---------------------------------------------------------------------------------------
def make_plots(G, U, summary, floor, B, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CG, CU = "#2a78d6", "#8a8984"
    REF = [("wholestructure_guided", "#1baf7a", "--", "earlier whole-structure guided"),
           ("plain_unguided", "#4a4a4a", "-.", "earlier plain unguided"),
           ("protenix_v1_unguided", "#eb6834", ":", "Protenix-v1 unguided")]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df"})
    panels = [("loss", "density loss, pocket+KY9 zone", True),
              ("cc_pocket", "map CC, pocket+ligand region", False),
              ("cc_ligand", "map CC, ligand region", False),
              ("rmsd_pocket", "pocket heavy-atom RMSD vs 8a2b (A)", False),
              ("rmsd_pocket_ca_vs_apo", "pocket CA RMSD vs 4i20 apo (A)", False),
              ("lig_rmsd", "KY9 RMSD vs crystal (A)", False),
              ("lig_centroid", "KY9 centroid offset (A)", False),
              ("lig_orient_deg", "KY9 orientation error (deg)", False),
              ("grad_norm", "raw |d loss / d x_noisy| (guided)", True)]
    out = []
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    steps = [r["step"] for r in G]
    for ax, (k, title, logy) in zip(axes.flat, panels):
        for tr, col, lab in ((U, CU, "frozen, unguided"), (G, CG, "frozen, pocket-guided")):
            if f"{k}/mean" not in tr[-2 if len(tr) > 1 else -1]:
                continue
            for b in range(B):
                ax.plot(steps, [r.get(f"{k}/s{b}", np.nan) for r in tr], color=col, lw=0.6, alpha=0.35)
            ax.plot(steps, [r.get(f"{k}/mean", np.nan) for r in tr], color=col, lw=2, label=f"{lab} (mean of {B})")
        if k in floor and k != "grad_norm":
            ax.axhline(floor[k], color="black", ls=":", lw=1, label="crystal model (floor)")
        for tag, col, ls, lab in REF:
            if k in summary.get(tag, {}):
                ax.axhline(summary[tag][k]["mean"], color=col, ls=ls, lw=1.2, label=f"{lab}, final")
        if k == "rmsd_pocket_ca_vs_apo":
            ax.axhline(meta["apo_superposition"]["ca_rmsd_pocket"], color="crimson", ls="-.", lw=1,
                       label="8a2b crystal vs 4i20")
        if logy:
            ax.set_yscale("log")
        ax.set_title(title); ax.set_xlabel("diffusion step")
        ax.legend(fontsize=6)
    fig.suptitle("8a2b + KY9, protein frozen at the crystal: pocket+ligand density guidance vs none "
                 "(same latents, same noise)", y=0.995)
    fig.tight_layout()
    p = os.path.join(OUT, "trajectories.png"); fig.savefig(p, dpi=150); out.append(p)

    KEYS = [("cc_pocket", "map CC, pocket+ligand"), ("cc_ligand", "map CC, ligand"),
            ("rmsd_pocket", "pocket RMSD (A)"), ("lig_rmsd", "KY9 RMSD (A)"),
            ("lig_centroid", "KY9 centroid offset (A)"), ("lig_orient_deg", "KY9 orientation (deg)")]
    cols = [("pocket_guided", "frozen,\npocket-guided", CG), ("frozen_unguided", "frozen,\nunguided", CU)] + \
           [(t, lab.replace("earlier ", "").replace(" ", "\n", 1), c) for t, c, _, lab in REF]
    fig, axes = plt.subplots(1, len(KEYS), figsize=(20, 3.8))
    import json as _json
    per = _json.load(open(os.path.join(OUT, "metrics.json")))["final_per_sample"]
    for ax, (k, title) in zip(axes, KEYS):
        for x, (tag, lab, col) in enumerate(cols):
            vals = per[tag][k]
            ax.scatter(np.full(len(vals), x) + np.linspace(-0.12, 0.12, len(vals)), vals, color=col, s=18)
            ax.hlines(np.mean(vals), x - 0.25, x + 0.25, color=col, lw=2)
        if k in floor:
            ax.axhline(floor[k], color="black", ls=":", lw=1)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=6)
        ax.set_title(title)
    fig.suptitle("final structures, per sample (bar = mean; dotted = crystal model)", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "final_comparison.png"); fig.savefig(p, dpi=150, bbox_inches="tight"); out.append(p)
    return out


def write_pml(B, n_steps):
    L = ["# open from the experiment folder:  pymol outputs_pocket_focus/load_in_pymol.pml",
         "set normalize_ccp4_maps, 0",
         "load gt/8a2b_gt_model.pdb, gt_holo", "load gt/4i20_apo_in_8a2b_frame.pdb, apo_4i20",
         "load gt/8a2b_gt_density.ccp4, gt_map", "isomesh gt_mesh, gt_map, 1.0, resn KY9, carve=3",
         "color grey70, gt_mesh"]
    for tag, col in (("pocket_guided", "skyblue"), ("frozen_unguided", "wheat")):
        for b in range(B):
            L.append(f"load outputs_pocket_focus/final/{tag}_sample{b}.pdb, {tag}_s{b}")
        L.append(f"load outputs_pocket_focus/final/{tag}_sample0.ccp4, {tag}_s0_map")
        L.append(f"isomesh {tag}_s0_mesh, {tag}_s0_map, 1.0, gt_holo and resn KY9, carve=3")
        L += [f"color {col}, {tag}_s0_mesh", f"disable {tag}_s0_mesh", f"color {col}, {tag}_s* and elem C"]
    L += ["hide everything", "show cartoon", "show sticks, resn KY9",
          "show lines, byres (resn KY9 around 5)", "color magenta, gt_holo and resn KY9 and elem C",
          "orient resn KY9", "# checkpoints: outputs_pocket_focus/checkpoints/step*/"]
    open(os.path.join(OUT, "load_in_pymol.pml"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
