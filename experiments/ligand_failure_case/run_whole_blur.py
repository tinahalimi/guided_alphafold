#!/usr/bin/env python
"""
Whole-structure density guidance of 8a2b + KY9 with blur: the protein is NOT frozen.

Same sampler, latents, noise, guidance scale 1.0 (w = 0.1) and superposition as run_guidance.py
(superposed on all mapped protein atoms; the ligand is free). Loss terms are the repo form
sum 0.5*|z(fo) - z(fc)| with z standardised by the target's mean/std on that term's zone.
Blur: B -> B + dB for every atom of BOTH target and model, cutoff rmax(dB) as in the basin scans.

Zones (voxels of the GT box):
  WHOLE0     within 3.5 A of any GT atom                         (the earlier whole-structure zone)
  WHOLE150   within rmax(150) = 6.76 A of any GT atom
  LIG        within 6.76 A of any GT pocket (25 residues) or KY9 atom (= the pocket dB=150 zone)
  PROT       WHOLE0 minus LIG                                     (disjoint from LIG by construction)
In every zone the model density comes from ALL model atoms with that zone's B/cutoff, so protein
atoms reaching LIG are blurred exactly like the target there.

Conditions (all from the same latents):
  whole_dB0           WHOLE0, dB 0                               (baseline, sharp everywhere)
  whole_dB150         WHOLE150, dB 150                           (variant 1)
  two_zone            PROT (dB 0) + lambda * LIG (dB 150)        (variant 2)
  frozen_pocket_dB150 protein frozen at the crystal outside the pocket, LIG dB 150 (reference)

MODE=calibrate  along the unguided whole-structure trajectory, at a few steps: per-atom gradient
                of each term on protein / pocket / KY9 atoms, and the share of the (normalised)
                guidance direction that lands on KY9, for each condition's loss.
MODE=run        the four conditions. LAMBDA = a number (fixed weight of the LIG term) or
                "balance" (per step and sample, lambda = |dPROT/dx| / |dLIG/dx|).

Per-group normalisation (condition name suffix "_alpha<a>", e.g. whole_dB150_alpha0.5): stage 1
normalises the KY9 part and the protein part of dL/dx_noisy separately to unit length and combines
them as step = a * g_hat_KY9 + (1 - a) * g_hat_protein (a = 1: ligand only, a = 0: protein only).
Suffix "_ens" (before _wL, e.g. whole_dB150_ens_wL1.0): ONE loss on the map averaged over the 5 samples,
L = sum 0.5*|z(fo) - z(mean_s fc_s)|, backpropagated to every sample; normalisation stays per sample and
per group. Every run also logs (no grad) that averaged-map loss next to the mean individual loss.
Suffix "_wL<w>": independent weights, step = w * g_hat_KY9 + W_PROT * g_hat_protein (W_PROT env, default 1).
"Protein" = every non-KY9 atom of the sampler (incl. atoms with no crystal counterpart, which get
gradient through the denoiser). Stage 2 (the repo's x ||delta|| over all atoms) is left unchanged:
it multiplies the whole direction by one scalar per sample, so it keeps the split exactly.

Real maps (REAL_MAP=holo | apo, see prepare_real_maps.py): the target fo is the experimental 2mFo-DFc map
(real_maps/8a2b_2fofc_box.ccp4 or real_maps/4i20_2fofc_box_8a2b_frame.ccp4) instead of the forward model of
the deposited model; no blur anywhere. Only the unblurred zones are built: WHOLE0 = within 3.5 A of the
deposited atoms of THAT crystal (8a2b protein + KY9, or 4i20 in the 8a2b frame), fo clipped at 0 as in the
repo. As in the repo's real-map loss, fc includes the crystal-symmetry images of the sampled model (lattice
of that crystal; image atoms kept if their crystal position is within SYM_CUT of the zone atoms), with
gradient through them. Condition "unguided" = same latents, no guidance. RMSDs are also reported vs 4i20.
Outputs: outputs_whole_blur/ (OUT_DIR overrides). wandb group 8a2b_whole_blur (WANDB_GROUP overrides).
"""
import csv
import json
import os
import sys
import time

import gemmi
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_pocket_guidance as rpg  # noqa: E402
from run_pocket_blur import ATP_SITE, ALLOSTERIC, site  # noqa: E402
from forward_model import AtomDensityEstimator, sample_fc  # noqa: E402
from prepare_gt import read_atoms, grid_points, write_ccp4, LIGAND  # noqa: E402
from src.utils.io import load_config  # noqa: E402
from src.protenix.metrics.rmsd import partially_aligned_rmsd  # noqa: E402

OUT = os.path.join(HERE, os.environ.get("OUT_DIR", "outputs_whole_blur"))
GT = rpg.GT
RES_OFFSET = rpg.RES_OFFSET
MODE = os.environ.get("MODE", "run")
LAMBDA = os.environ.get("LAMBDA", "1")
DB = 150.0
CB_WIDEST = 42.0
VOX_CHUNK = 150_000
CAL_STEPS = [10, 50, 90, 110, 130, 150, 170]
WANDB_GROUP = os.environ.get("WANDB_GROUP", "8a2b_whole_blur")
W_PROT = float(os.environ.get("W_PROT", "1.0"))
REAL_MAP = os.environ.get("REAL_MAP", "")
REAL = {"holo": dict(map="real_maps/8a2b_2fofc_box.ccp4", zone="gt/8a2b_gt_model.pdb", xtal="data/8a2b_final.pdb"),
        "apo": dict(map="real_maps/4i20_2fofc_box_8a2b_frame.ccp4", zone="gt/4i20_apo_in_8a2b_frame.pdb",
                    xtal="data/4i20_final.pdb")}
SYM_CUT = 12.0      # 3.5 (zone) + 3.5 (kernel) + 5 A for the model moving away from the crystal positions
HINGE = {718, 719, 792, 793, 794, 795, 796, 797, 841, 842, 844}


def site_hinge(cont):
    """ATP: any hinge contact and <= 2 allosteric; allosteric: no hinge and >= 3 allosteric."""
    h, a = len(HINGE & set(cont)), len(ALLOSTERIC & set(cont))
    return "ATP site" if h and a <= 2 else "allosteric" if not h and a >= 3 else "mixed/between"


def group_direction(grad, lig_mask, w_lig, w_prot):
    """step = w_lig * unit(grad on KY9) + w_prot * unit(grad on the rest), per sample.
    Zero-norm guard as rpg.guidance_direction: a group with zero gradient is left at zero."""
    out = torch.zeros_like(grad)
    for sel, w in ((lig_mask, w_lig), (~lig_mask, w_prot)):
        g = grad[:, sel]
        n = g.flatten(1).norm(dim=-1)
        n = torch.where(n > 0, n, torch.ones_like(n))
        out[:, sel] = w * g / n[:, None, None]
    return out
KEYS = ["lig_rmsd", "lig_centroid", "lig_orient_deg", "lig_internal", "rmsd_pocket", "rmsd_ca", "cc_ligand"]


def main():
    device = "cuda:0"
    cfg = load_config(rpg.CONFIG)
    gcfg = cfg.diffusion_process.guidance
    SCALE = float(gcfg.guidance_direction_scale_factor)
    STEP_SIZE = float(gcfg.step_size)
    START_FROM = int(gcfg.start_guidance_from)
    B = int(cfg.general.batch_size)
    for d in ("final",):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    meta = json.load(open(os.path.join(GT, "meta.json")))
    g = np.load(os.path.join(GT, "grid.npz"))
    lo, n, s, RMAX = g["lo"], g["n"], float(g["spacing"]), float(g["rmax"])
    pts = grid_points(lo, n, s)
    st_gt, gt_atoms = read_atoms(os.path.join(GT, "8a2b_gt_model.pdb"))
    cell, sg = st_gt.cell, st_gt.spacegroup_hm
    prot_key, lig_key = {}, {}
    for a in gt_atoms:
        (lig_key.setdefault(a["name"], a) if a["is_ligand"]
         else prot_key.setdefault((a["resnum"], a["name"]), a))
    pocket = set(meta["pocket_residues"])

    t0 = time.time()
    mgr = rpg.build_generator(cfg, device)
    N = int(cfg.model_manager.diffusion_N)
    print(f"generator built in {time.time()-t0:.0f}s", flush=True)
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
                assert r["resname"] == res_name[i]
                gen_prot.append(i); recs.append(r)
    assert len(gen_lig) == 48
    mapped = gen_prot + gen_lig

    T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=device)
    true_xyz = T([r["xyz"] for r in recs])
    n_p = len(gen_prot)
    pocket_idx = [k for k, r in enumerate(recs[:n_p]) if r["resnum"] in pocket]
    ca_idx = [k for k, i in enumerate(gen_prot) if atom_name[i] == "CA"]
    prot_resnum = np.array([r["resnum"] for r in recs[:n_p]])
    lig_true = true_xyz[n_p:]
    if REAL_MAP:      # 4i20 (in the 8a2b frame) as a second reference for the protein RMSDs
        _, apo_atoms = read_atoms(os.path.join(GT, "4i20_apo_in_8a2b_frame.pdb"))
        apo_key = {}
        for a in apo_atoms:
            apo_key.setdefault((a["resnum"], a["name"]), a)
        has_apo = [k for k, r in enumerate(recs[:n_p]) if (r["resnum"], r["name"]) in apo_key]
        apo_xyz = T([apo_key[(recs[k]["resnum"], recs[k]["name"])]["xyz"] for k in has_apo])
        pocket_apo = [j for j, k in enumerate(has_apo) if recs[k]["resnum"] in pocket]
        ca_apo = [j for j, k in enumerate(has_apo) if recs[k]["name"] == "CA"]
        KEYS.extend(["rmsd_pocket_apo", "rmsd_ca_apo"])
    lig_true_np = lig_true.cpu().numpy()
    lig_lat = torch.tensor(gen_lig, device=device)
    pocket_lat = torch.tensor([gen_prot[k] for k in pocket_idx], device=device)

    est = AtomDensityEstimator(device)
    occ1 = torch.ones(1, len(mapped), device=device)
    bfac1 = T([r["b"] for r in recs])[None]
    el1 = est.get_element_indexes(np.array([r["element"] for r in recs]))[None]
    full_grid = T(pts)

    def near(points, atoms, r):
        out = torch.zeros(len(points), dtype=torch.bool, device=device)
        for c in range(0, len(points), 200000):
            out[c:c + 200000] = torch.cdist(points[c:c + 200000], atoms).min(1)[0] < r
        return out
    G_xyz = T([a["xyz"] for a in gt_atoms]); G_b = T([a["b"] for a in gt_atoms])
    G_occ = T([a["occ"] for a in gt_atoms])
    G_el = est.get_element_indexes(np.array([a["element"] for a in gt_atoms]))
    roi_gt = T([a["xyz"] for a in gt_atoms if a["is_ligand"] or a["resnum"] in pocket])
    Bmed = float(np.median([a["b"] for a in gt_atoms]))
    R150 = RMAX * float(np.sqrt((Bmed + CB_WIDEST + DB) / (Bmed + CB_WIDEST)))

    def make_zone(mask, db, rmax, name):
        zp = full_grid[mask]
        with torch.no_grad():
            fo = torch.cat([sample_fc(est, G_xyz[None], G_occ[None], G_el[None], (G_b + db)[None],
                                      zp[None, c:c + VOX_CHUNK], rmax)[0]
                            for c in range(0, len(zp), VOX_CHUNK)])
        print(f"  zone {name:9s}: {len(zp):>9,} voxels, dB {db:3.0f}, cutoff {rmax:.2f} A", flush=True)
        return dict(name=name, pts=zp, fo=fo, mu=fo.mean(), sd=fo.std() + 1e-6, db=db, rmax=rmax,
                    lig=near(zp, lig_true, RMAX if db == 0 else rmax))

    if REAL_MAP:
        rc = REAL[REAL_MAP]
        mp = gemmi.read_ccp4_map(os.path.join(HERE, rc["map"])); mp.setup(float("nan"))
        box = np.zeros(tuple(int(x) for x in n), dtype=np.float32)
        mp.grid.interpolate_values(box, gemmi.Transform(gemmi.Mat33((np.eye(3) * s).tolist()), gemmi.Vec3(*lo)), order=1)
        fo_full = T(box.ravel()).clamp_min(0)                      # repo: fo.clip(0)
        _, z_atoms = read_atoms(os.path.join(HERE, rc["zone"]))
        Zxyz = T([a["xyz"] for a in z_atoms])

        def make_zone(mask, db, rmax, name):
            zp, fo = full_grid[mask], fo_full[mask]
            print(f"  zone {name:9s}: {len(zp):>9,} voxels, REAL {REAL_MAP} map, fo mean {fo.mean():.3f} sd {fo.std():.3f}"
                  f" max {fo.max():.2f}", flush=True)
            return dict(name=name, pts=zp, fo=fo, mu=fo.mean(), sd=fo.std() + 1e-6, db=0.0, rmax=rmax,
                        lig=near(zp, lig_true, RMAX))
        Z = dict(WHOLE0=make_zone(near(full_grid, Zxyz, RMAX), 0.0, RMAX, "WHOLE0"))
        MZ = make_zone(near(full_grid, roi_gt, RMAX), 0.0, RMAX, "metrics")

    m_whole0 = near(full_grid, G_xyz, RMAX)
    m_lig = near(full_grid, roi_gt, R150)
    Z = Z if REAL_MAP else dict(WHOLE0=make_zone(m_whole0, 0.0, RMAX, "WHOLE0"),
             WHOLE150=make_zone(near(full_grid, G_xyz, R150), DB, R150, "WHOLE150"),
             LIG=make_zone(m_lig, DB, R150, "LIG"),
             PROT=make_zone(m_whole0 & ~m_lig, 0.0, RMAX, "PROT"))
    assert not bool((m_lig & (m_whole0 & ~m_lig)).any())
    # unblurred pocket zone of the earlier runs, for the ligand-region CC only
    m_pk0 = near(full_grid, roi_gt, RMAX)
    MZ = MZ if REAL_MAP else make_zone(m_pk0, 0.0, RMAX, "metrics")

    # crystal-symmetry images of the model (real maps only): image = M x + v per image atom, fixed (M, v)
    ext = lambda x: x
    if REAL_MAP:
        from scipy.spatial import cKDTree
        xst = gemmi.read_structure(os.path.join(HERE, rc["xtal"]))
        if REAL_MAP == "apo":     # 4i20 frame <-> 8a2b frame, the superposition of gt/4i20_apo_in_8a2b_frame.pdb
            from prepare_gt import kabsch
            P = np.array([a.pos.tolist() for ch in xst[0] for r in ch if r.name != "HOH" for a in r
                          if a.element.name not in ("H", "D")])
            Rf, tf = kabsch(P, Zxyz.cpu().numpy())
        else:
            Rf, tf = np.eye(3), np.zeros(3)
        O, F = np.array(xst.cell.orth.mat), np.array(xst.cell.frac.mat)
        ref = true_xyz.cpu().numpy()
        tree = cKDTree(Zxyz.cpu().numpy())
        Ms, vs, src = [], [], []
        for op in gemmi.SpaceGroup(xst.spacegroup_hm).operations():
            Rop, top = np.array(op.rot) / op.DEN, np.array(op.tran) / op.DEN
            for sh in np.array(np.meshgrid(*[range(-2, 3)] * 3)).reshape(3, -1).T:
                if np.allclose(Rop, np.eye(3)) and not sh.any():
                    continue
                Mc, vc = O @ Rop @ F, O @ (top + sh)           # crystal frame
                M = Rf @ Mc @ Rf.T                              # 8a2b frame: x -> Rf (Mc (Rf^T (x - tf)) + vc) + tf
                v = Rf @ (vc - Mc @ Rf.T @ tf) + tf
                keep = np.flatnonzero(tree.query(ref @ M.T + v, distance_upper_bound=SYM_CUT)[0] < SYM_CUT)
                if len(keep):
                    Ms.append(np.repeat(M[None], len(keep), 0)); vs.append(np.repeat(v[None], len(keep), 0)); src.append(keep)
        sym_M, sym_v, sym_src = T(np.concatenate(Ms)), T(np.concatenate(vs)), torch.tensor(np.concatenate(src), device=device)
        ext = lambda x: torch.cat([x, torch.einsum("mij,bmj->bmi", sym_M, x[:, sym_src]) + sym_v], dim=1)
        occ1, el1, bfac1 = (torch.cat([t_, t_[:, sym_src]], dim=1) for t_ in (occ1, el1, bfac1))
        print(f"  symmetry ({xst.spacegroup_hm}, {len(Ms)} images): {len(sym_src):,} image atoms within {SYM_CUT} A "
              f"of the zone atoms (model has {len(mapped):,})", flush=True)

    def zone_loss_grad(c, zn):
        """c [b, n_mapped, 3] (detached) -> per-sample loss [b] and dL/dc [b, n_mapped, 3]."""
        Ls = torch.zeros(c.shape[0], device=device)
        Gc = torch.zeros_like(c)
        for b in range(c.shape[0]):
            x = c[b:b + 1].detach().clone().requires_grad_(True)
            for s0 in range(0, len(zn["pts"]), VOX_CHUNK):
                fc = sample_fc(est, ext(x), occ1, el1, bfac1 + zn["db"], zn["pts"][None, s0:s0 + VOX_CHUNK],
                               zn["rmax"])[0]
                l = (0.5 * ((zn["fo"][s0:s0 + VOX_CHUNK] - fc) / zn["sd"]).abs()).sum()
                l.backward()
                Ls[b] += l.detach()
            Gc[b] = x.grad[0]
        return Ls, Gc

    ENS_CHUNK = 40_000

    def ens_stats(c, zn, grad=False):
        """Averaged-map loss and individual losses on zone zn (target stats as the individual loss).
        grad=True: also dL_avg/dc [b, n, 3] (the ensemble guidance gradient)."""
        x = c.detach().clone().requires_grad_(grad)
        b = c.shape[0]
        L_avg = torch.zeros((), device=device); L_ind = torch.zeros(b, device=device)
        with torch.enable_grad() if grad else torch.no_grad():
            for s0 in range(0, len(zn["pts"]), ENS_CHUNK):
                fc = sample_fc(est, ext(x), occ1.expand(b, -1), el1.expand(b, -1), bfac1.expand(b, -1) + zn["db"],
                               zn["pts"][None, s0:s0 + ENS_CHUNK].expand(b, -1, -1), zn["rmax"])
                fo = zn["fo"][s0:s0 + ENS_CHUNK]
                la = (0.5 * ((fo - fc.mean(0)) / zn["sd"]).abs()).sum()
                if grad:
                    la.backward()
                L_avg += la.detach()
                L_ind += (0.5 * ((fo - fc.detach()) / zn["sd"]).abs()).sum(-1)
        return L_avg, L_ind, (x.grad if grad else None)

    tgt = torch.zeros(B, len(aa), 3, device=device)
    tgt[:, gen_prot] = true_xyz[:n_p][None].expand(B, -1, -1).clone()
    msk_all = torch.zeros(B, len(aa), device=device)
    msk_all[:, gen_prot] = 1.0                                       # whole-structure runs
    roi = np.zeros(len(mapped), bool); roi[pocket_idx] = True; roi[n_p:] = True
    frozen = torch.tensor(~roi, device=device)
    msk_frz = torch.zeros(B, len(aa), device=device)
    msk_frz[:, [mapped[k] for k in np.flatnonzero(~roi)]] = 1.0      # frozen reference
    mapped_t = torch.tensor(mapped, device=device)

    def x0_aligned(lat, i, freeze):
        x0 = mgr.get_x_0_hat_from_x_noisy(lat, start_index=i, inplace_safe=False)
        msk = msk_frz if freeze else msk_all
        out = partially_aligned_rmsd(pred_pose=x0, true_pose=tgt, align_mask=msk, atom_mask=msk, reduce=True)
        al = out[2]
        c = al[:, mapped_t]
        if freeze:
            c = torch.where(frozen[None, :, None], true_xyz[None], c)
            al = al.index_copy(1, mapped_t, c)
        return al, c, out[3], out[4]

    def metrics(c):
        with torch.no_grad():
            d2 = ((c - true_xyz) ** 2).sum(-1)
            fc = torch.cat([sample_fc(est, ext(c), occ1.expand(len(c), -1), el1.expand(len(c), -1),
                                      bfac1.expand(len(c), -1), MZ["pts"][None, s0:s0 + VOX_CHUNK]
                                      .expand(len(c), -1, -1), RMAX)
                            for s0 in range(0, len(MZ["pts"]), VOX_CHUNK)], dim=1)
            m = dict(rmsd_pocket=d2[:, pocket_idx].mean(-1).sqrt(), rmsd_ca=d2[:, ca_idx].mean(-1).sqrt(),
                     cc_ligand=rpg.pearson(fc[:, MZ["lig"]], MZ["fo"][MZ["lig"]]))
            if REAL_MAP:
                da = ((c[:, has_apo] - apo_xyz) ** 2).sum(-1)
                m.update(rmsd_pocket_apo=da[:, pocket_apo].mean(-1).sqrt(), rmsd_ca_apo=da[:, ca_apo].mean(-1).sqrt())
            m = {k: v.cpu().numpy().astype(float) for k, v in m.items()}
            cl = c[:, n_p:].cpu().numpy()
            lm = [rpg.ligand_metrics(cl[b], lig_true_np) for b in range(len(c))]
            for k in lm[0]:
                m[k] = np.array([x[k] for x in lm])
        return m

    def loss_grad(cond, c):
        """-> total loss [b], dL/dc [b, n, 3], per-term grads (for logging), lambda [b]."""
        if cond == "whole_dB0":
            L, G = zone_loss_grad(c, Z["WHOLE0"]); return L, G, {}, None
        if cond == "whole_dB150":
            L, G = zone_loss_grad(c, Z["WHOLE150"]); return L, G, {}, None
        if cond == "frozen_pocket_dB150":
            L, G = zone_loss_grad(c, Z["LIG"]); return L, G, {}, None
        Lp, Gp = zone_loss_grad(c, Z["PROT"])
        Ll, Gl = zone_loss_grad(c, Z["LIG"])
        if LAMBDA == "balance":
            lam = Gp.flatten(1).norm(dim=-1) / Gl.flatten(1).norm(dim=-1).clamp_min(1e-12)
        else:
            lam = torch.full((c.shape[0],), float(LAMBDA), device=device)
        return Lp + lam * Ll, Gp + lam[:, None, None] * Gl, dict(PROT=Gp, LIG=Gl), lam

    rms_on = lambda G, idx: G[:, idx].norm(dim=-1).pow(2).mean(-1).sqrt()     # per-atom RMS, [b]
    prot_all = list(range(n_p))

    lat0 = mgr.get_x_noisy(mgr.get_x_start(B)).detach().clone()
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all()

    def reset():
        torch.cuda.set_rng_state_all(rng_cuda); torch.set_rng_state(rng_cpu)

    # ================================================================== calibration
    if MODE == "calibrate":
        reset()
        lat = lat0.clone()
        rows = []
        for i in range(max(CAL_STEPS) + 1):
            if i in CAL_STEPS:
                row = dict(step=i, t_hat=float(mgr.get_t_hat(i)))
                for cond, freeze in (("whole_dB0", False), ("whole_dB150", False),
                                     ("two_zone", False), ("frozen_pocket_dB150", True)):
                    lat_g = lat.clone().requires_grad_(True)
                    al, c, _, _ = x0_aligned(lat_g, i, freeze)
                    _, G, terms, lam = loss_grad(cond, c.detach())
                    for tn, Gt in ([("total", G)] + list(terms.items())):
                        for grp, idx in (("protein", prot_all), ("pocket", pocket_idx), ("KY9", list(range(n_p, len(mapped))))):
                            row[f"{cond}/{tn}/dLdx_rms_{grp}"] = float(rms_on(Gt, idx).mean())
                        row[f"{cond}/{tn}/norm"] = float(Gt.flatten(1).norm(dim=-1).mean())
                    if terms:
                        # KY9 share of the latent direction for: PROT alone, LIG alone, lambda=1, balanced
                        lat_dirs = {}
                        for tn, Gt in terms.items():
                            lat_g.grad = None
                            c.backward(Gt, retain_graph=True)
                            lat_dirs[tn] = lat_g.grad.clone()
                        bal = terms["PROT"].flatten(1).norm(dim=-1) / terms["LIG"].flatten(1).norm(dim=-1)
                        row[f"{cond}/balance_lambda"] = [float(x) for x in bal]
                        combos = dict(PROT_only=lat_dirs["PROT"], LIG_only=lat_dirs["LIG"],
                                      lambda1=lat_dirs["PROT"] + lat_dirs["LIG"],
                                      balanced=lat_dirs["PROT"] + bal[:, None, None] * lat_dirs["LIG"])
                    else:
                        lat_g.grad = None
                        c.backward(G)
                        combos = dict(total=lat_g.grad.clone())
                    for k, v in combos.items():
                        u = v / v.flatten(1).norm(dim=-1)[:, None, None]
                        row[f"{cond}/{k}/u_share_KY9"] = float(u[:, lig_lat].flatten(1).norm(dim=-1).mean())
                        row[f"{cond}/{k}/u_share_pocket"] = float(u[:, pocket_lat].flatten(1).norm(dim=-1).mean())
                rows.append(row)
                print(f"\nstep {i} (t_hat {row['t_hat']:.2f})", flush=True)
                for k, v in row.items():
                    if "/" in k:
                        print(f"   {k:45s} {v if isinstance(v, list) else round(v, 5)}")
            with torch.no_grad():
                al, c, R, Tr = x0_aligned(lat, i, False)
                nx = (R.permute(0, 2, 1)[:, None] @ (al - Tr)[..., None]).squeeze(-1)
                lat = mgr.get_x_t_from_x_0_hat(lat, nx, i, i + 1)
                lat = mgr.get_x_noisy(lat, i, i + 1).detach().clone()
        json.dump(rows, open(os.path.join(OUT, "calibration.json"), "w"), indent=1)
        print(f"wrote {OUT}/calibration.json")
        return

    # ================================================================== runs
    b_out = np.zeros(len(aa)); occ_out = np.zeros(len(aa))
    b_out[mapped] = [r["b"] for r in recs]; occ_out[mapped] = 1.0

    def write_model(path, xyz):
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

    def write_map(path, cm):
        with torch.no_grad():
            v = torch.cat([sample_fc(est, ext(cm[None]), occ1, el1, bfac1, full_grid[None, s0:s0 + VOX_CHUNK], RMAX)[0]
                           for s0 in range(0, len(full_grid), VOX_CHUNK)])
        write_ccp4(path, v.cpu().numpy(), lo, n, s)

    import wandb
    wb_mode = os.environ.get("WANDB_MODE", "offline")

    lig_mask = torch.zeros(len(aa), dtype=torch.bool, device=device)
    lig_mask[lig_lat] = True

    def run(cond):
        if "_wL" in cond:
            base, _, a = cond.partition("_wL")
            weights = (float(a), W_PROT)
        else:
            base, _, a = cond.partition("_alpha")
            weights = (float(a), 1.0 - float(a)) if a else None
        alpha = weights
        ens = base.endswith("_ens")
        base = base.removesuffix("_ens")
        freeze = base == "frozen_pocket_dB150"
        ens_zone = Z["WHOLE0"] if base in ("whole_dB0", "unguided") else Z["LIG"] if base == "frozen_pocket_dB150" else Z["WHOLE150"]
        wr = wandb.init(entity=rpg.WANDB_ENTITY, project=rpg.WANDB_PROJECT, group=WANDB_GROUP, name=cond,
                        job_type=cond, mode=wb_mode, reinit=True,
                        config=dict(guidance_scale=SCALE, step_size=STEP_SIZE, frozen=freeze, db_ligand_zone=DB,
                                    lambda_lig=LAMBDA if base == "two_zone" else None,
                                    normalisation="per-group" if alpha is not None else "global", alpha=alpha,
                                    zones={k: int(len(v["pts"])) for k, v in Z.items()}, batch=B,
                                    real_map=REAL[REAL_MAP]["map"] if REAL_MAP else None,
                                    n_symmetry_image_atoms=int(len(sym_src)) if REAL_MAP else 0))
        reset()
        lat = lat0.clone()
        trace = []
        t1 = time.time()
        for i in range(N):
            guide = i > START_FROM and i != N - 1 and base != "unguided"
            if guide:
                lat.requires_grad_(True)
            with (torch.enable_grad() if guide else torch.no_grad()):
                al, c, R, Tr = x0_aligned(lat, i, freeze)
            gd, extra = None, {}
            if guide:
                if ens:
                    L_avg, L_ind, G = ens_stats(c, ens_zone, grad=True)
                    terms, lam = {}, None
                else:
                    L, G, terms, lam = loss_grad(base, c.detach())
                c.backward(G)
                grad = lat.grad
                assert grad is not None and float(grad.abs().max()) > 0, "guidance is dead"
                with torch.no_grad():
                    gd = (rpg.guidance_direction(grad, SCALE) if alpha is None
                          else SCALE * group_direction(grad, lig_mask, *weights))
                    # share of the APPLIED direction on KY9 (= the old u_share for the global norm)
                    gn = gd.flatten(1).norm(dim=-1)
                    share = gd[:, lig_lat].flatten(1).norm(dim=-1) / gn
                    # actual displacements after stage 2: guidance vs the sampler's own step, per group
                    nx_ = (R.permute(0, 2, 1)[:, None] @ (al.detach() - Tr)[..., None]).squeeze(-1)
                    delta = (lat.detach() - nx_) / float(mgr.get_t_hat(i))
                    disp = STEP_SIZE * delta.flatten(1).norm(dim=-1)[:, None, None] * gd     # as stage 2
                    rmsg = lambda v, sel: v[:, sel].norm(dim=-1).pow(2).mean(-1).sqrt()
                    extra = {"u_share_KY9/mean": float(share.mean()),
                             **{f"u_share_KY9/s{b}": float(share[b]) for b in range(B)},
                             "guid_over_sampler_KY9/mean": float((rmsg(disp, lig_mask) / rmsg(delta, lig_mask)).mean()),
                             "guid_over_sampler_protein/mean": float((rmsg(disp, ~lig_mask) / rmsg(delta, ~lig_mask)).mean()),
                             "guid_KY9_per_atom_A/mean": float(rmsg(disp, lig_mask).mean()),
                             "dir_norm/mean": float(gn.mean()),
                             "u_share_pocket/mean": float((gd[:, pocket_lat].flatten(1).norm(dim=-1) / gn).mean()),
                             "dLdx_rms_protein/mean": float(rms_on(G, prot_all).mean()),
                             "dLdx_rms_KY9/mean": float(rms_on(G, list(range(n_p, len(mapped)))).mean())}
                    if lam is not None:
                        extra["lambda/mean"] = float(lam.mean())
                        for tn, Gt in terms.items():
                            extra[f"{tn}_dLdx_rms_protein/mean"] = float(rms_on(Gt, prot_all).mean())
                            extra[f"{tn}_dLdx_rms_KY9/mean"] = float(rms_on(Gt, list(range(n_p, len(mapped)))).mean())
                lat.grad = None
            c = c.detach(); al = al.detach()
            if not (guide and ens):
                L_avg, L_ind, _ = ens_stats(c, ens_zone)
            cen = c[:, n_p:].mean(1)
            iu = torch.triu_indices(B, B, 1, device=device)
            pd = (cen[:, None] - cen[None]).norm(dim=-1)[iu[0], iu[1]]
            pr = (c[:, None, n_p:] - c[None, :, n_p:]).norm(dim=-1).pow(2).mean(-1).sqrt()[iu[0], iu[1]]
            extra.update({"ens_loss_avgmap": float(L_avg), "ens_loss_ind_mean": float(L_ind.mean()),
                          "ens_loss_ratio": float(L_avg / L_ind.mean()),
                          "pose_spread_centroid_mean": float(pd.mean()), "pose_spread_centroid_max": float(pd.max()),
                          "pose_spread_rmsd_mean": float(pr.mean())})
            m = metrics(c)
            row = dict(step=i, t_hat=float(mgr.get_t_hat(i)),
                       **{f"{k}/s{b}": float(v[b]) for k, v in m.items() for b in range(B)},
                       **{f"{k}/mean": float(v.mean()) for k, v in m.items()}, **extra)
            trace.append(row)
            wr.log(row, step=i)
            if i % 10 == 0 or i == N - 1:
                print(f"  {cond:20s} {i:3d} t {row['t_hat']:7.2f}  KY9 {m['lig_rmsd'].mean():5.2f}  centroid "
                      f"{m['lig_centroid'].mean():5.2f}  pocket {m['rmsd_pocket'].mean():5.2f}  CA {m['rmsd_ca'].mean():5.2f}"
                      + (f"  share KY9 {extra['u_share_KY9/mean']:.2f}  guid/sampler KY9 "
                         f"{extra['guid_over_sampler_KY9/mean']:.2f} prot {extra['guid_over_sampler_protein/mean']:.2f}"
                         if "u_share_KY9/mean" in extra else "")
                      + f"  spread {extra['pose_spread_centroid_mean']:.2f}  Lavg/Lind {extra['ens_loss_ratio']:.3f}"
                      + (f"  lambda {extra['lambda/mean']:.3g}" if "lambda/mean" in extra else ""), flush=True)
            with torch.no_grad():
                nx = (R.permute(0, 2, 1)[:, None] @ (al - Tr)[..., None]).squeeze(-1)
            lat = mgr.get_x_t_from_x_0_hat(lat.detach(), nx, i, i + 1, guidance_direction=gd, step_size=STEP_SIZE,
                                           normalize_gradients=bool(gcfg.normalize_gradients))
            if i < N - 1:
                lat = mgr.get_x_noisy(lat, i, i + 1)
            lat = lat.detach().clone()
        wr.summary.update(dict(seconds=time.time() - t1))
        wr.finish()
        print(f"  [{cond}] {N} steps in {time.time()-t1:.0f}s", flush=True)
        # unblurred map CC vs the target map (GT, or the real map): averaged map and each sample, whole map + KY9 region
        zw = Z["WHOLE0"]
        with torch.no_grad():
            fcw = torch.cat([sample_fc(est, ext(c), occ1.expand(B, -1), el1.expand(B, -1), bfac1.expand(B, -1),
                                       zw["pts"][None, s0:s0 + ENS_CHUNK].expand(B, -1, -1), RMAX)
                             for s0 in range(0, len(zw["pts"]), ENS_CHUNK)], dim=1)
            ccs = {}
            for reg, sel in (("whole", slice(None)), ("KY9", zw["lig"])):
                fo_r = zw["fo"][sel]
                ccs[f"cc_{reg}_avgmap"] = float(rpg.pearson(fcw.mean(0)[sel], fo_r))
                ccs[f"cc_{reg}_per_sample"] = [float(x) for x in rpg.pearson(fcw[:, sel], fo_r)]
            write_ccp4(os.path.join(OUT, "final", f"{cond}_avgmap.ccp4"),
                       torch.cat([sample_fc(est, ext(c), occ1.expand(B, -1), el1.expand(B, -1), bfac1.expand(B, -1),
                                            full_grid[None, s0:s0 + ENS_CHUNK].expand(B, -1, -1), RMAX).mean(0)
                                  for s0 in range(0, len(full_grid), ENS_CHUNK)]).cpu().numpy(), lo, n, s)
        trace[-1]["final_cc"] = ccs
        print(f"  [{cond}] unblurred CC: " + ", ".join(f"{k} {v if isinstance(v, float) else [round(x, 3) for x in v]}"
                                                   for k, v in ccs.items()), flush=True)
        cn = c.cpu().numpy()
        contacts = []
        for b in range(B):
            write_model(os.path.join(OUT, "final", f"{cond}_sample{b}.pdb"), al[b].cpu().numpy())
            write_map(os.path.join(OUT, "final", f"{cond}_sample{b}.ccp4"), c[b])
            dmin = np.linalg.norm(cn[b, :n_p, None] - cn[b, None, n_p:], axis=-1).min(1)
            contacts.append(sorted({int(r) for r in prot_resnum[dmin < 4.0]}))
        return trace, contacts

    conds = ["whole_dB0", "whole_dB150", "two_zone", "frozen_pocket_dB150"]
    if os.environ.get("CONDS"):
        conds = os.environ["CONDS"].split(",")
    res = {}
    for cond in conds:
        print(f"\n{cond}", flush=True)
        res[cond] = run(cond)
    resname = {r["resnum"]: r["resname"] for r in recs[:n_p]}
    analyse(res, B, resname, {k: int(len(v["pts"])) for k, v in Z.items()})


def analyse(res, B, resname, zones):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = lambda r: f"{resname.get(r, '?')[:1]}{resname.get(r, '?')[1:].lower()}{r}"
    out = dict(zones=zones, lambda_lig=LAMBDA, conditions={})
    for cond, (tr, cont) in res.items():
        fin = tr[-1]
        per = [dict(sample=b, **{k: fin[f"{k}/s{b}"] for k in KEYS}, site=site(cont[b]),
                    site_hinge_rule=site_hinge(cont[b]),
                    mean_share_KY9=float(np.nanmean([r.get(f"u_share_KY9/s{b}", np.nan) for r in tr[1:-1]]))
                    if any(f"u_share_KY9/s{b}" in r for r in tr) else None,
                    atp_contacts=[name(r) for r in cont[b] if r in ATP_SITE],
                    allosteric_contacts=[name(r) for r in cont[b] if r in ALLOSTERIC],
                    contacts=[name(r) for r in cont[b]]) for b in range(B)]
        out["conditions"][cond] = dict(per_sample=per, mean={k: float(np.mean([p[k] for p in per])) for k in KEYS},
                                       final_cc=fin.get("final_cc"),
                                       final_ens_loss={k: fin.get(k) for k in ("ens_loss_avgmap", "ens_loss_ind_mean",
                                                                              "ens_loss_ratio", "pose_spread_centroid_mean",
                                                                              "pose_spread_centroid_max", "pose_spread_rmsd_mean")},
                                       n_allosteric=sum(p["site_hinge_rule"] == "allosteric" for p in per))
        print(f"\n{cond}")
        for p in per:
            print(f"  s{p['sample']}: KY9 rmsd {p['lig_rmsd']:5.2f}  centroid {p['lig_centroid']:5.2f}  orient "
                  f"{p['lig_orient_deg']:5.1f}  internal {p['lig_internal']:.2f}  pocket {p['rmsd_pocket']:.2f}  "
                  f"CA {p['rmsd_ca']:.2f}  CClig {p['cc_ligand']:.3f} -> {p['site_hinge_rule']:13s} "
                  f"ATP[{' '.join(p['atp_contacts'])}] allo[{' '.join(p['allosteric_contacts'])}]")
        print("  mean: " + "  ".join(f"{k} {v:.3f}" for k, v in out["conditions"][cond]["mean"].items()))
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for cond, (tr, _) in res.items():
        tr = [{k: v for k, v in r.items() if k != "final_cc"} for r in tr]
        keys = sorted({k for r in tr for k in r})
        with open(os.path.join(OUT, f"trace_{cond}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(tr)

    col = {"whole_dB0": "#2a78d6", "whole_dB150": "#1baf7a", "two_zone": "#6a3d9a", "frozen_pocket_dB150": "#d6402a"}
    alphas = [c for c in res if "_alpha" in c or "_wL" in c]
    col.update({c: plt.cm.plasma(x) for c, x in zip(alphas, np.linspace(0.05, 0.85, max(1, len(alphas))))})
    if alphas:
        col["whole_dB150"] = "#555555"
    panels = [("lig_rmsd", "KY9 RMSD vs crystal (A)"), ("rmsd_pocket", "pocket heavy-atom RMSD (A)"),
              ("lig_centroid", "KY9 centroid offset (A)"), ("rmsd_ca", "whole-protein CA RMSD (A)"),
              ("lig_orient_deg", "KY9 orientation error (deg)"), ("u_share_KY9", "share of guidance direction on KY9"),
              ("guid_over_sampler_KY9", "KY9: guidance step / sampler step"),
              ("guid_over_sampler_protein", "protein: guidance step / sampler step"),
              ("pose_spread_centroid_mean", "KY9 pose spread: mean pairwise centroid distance (A)"),
              ("pose_spread_rmsd_mean", "KY9 pose spread: mean pairwise RMSD (A)"),
              ("ens_loss_ratio", "averaged-map loss / mean individual loss (target zone)")]
    fig, axes = plt.subplots(3, 4, figsize=(21, 13))
    for ax, (k, title) in zip(axes.flat, panels):
        for cond, (tr, _) in res.items():
            st = [r["step"] for r in tr]
            if k.startswith(("pose_spread", "ens_loss")):
                ax.plot(st, [r.get(k, np.nan) for r in tr], color=col.get(cond, "k"), lw=1.5, label=cond)
                continue
            if k in ("u_share_KY9", "guid_over_sampler_KY9", "guid_over_sampler_protein"):
                ax.plot(st, [r.get(f"{k}/mean", np.nan) for r in tr], color=col.get(cond, "k"), lw=1.5, label=cond)
                continue
            for b in range(B):
                ax.plot(st, [r[f"{k}/s{b}"] for r in tr], color=col.get(cond, "k"), lw=0.5, alpha=0.3)
            ax.plot(st, [r[f"{k}/mean"] for r in tr], color=col.get(cond, "k"), lw=2, label=cond)
        ax.set_title(title); ax.set_xlabel("diffusion step"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax in list(axes.flat[5:8]) + [axes.flat[8], axes.flat[9]]:
        ax.set_yscale("log")
    axes.flat[-1].set_visible(False)
    fig.suptitle((f"8a2b + KY9 whole-structure guidance on the REAL {REAL_MAP} 2mFo-DFc map, no blur, "
                  f"w_P = {W_PROT:g} for _wL runs; " if REAL_MAP else
                  f"8a2b + KY9 whole-structure dB=150, per-group normalisation, w_P = {W_PROT:g} for _wL runs "
                  "(grey = global norm); " if alphas else
                  f"8a2b + KY9 whole-structure guidance with blur (scale 1.0, two-zone lambda = {LAMBDA}); ") +
                 "thin = samples, thick = mean of 5")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=140)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
