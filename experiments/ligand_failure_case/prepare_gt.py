#!/usr/bin/env python
"""
Ground truth for the ligand-failure-case experiment (EGFR kinase, 8a2b holo + KY9, 4i20 apo).

Everything is expressed in ONE Cartesian frame, the deposited 8a2b crystal frame:

  gt/8a2b_gt_model.pdb            the atoms the GT map is computed from: 8a2b chain A protein
                                  + KY9. No waters, no hydrogens. Altlocs kept with their
                                  deposited occupancies; deposited B-factors.
  gt/8a2b_gt_ligand.pdb           KY9 alone (48 heavy atoms)
  gt/8a2b_gt_density.ccp4         ground-truth density = the guided-AlphaFold forward model
                                  (sample_fc_from_all_atom, see forward_model.py) of
                                  8a2b_gt_model.pdb on a Cartesian box grid
  gt/4i20_apo_in_8a2b_frame.pdb   deposited 4i20 (apo), superposed onto 8a2b on CA atoms
  gt/grid.npz, gt/meta.json       grid definition, pocket residues, checks

Grid: spacing = resolution(8a2b)/3, box = bounding box of the GT atoms + BOX_PAD, with the
origin snapped to a multiple of the spacing so the CCP4 file can carry it as an integer
NSTART. Every predicted-structure map in this experiment is written on this same grid.

No crystal symmetry, no bulk solvent, no resolution truncation, no noise: the GT is the
forward model applied to the deposited holo model, so the guidance loop compares like with
like.

Run (head-node GPU is fine with guided_af3, ~1 min):
    python prepare_gt.py
"""
import json
import os

import gemmi
import numpy as np
import torch

from forward_model import AtomDensityEstimator, sample_fc, check_against_repo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GT = os.path.join(HERE, "gt")
HOLO_PDB = os.path.join(DATA, "8a2b_final.pdb")
APO_PDB = os.path.join(DATA, "4i20_final.pdb")
LIGAND = "KY9"
RMAX = 3.5          # forward-model kernel cutoff (A), as in the earlier Fc maps / steps
BOX_PAD = 4.5       # > RMAX, so no density is cut off at the box edge
POCKET_CUTOFF = 5.0  # protein residue with any heavy atom within this of KY9 -> pocket


# ---------------------------------------------------------------------------------------
# shared helpers (imported by run_guidance.py)
# ---------------------------------------------------------------------------------------
def read_atoms(path, ligand=LIGAND):
    """Heavy, non-water atoms of the first model as a list of dicts (altlocs kept)."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    out = []
    for ch in st[0]:
        for r in ch:
            if r.name == "HOH":
                continue
            for a in r:
                if a.element.name in ("H", "D"):
                    continue
                out.append(dict(chain=ch.name, resname=r.name, resnum=r.seqid.num,
                                name=a.name, altloc=a.altloc if a.altloc != "\x00" else "",
                                element=a.element.name, xyz=np.array(a.pos.tolist()),
                                b=float(a.b_iso), occ=float(a.occ),
                                is_ligand=(r.name == ligand)))
    return st, out


def grid_points(lo, n, s):
    axes = [lo[i] + s * np.arange(n[i]) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)


def write_ccp4(path, values, lo, n, s):
    """Write a box map so that voxel (u,v,w) sits at lo + s*(u,v,w) in Cartesian A.

    P1 cell = the box itself (n*s), origin carried as integer NSTART = lo/s. Opens in the
    right place in PyMOL, ChimeraX and Coot next to the PDBs, with nothing to re-fit.
    """
    start = np.rint(np.asarray(lo) / s).astype(int)
    assert np.allclose(start * s, lo, atol=1e-4), "grid origin is not a multiple of spacing"
    g = gemmi.FloatGrid(int(n[0]), int(n[1]), int(n[2]))
    g.set_unit_cell(gemmi.UnitCell(n[0] * s, n[1] * s, n[2] * s, 90, 90, 90))
    g.spacegroup = gemmi.find_spacegroup_by_name("P1")
    arr = np.array(g, copy=False)
    arr[:] = np.asarray(values, dtype=np.float32).reshape(tuple(n))
    m = gemmi.Ccp4Map()
    m.grid = g
    m.update_ccp4_header(2, True)
    for k, v in zip((5, 6, 7), start):
        m.set_header_i32(k, int(v))
    m.write_ccp4_map(path)


def check_ccp4(path, values, pts, rng, n_check=2000):
    """Read the file back and interpolate at true voxel positions: must equal the values."""
    m = gemmi.read_ccp4_map(path)
    m.setup(float("nan"))
    idx = rng.choice(len(pts), n_check, replace=False)
    got = np.array([m.grid.interpolate_value(gemmi.Position(*map(float, pts[i]))) for i in idx])
    return float(np.abs(got - np.asarray(values)[idx]).max())


def kabsch(P, Q):
    """R, t with P @ R.T + t ~= Q."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - pc @ R.T


# ---------------------------------------------------------------------------------------
def main():
    os.makedirs(GT, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    checks = {}

    # ---- GT model --------------------------------------------------------------------
    st, atoms = read_atoms(HOLO_PDB)
    res = float(st.resolution)
    s = res / 3.0
    lig = [a for a in atoms if a.get("is_ligand")]
    prot = [a for a in atoms if not a["is_ligand"]]
    other = sorted({a["resname"] for a in atoms
                    if not a["is_ligand"] and gemmi.find_tabulated_residue(a["resname"]) is None})
    print(f"8a2b: resolution {res:.2f} A -> spacing {s:.4f} A; {len(prot)} protein atoms, "
          f"{len(lig)} {LIGAND} atoms; non-standard residues kept: {other or 'none'}")
    assert len(lig) == 48, f"expected 48 KY9 heavy atoms, got {len(lig)}"

    gt_st = gemmi.read_structure(HOLO_PDB)
    gt_st.remove_waters()
    gt_st.remove_hydrogens()
    gt_st.remove_empty_chains()
    gt_st.write_pdb(os.path.join(GT, "8a2b_gt_model.pdb"))
    lst = gemmi.read_structure(HOLO_PDB)
    for ch in lst[0]:
        for i in range(len(ch) - 1, -1, -1):
            if ch[i].name != LIGAND:
                del ch[i]
    lst.remove_empty_chains()
    lst.remove_hydrogens()
    lst.write_pdb(os.path.join(GT, "8a2b_gt_ligand.pdb"))
    _, reread = read_atoms(os.path.join(GT, "8a2b_gt_model.pdb"))
    assert len(reread) == len(atoms), "written GT model does not hold the GT atoms"

    # ---- grid ------------------------------------------------------------------------
    X = np.array([a["xyz"] for a in atoms])
    lo = np.floor((X.min(0) - BOX_PAD) / s) * s
    n = (np.ceil((X.max(0) + BOX_PAD - lo) / s).astype(int) + 1)
    pts = grid_points(lo, n, s)
    print(f"grid {n[0]}x{n[1]}x{n[2]} = {len(pts):,} voxels, origin {lo.round(3)}")

    # ---- GT density ------------------------------------------------------------------
    est = AtomDensityEstimator(device)
    T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=device)
    xyz = T(X)[None]
    occ = T([a["occ"] for a in atoms])[None]
    b = T([a["b"] for a in atoms])[None]
    el = est.get_element_indexes(np.array([a["element"] for a in atoms]))[None].clone()
    grid = T(pts)[None]

    # forward model wrapper == repo original, on a sub-grid small enough for the original
    sub = grid[:, rng.choice(len(pts), 20000, replace=False)]
    diff, peak = check_against_repo(est, xyz, occ, el, b, sub, RMAX)
    print(f"check: chunked forward model vs repo original, max |diff| {diff:.2e} (peak {peak:.2f})")
    assert diff <= 1e-5 * max(peak, 1.0)
    checks["forward_vs_repo_max_abs_diff"] = diff

    with torch.no_grad():
        rho = sample_fc(est, xyz, occ, el, b, grid, RMAX)[0].cpu().numpy()
    print(f"GT density: max {rho.max():.3f}, mean {rho.mean():.4f} e/A^3, "
          f"nonzero {(rho > 0).mean()*100:.1f}% of voxels")

    path = os.path.join(GT, "8a2b_gt_density.ccp4")
    write_ccp4(path, rho, lo, n, s)
    err = check_ccp4(path, rho, pts, rng)
    print(f"check: CCP4 read back + interpolated at voxel positions, max |diff| {err:.2e}")
    assert err < 1e-4
    checks["ccp4_roundtrip_max_abs_diff"] = err

    # ---- pocket ----------------------------------------------------------------------
    L = np.array([a["xyz"] for a in lig])
    pocket = sorted({a["resnum"] for a in prot
                     if np.linalg.norm(L - a["xyz"], axis=1).min() < POCKET_CUTOFF})
    print(f"pocket ({POCKET_CUTOFF} A from {LIGAND}): {len(pocket)} residues {pocket}")

    # ---- 4i20 apo into the 8a2b frame ------------------------------------------------
    def ca_map(path):
        stx = gemmi.read_structure(path)
        stx.remove_alternative_conformations()
        return {r.seqid.num: (r.name, np.array(r["CA"][0].pos.tolist()))
                for r in stx[0][0].get_polymer() if r.find_atom("CA", "*")}
    ca_h, ca_a = ca_map(HOLO_PDB), ca_map(APO_PDB)
    common = [k for k in ca_h if k in ca_a]
    mism = [k for k in common if ca_h[k][0] != ca_a[k][0]]
    assert not mism, f"4i20/8a2b residue types differ at {mism[:5]} -- numbering not shared"
    P = np.array([ca_a[k][1] for k in common])
    Q = np.array([ca_h[k][1] for k in common])
    keep = np.ones(len(common), bool)
    for _ in range(5):   # trimmed fit: superpose on the part that did not move
        R, t = kabsch(P[keep], Q[keep])
        dev = np.linalg.norm(P @ R.T + t - Q, axis=1)
        keep = dev < 2.0
    rms_all = float(np.sqrt((dev ** 2).mean()))
    rms_core = float(np.sqrt((dev[keep] ** 2).mean()))
    apo = gemmi.read_structure(APO_PDB)
    apo.remove_waters()
    apo.remove_hydrogens()
    apo[0].transform_pos_and_adp(gemmi.Transform(gemmi.Mat33(R.tolist()), gemmi.Vec3(*t)))
    apo.cell, apo.spacegroup_hm = gt_st.cell, gt_st.spacegroup_hm
    apo.write_pdb(os.path.join(GT, "4i20_apo_in_8a2b_frame.pdb"))
    pk = [i for i, k in enumerate(common) if k in pocket]
    rms_pocket = float(np.sqrt((dev[pk] ** 2).mean()))
    print(f"4i20 -> 8a2b: {len(common)} common CA, fitted on {keep.sum()} (dev < 2 A): "
          f"core {rms_core:.2f} A, all {rms_all:.2f} A, pocket CA {rms_pocket:.2f} A")

    np.savez(os.path.join(GT, "grid.npz"), lo=lo, n=n, spacing=s, rmax=RMAX)
    json.dump(dict(resolution=res, spacing=s, rmax=RMAX, box_pad=BOX_PAD, grid_origin=lo.tolist(),
                   grid_shape=n.tolist(), n_voxels=int(len(pts)), ligand=LIGAND,
                   n_protein_atoms=len(prot), n_ligand_atoms=len(lig), pocket_residues=pocket,
                   pocket_cutoff=POCKET_CUTOFF,
                   apo_superposition=dict(n_common_ca=len(common), n_fit=int(keep.sum()),
                                          ca_rmsd_core=rms_core, ca_rmsd_all=rms_all,
                                          ca_rmsd_pocket=rms_pocket),
                   checks=checks),
              open(os.path.join(GT, "meta.json"), "w"), indent=2)
    print(f"wrote {sorted(os.listdir(GT))}")


if __name__ == "__main__":
    main()
