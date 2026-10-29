#!/usr/bin/env python
"""
Write blurred (dB = 150 A^2) density maps as CCP4, on the GT grid (same box/frame as every other map here).

Blur exactly as in the guidance: B -> B + 150 for every atom, kernel cutoff
rmax = 3.5 * sqrt((Bmed + 42 + 150) / (Bmed + 42)) = 6.76 A. Model maps use the atoms with a crystal
counterpart (occupancy 1 in the written PDBs), occupancy 1 and the crystal B, as in the runs.

  gt/8a2b_gt_density_dB150.ccp4
  outputs_whole_blur_ens/final_blur150/<cond>_sample<i>_dB150.ccp4  and  <cond>_avgmap_dB150.ccp4
Check: the same code with dB = 0 reproduces gt/8a2b_gt_density.ccp4.
"""
import os
import sys

import gemmi
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from forward_model import AtomDensityEstimator, sample_fc  # noqa: E402
from prepare_gt import read_atoms, grid_points, write_ccp4, LIGAND  # noqa: E402

DB = 150.0
CB_WIDEST = 42.0
CHUNK = 40_000
ENS = os.path.join(HERE, "outputs_whole_blur_ens")
CONDS = ["whole_dB150_ens_wL1.0", "whole_dB150_wL1.0"]
dev = "cuda:0"


def main():
    g = np.load(os.path.join(HERE, "gt", "grid.npz"))
    lo, n, s, RMAX = g["lo"], g["n"], float(g["spacing"]), float(g["rmax"])
    pts = torch.tensor(grid_points(lo, n, s), dtype=torch.float32, device=dev)
    est = AtomDensityEstimator(dev)
    _, gt = read_atoms(os.path.join(HERE, "gt", "8a2b_gt_model.pdb"))
    Bmed = float(np.median([a["b"] for a in gt]))
    rmax_b = RMAX * float(np.sqrt((Bmed + CB_WIDEST + DB) / (Bmed + CB_WIDEST)))
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)

    def dens(xyz, occ, el, b, db, rmax):
        with torch.no_grad():
            return torch.cat([sample_fc(est, xyz[None], occ[None], el[None], (b + db)[None],
                                        pts[None, c:c + CHUNK], rmax)[0]
                              for c in range(0, len(pts), CHUNK)]).cpu().numpy()

    G = (T([a["xyz"] for a in gt]), T([a["occ"] for a in gt]),
         est.get_element_indexes(np.array([a["element"] for a in gt])), T([a["b"] for a in gt]))
    ref = np.array(gemmi.read_ccp4_map(os.path.join(HERE, "gt", "8a2b_gt_density.ccp4")).grid, copy=False).ravel()
    d0 = dens(*G, 0.0, RMAX)
    print(f"check dB=0 vs gt map: max |diff| {np.abs(d0 - ref).max():.1e} (map max {ref.max():.2f})")
    assert np.abs(d0 - ref).max() < 1e-3
    p = os.path.join(HERE, "gt", "8a2b_gt_density_dB150.ccp4")
    write_ccp4(p, dens(*G, DB, rmax_b), lo, n, s)
    print("wrote", p, f"(dB {DB:.0f}, cutoff {rmax_b:.2f} A)")

    out = os.path.join(ENS, "final_blur150")
    os.makedirs(out, exist_ok=True)
    for cond in CONDS:
        maps = []
        for i in range(5):
            st = gemmi.read_structure(os.path.join(ENS, "final", f"{cond}_sample{i}.pdb"))
            at = [a for ch in st[0] for r in ch for a in r if a.occ > 0 and a.element.name != "H"]
            m = dens(T([a.pos.tolist() for a in at]), T([1.0] * len(at)),
                     est.get_element_indexes(np.array([a.element.name for a in at])),
                     T([a.b_iso for a in at]), DB, rmax_b)
            maps.append(m)
            write_ccp4(os.path.join(out, f"{cond}_sample{i}_dB150.ccp4"), m, lo, n, s)
        write_ccp4(os.path.join(out, f"{cond}_avgmap_dB150.ccp4"), np.mean(maps, 0), lo, n, s)
        print(f"wrote {cond}: 5 samples + averaged map -> {out}")


if __name__ == "__main__":
    main()
