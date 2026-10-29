#!/usr/bin/env python
"""
Real (experimental) density maps for 8a2b (holo + KY9) and 4i20 (apo) from the deposited MTZs.

Map = 2mFo-DFc, i.e. the FWT/PHWT columns, made exactly as the repo's X-ray pipeline does it
(src/utils/process_pipeline_inputs/ed_targets_density.py: get_2fofc_maps): phenix.f000 for F(0,0,0),
then CCP4 fft with LABIN F1=FWT PHI=PHWT SIG1=SIGFP and VF000 <cell volume> <F000>, so the map is on
an absolute scale (e/A^3) like the forward model. No blurring, no sharpening, no truncation.

  real_maps/<pdb>_2fofc_cell.ccp4         full map in the crystal frame (open with data/<pdb>_final.pdb)
  real_maps/8a2b_2fofc_box.ccp4           8a2b map resampled (tricubic) on the GT box grid (gt/grid.npz),
                                          same frame/grid as every other map here
  real_maps/4i20_2fofc_box_8a2b_frame.ccp4  4i20 map moved with the same superposition as
                                          gt/4i20_apo_in_8a2b_frame.pdb, on the GT box grid
  real_maps/meta.json                     F000, map stats, checks

Checks: (1) CCP4 fft vs gemmi FFT (+F000/V) of the same columns; (2) box map read back vs the
interpolated values; (3) apo transform reproduces gt/4i20_apo_in_8a2b_frame.pdb.

Run on the head node (CPU, needs the ccp4 + phenix modules):
    python prepare_real_maps.py
"""
import json
import os
import re
import subprocess
import sys

import gemmi
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_gt import grid_points, write_ccp4, kabsch  # noqa: E402

OUT = os.path.join(HERE, "real_maps")
DATA = os.path.join(HERE, "data")
GT = os.path.join(HERE, "gt")
MODULES = "module load ccp4/7.0 phenix/2.0-5936 >/dev/null 2>&1"
SAMPLE = 6          # fft grid spacing ~ d_min / 6 (fine grid for the tricubic resampling)


def sh(cmd, cwd):
    r = subprocess.run(["bash", "-lc", f"{MODULES} && {cmd}"], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def cell_map(pdb):
    pdb_path, mtz_path = os.path.join(DATA, f"{pdb}_final.pdb"), os.path.join(DATA, f"{pdb}_final.mtz")
    out = os.path.join(OUT, f"{pdb}_2fofc_cell.ccp4")
    f000 = float(re.search(r"F\(0,0,0\)=([\d.]+)", sh(f"phenix.f000 {pdb_path}", OUT)).group(1))
    mtz = gemmi.read_mtz_file(mtz_path)
    vol = mtz.get_cell().volume
    sh(f"fft hklin {mtz_path} mapout {out} << EOF\nLABIN F1=FWT PHI=PHWT SIG1=SIGFP\n"
       f"VF000 {vol} {f000}\nGRID SAMPLE {SAMPLE}\nEOF", OUT)
    m = gemmi.read_ccp4_map(out)
    m.setup(float("nan"))
    assert not np.isnan(np.array(m.grid)).any()
    # Reflections with no measured FP (PDB-REDO fills their FWT with DFc, i.e. pure model) are left out
    # by fft because SIG1=SIGFP is missing there -> the map holds measured data only.
    # Independent FFT of the same reflections with gemmi (gemmi omits F000 -> add F000 / V).
    miss = np.isnan(mtz.column_with_label("FP").array)
    d_obs = float(mtz.make_d_array()[~miss].min())
    data = mtz.array.copy()
    data[miss, mtz.column_labels().index("FWT")] = 0
    mtz.set_data(data)
    g2 = mtz.transform_f_phi_to_map("FWT", "PHWT", exact_size=list(m.grid.shape))
    a1, a2 = np.array(m.grid), np.array(g2) + f000 / vol
    check = float(np.abs(a1 - a2).max())
    print(f"{pdb}: d_min {mtz.resolution_high():.2f} A (measured FP to {d_obs:.2f} A; {miss.sum()} of {len(miss)} "
          f"reflections unmeasured, left out), F000 {f000:.0f}, V {vol:.0f} A^3, grid {m.grid.shape}, "
          f"cell mean {a1.mean():.3f} sd {a1.std():.3f} e/A^3; CCP4 fft vs gemmi FFT max |diff| {check:.1e}")
    return m.grid, dict(f000=f000, volume=vol, d_min=mtz.resolution_high(), d_min_measured=d_obs,
                        n_reflections=int(len(miss)), n_unmeasured_left_out=int(miss.sum()), grid=list(m.grid.shape),
                        cell_mean=float(a1.mean()), cell_sd=float(a1.std()), fft_vs_gemmi_max_abs_diff=check)


def resample(grid, lo, n, s, R=np.eye(3), t=np.zeros(3)):
    """Values of the crystal map at the GT box voxels p, read at R @ p + t (crystal frame), tricubic."""
    arr = np.zeros(tuple(int(x) for x in n), dtype=np.float32)
    M = R * s
    tr = gemmi.Transform(gemmi.Mat33(M.tolist()), gemmi.Vec3(*(R @ lo + t)))
    grid.interpolate_values(arr, tr, order=3)
    return arr.ravel(), tr


def main():
    os.makedirs(OUT, exist_ok=True)
    g = np.load(os.path.join(GT, "grid.npz"))
    lo, n, s = g["lo"], g["n"], float(g["spacing"])
    pts = grid_points(lo, n, s)
    rng = np.random.default_rng(0)
    meta = {}

    # ---- holo: box is already in the 8a2b crystal frame
    gh, meta["8a2b"] = cell_map("8a2b")
    vh, _ = resample(gh, lo, n, s)

    # ---- apo: the transform that put 4i20 into the 8a2b frame, inverted (box point -> 4i20 frame)
    ga, meta["4i20"] = cell_map("4i20")
    P = np.array([a.pos.tolist() for c in gemmi.read_structure(os.path.join(DATA, "4i20_final.pdb"))[0]
                  for r in c if r.name != "HOH" for a in r if a.element.name not in ("H", "D")])
    Q = np.array([a.pos.tolist() for c in gemmi.read_structure(os.path.join(GT, "4i20_apo_in_8a2b_frame.pdb"))[0]
                  for r in c for a in r])
    assert len(P) == len(Q), (len(P), len(Q))
    R, t = kabsch(P, Q)                                  # apo frame -> 8a2b frame: P R^T + t = Q
    sup_dev = float(np.abs(P @ R.T + t - Q).max())
    va, _ = resample(ga, lo, n, s, R.T, -R.T @ t)        # 8a2b frame -> apo frame
    meta["4i20"]["superposition_reproduces_gt_pdb_max_dev_A"] = sup_dev

    for pdb, v, gr, Rm, tm, name in (("8a2b", vh, gh, np.eye(3), np.zeros(3), "8a2b_2fofc_box.ccp4"),
                                     ("4i20", va, ga, R.T, -R.T @ t, "4i20_2fofc_box_8a2b_frame.ccp4")):
        path = os.path.join(OUT, name)
        write_ccp4(path, v, lo, n, s)
        # (a) the resampling: direct tricubic at random voxels; (b) the file: read back at voxel positions
        idx = rng.choice(len(pts), 2000, replace=False)
        direct = np.array([gr.tricubic_interpolation(gemmi.Position(*(Rm @ pts[i] + tm))) for i in idx])
        mb = gemmi.read_ccp4_map(path); mb.setup(float("nan"))
        back = np.array([mb.grid.interpolate_value(gemmi.Position(*map(float, pts[i]))) for i in idx])
        meta[pdb].update(box_file=os.path.relpath(path, HERE), box_mean=float(v.mean()), box_sd=float(v.std()),
                         box_max=float(v.max()), resample_vs_direct_max_abs_diff=float(np.abs(direct - v[idx]).max()),
                         file_roundtrip_max_abs_diff=float(np.abs(back - v[idx]).max()))
        print(f"wrote {path}: mean {v.mean():.3f} sd {v.std():.3f} max {v.max():.2f}; resample vs direct "
              f"{meta[pdb]['resample_vs_direct_max_abs_diff']:.1e}, file round trip "
              f"{meta[pdb]['file_roundtrip_max_abs_diff']:.1e}")
    print(f"4i20 superposition reproduces gt/4i20_apo_in_8a2b_frame.pdb to {sup_dev:.1e} A")

    # ---- how the real holo map compares to the synthetic GT map (forward model of the deposited model)
    syn = np.array(gemmi.read_ccp4_map(os.path.join(GT, "8a2b_gt_density.ccp4")).grid).ravel()
    nz = syn > 0.05 * syn.max()
    A = np.stack([syn[nz], np.ones(nz.sum())], 1)
    k, b = np.linalg.lstsq(A, vh[nz], rcond=None)[0]
    meta["8a2b_vs_synthetic_gt"] = dict(cc_where_synthetic_gt_above_5pct=float(np.corrcoef(syn[nz], vh[nz])[0, 1]),
                                        fit_real_eq_k_syn_plus_b=[float(k), float(b)])
    print(f"8a2b real vs synthetic GT map (voxels where GT > 5% of max): CC {meta['8a2b_vs_synthetic_gt']['cc_where_synthetic_gt_above_5pct']:.3f}, "
          f"real ~ {k:.3f} * synthetic + {b:.3f}")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
