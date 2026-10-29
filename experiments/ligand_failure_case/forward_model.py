"""
The guided-AlphaFold forward model (real-space Fc from atoms), imported from the repo.

The density math is the repo's own: `AtomDensityEstimator.compute_density_from_distances`
with the repo's Cromer-Mann tables (src/utils/density_guidance/density_atomsf.py).

Only the neighbour search is replaced. The repo's `sample_fc_from_all_atom` builds the full
[batch, n_grid, n_atoms] distance tensor before chunking. Here that is 5 x ~1e6 x 2373, about
1e10 floats, so it would not fit on any GPU. `sample_fc` finds the same (grid point, atom)
pairs in chunks and then runs the repo code unchanged. The result is identical;
`check_against_repo` verifies this on a grid small enough for the original to run.
"""
import os
import sys

import torch


def _find_repo():
    """The guided_alphafold checkout: $GUIDED_AF_REPO, else this folder inside the repo
    (<repo>/experiments/ligand_failure_case), else a sibling checkout (../guided_alphafold/guided_alphafold)."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("GUIDED_AF_REPO", ""), os.path.join(here, "..", ".."),
             os.path.join(here, "..", "guided_alphafold", "guided_alphafold")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "experiment_manager.py")):
            return os.path.abspath(c)
    raise RuntimeError("guided_alphafold repo not found; set GUIDED_AF_REPO=/path/to/guided_alphafold")


REPO = _find_repo()
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.utils.density_guidance.density_estimator import AtomDensityEstimator  # noqa: E402,F401


def sample_fc(est, atoms, occ, el, b, xyz, rmax, elem_budget=50_000_000):
    """Same output as est.sample_fc_from_all_atom(atoms, occ, el, b, xyz, rmax).

    atoms [B,N,3], occ/el/b [B,N], xyz [B,G,3] -> density [B,G] in e/A^3.
    """
    nb, na, ng = atoms.shape[0], atoms.shape[1], xyz.shape[1]
    chunk = max(1, elem_budget // max(1, nb * na))
    with torch.no_grad():
        pairs = []
        for s in range(0, ng, chunk):
            d = (atoms[:, None] - xyz[:, s:s + chunk, None]).norm(dim=-1)
            nz = (d <= rmax).nonzero()
            nz[:, 1] += s
            pairs.append(nz)
        pb, px, pa = torch.cat(pairs).unbind(-1)
    out = torch.zeros(nb, ng, dtype=torch.float32, device=atoms.device)
    if len(pb) == 0:
        return out
    d2 = (xyz[pb, px] - atoms[pb, pa]).pow(2).sum(-1)
    dens = est.compute_density_from_distances(d2, b[pb, pa], el[pb, pa]) * occ[pb, pa]
    out.index_put_((pb, px), dens, accumulate=True)
    return out


def check_against_repo(est, atoms, occ, el, b, xyz, rmax):
    """Max |difference| between sample_fc and the repo's un-chunked original."""
    ref = est.sample_fc_from_all_atom(atoms, occ, el, b, xyz, rmax=rmax)
    new = sample_fc(est, atoms, occ, el, b, xyz, rmax, elem_budget=1_000_000)
    return float((ref - new).abs().max()), float(ref.abs().max())
