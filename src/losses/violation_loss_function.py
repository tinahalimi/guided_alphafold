import numpy as np

from .abstract_loss_funciton import AbstractLossFunction
from ..utils.io import chain_segments_from_atom_array
from ..utils.openfold_violations.violations import find_structural_violations, get_atom14_positions


class ViolationLossFunction(AbstractLossFunction):
    def __init__(self, atom_array):
        self.atom_array = atom_array
        # get_atom14_positions() and find_structural_violations() key residues on res_id
        # alone. res_id restarts at 1 in every chain, so for a multimer np.unique(res_id)
        # collapses all chains onto one another and each atom lookup matches once per
        # chain. They also derive residue_index as a plain arange and treat consecutive
        # indices as peptide-bonded, which would bond the last residue of one chain to
        # the first of the next. Evaluating each chain separately avoids both issues and
        # reuses that code unchanged.
        _, chain_segments = chain_segments_from_atom_array(atom_array)

        # This term is protein stereochemistry: peptide-bond geometry, amino-acid
        # rotamers, and the atom14 layout keyed on residue name. A ligand or nucleic
        # chain has none of that -- restype_name_to_atom14_names has no entry for a CCD
        # code -- so those chains are left out rather than crashing the lookup. Their
        # internal geometry is the diffusion model's own business here; the bond-length
        # loss is the term that covers all entity types.
        self._chain_segments = [
            (start, stop) for start, stop in chain_segments
            if self._is_protein_segment(atom_array, start, stop)
        ]
        if not self._chain_segments:
            raise ValueError(
                "ViolationLossFunction was given a construct with no protein chain. This "
                "term only scores protein stereochemistry; set violation_loss_weight to 0."
            )
        self._skipped_chains = len(chain_segments) - len(self._chain_segments)
        if self._skipped_chains:
            print(
                f"[violation loss] scoring {len(self._chain_segments)} protein chain(s); "
                f"skipping {self._skipped_chains} non-protein chain(s)"
            )
        self._last_loss = None

    @staticmethod
    def _is_protein_segment(atom_array, start, stop):
        if hasattr(atom_array, "is_protein"):
            return bool(np.asarray(atom_array.is_protein[start:stop]).all())
        # Older topologies without the annotation: fall back to the residue names the
        # atom14 layout actually knows about.
        from ..utils.openfold_violations import residue_constants
        return all(
            str(res_name) in residue_constants.restype_name_to_atom14_names
            for res_name in atom_array.res_name[start:stop]
        )

    def _chain_violations_loss(self, atom_array, atom_coords):
        atom_coords_14 = get_atom14_positions(atom_array, atom_coords)
        violations_dict = find_structural_violations(atom_coords_14, atom_array)
        return (
            violations_dict["between_residues"]["clashes_mean_loss"]
            + violations_dict["between_residues"]["connections_per_residue_loss_sum"].mean()
            + violations_dict["within_residues"]["per_atom_loss_sum"].mean()
        )

    def get_violations_loss(self, x_0_hat):
        # Mean over chains, so the configured violation_loss_weight keeps the same
        # meaning regardless of oligomeric state. For a single chain this is exactly the
        # previous value. Note that inter-chain steric clashes are not covered by this
        # term, since each chain is evaluated in isolation.
        losses = [
            self._chain_violations_loss(self.atom_array[start:stop], x_0_hat[:, start:stop])
            for start, stop in self._chain_segments
        ]
        return sum(losses) / len(losses)

    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        loss = self.get_violations_loss(x_0_hat)
        self._last_loss = loss.item()
        return loss, None

    def wandb_log(self, x_0_hat):
        return {"violation loss": self._last_loss}
