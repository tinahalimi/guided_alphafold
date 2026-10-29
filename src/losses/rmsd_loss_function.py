"""
    RMSD guidance: an oracle control for the guidance loop.

    Every other loss in this package scores a sample against an experimental measurement.
    This one scores it against a known reference structure instead, so there is nothing to
    model and nothing to fit -- the objective is just the coordinate deviation from the
    answer. That makes it useful for two things: checking that the guidance machinery
    (loss -> gradient on the noisy coordinates -> denoising step) actually moves samples
    where it is told to, and measuring the best RMSD the sampler can reach when the
    likelihood is perfect. It is not a structure determination method.
"""

import csv
import json
import os

import torch

from .abstract_loss_funciton import AbstractLossFunction
from ..protenix.metrics.rmsd import align_pred_to_true, rmsd
from ..utils.rmsd_reference import (
    ENTITY_KIND_ANNOTATIONS,
    load_or_build_reference_atom_map,
)


class RMSDGuidanceLossFunction(AbstractLossFunction):
    """
        Deviation of the generated structure from a reference structure.

        The correspondence between the model's atom ordering and the reference is an
        explicit, cached index map (see src/utils/rmsd_reference.py) rather than a
        positional zip, so atoms the reference does not resolve are simply left out and
        the ligand -- whose atom order differs between the CCD and the crystal file -- is
        matched by name.

        Alignment and scoring are separate atom sets. Both default to every mapped atom,
        which is the plain "superpose the whole thing and measure it" objective, but they
        can be narrowed independently: superposing on the protein while scoring the ligand
        measures the ligand's pose in its pocket, whereas superposing on the ligand alone
        measures its internal conformation.
    """

    def __init__(
        self,
        atom_array,
        reference_pdb,
        chain_map,
        device,
        alignment_selection=None,
        scoring_selection=None,
        objective="rmsd",
        cache_dir=None,
        allowing_reflection=False,
        require_all_chains_mapped=True,
        track_trajectory=True,
        trajectory_every_n_steps=1,
        verbose=True,
    ):
        if objective not in ("rmsd", "msd"):
            raise ValueError(f"objective must be 'rmsd' or 'msd', got {objective!r}")

        self._device = device
        self._objective = objective
        self._allowing_reflection = allowing_reflection
        self._track_trajectory = bool(track_trajectory)
        self._trajectory_every_n_steps = max(1, int(trajectory_every_n_steps))

        self.atom_map = load_or_build_reference_atom_map(
            atom_array,
            reference_pdb,
            chain_map,
            cache_dir=cache_dir,
            require_all_chains_mapped=require_all_chains_mapped,
            verbose=verbose,
        ).to(device)

        self.alignment_mask = self.atom_map.selection_mask(alignment_selection, device=device)
        self.scoring_mask = self.atom_map.selection_mask(scoring_selection, device=device)
        if not bool(self.alignment_mask.any()):
            raise ValueError("The alignment selection matches none of the mapped atoms.")
        if not bool(self.scoring_mask.any()):
            raise ValueError("The scoring selection matches none of the mapped atoms.")

        # Per-entity-kind masks, so the reported number can be broken down without
        # re-deriving the correspondence.
        self.entity_kind_masks = {
            kind: self.atom_map.entity_kind_mask(kind, device=device)
            for kind in ENTITY_KIND_ANNOTATIONS
        }
        self.entity_kind_masks = {
            kind: mask for kind, mask in self.entity_kind_masks.items() if bool(mask.any())
        }

        self._last_report = {}
        # One entry per evaluated denoising step. The breakdown is computed anyway to form
        # the loss, so keeping it costs a handful of floats per step and turns the run into
        # a trajectory rather than a single end-point number.
        self.trajectory = []

        if verbose:
            print("[rmsd] " + self.atom_map.summary().replace("\n", "\n[rmsd] "))
            print(
                f"[rmsd] aligning on {int(self.alignment_mask.sum())} atoms, "
                f"scoring {int(self.scoring_mask.sum())} atoms, objective={objective}"
            )
            for kind, mask in sorted(self.entity_kind_masks.items()):
                print(f"[rmsd]   scored {kind} atoms: {int((mask & self.scoring_mask).sum())}")

    # ------------------------------------------------------------------ scoring

    def fit_transform(self, x_0_hat):
        """
            The rigid transform that takes the sample into the reference frame.

            Fitted on the alignment set only, but returned so it can be applied to every
            atom -- which is what makes a written-out structure comparable with the
            reference in a viewer. Differentiable in the input, so gradients reach mapped
            atoms outside the alignment set through the fit itself.
        """
        predicted = x_0_hat[:, self.atom_map.model_indices]
        batch_size = predicted.shape[0]
        reference = self.atom_map.reference_coordinates.unsqueeze(0).expand(batch_size, -1, -1)

        alignment_mask = self.alignment_mask.to(predicted.dtype).unsqueeze(0).expand(batch_size, -1)
        _, rotation, translation = align_pred_to_true(
            predicted,
            reference,
            atom_mask=alignment_mask,
            allowing_reflection=self._allowing_reflection,
        )
        return rotation, translation, predicted, reference

    def apply_transform(self, coordinates, rotation, translation):
        """Move coordinates into the reference frame. Shapes [B, N, 3] and [B, 3, 3] / [B, 1, 3]."""
        return torch.matmul(coordinates, rotation.transpose(-1, -2)) + translation

    def superpose(self, x_0_hat):
        """
            Superpose the mapped model atoms onto the reference over the alignment set.

            Returns the transformed mapped coordinates [B, M, 3] and the reference
            coordinates broadcast to the same shape.
        """
        rotation, translation, predicted, reference = self.fit_transform(x_0_hat)
        return self.apply_transform(predicted, rotation, translation), reference

    def superpose_all_atoms(self, x_0_hat):
        """
            Every model atom moved into the reference frame by the alignment-set fit.

            Unmapped atoms come along for the ride, exactly as they should: the transform is
            rigid, so a structure written this way overlays the reference in a viewer with
            nothing re-fitted, and the parts the reference does not resolve stay where the
            model put them relative to the rest.
        """
        rotation, translation, _, _ = self.fit_transform(x_0_hat)
        return self.apply_transform(x_0_hat, rotation, translation)

    def rmsd_breakdown(self, x_0_hat):
        """
            Per-sample RMSD over the scoring set, plus a breakdown per entity kind.

            All figures use the one superposition fitted on the alignment set, so the
            protein and ligand numbers are directly comparable -- the ligand figure is its
            pose in the frame the protein fixed, not a separate ligand-only fit.
        """
        transformed, reference = self.superpose(x_0_hat)

        breakdown = {
            "total": rmsd(transformed, reference, mask=self.scoring_mask.to(transformed.dtype), reduce=False)
        }
        for kind, kind_mask in self.entity_kind_masks.items():
            mask = kind_mask & self.scoring_mask
            if bool(mask.any()):
                breakdown[kind] = rmsd(
                    transformed, reference, mask=mask.to(transformed.dtype), reduce=False
                )
        return breakdown

    def get_rmsd_loss(self, x_0_hat):
        breakdown = self.rmsd_breakdown(x_0_hat)
        per_sample = breakdown["total"]
        loss = per_sample.mean() if self._objective == "rmsd" else per_sample.pow(2).mean()

        self._last_report = {
            key: [round(float(value), 4) for value in values.detach().cpu().tolist()]
            for key, values in breakdown.items()
        }
        self._last_report["loss"] = float(loss.detach().cpu())
        return loss

    # -------------------------------------------------------- loss-plugin interface

    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        loss = self.get_rmsd_loss(x_0_hat)
        self._record_trajectory(time=time, diffusion_index=i, step=step)
        return loss, None

    def _record_trajectory(self, time=None, diffusion_index=None, step=None):
        """
            Append the step just scored to the trajectory.

            Reads the values get_rmsd_loss() already cached, so nothing is recomputed. The
            index the diffusion loop passes ascends as the noise level falls, so the
            trajectory reads left to right as denoising progresses.
        """
        if not self._track_trajectory:
            return
        position = step if step is not None else len(self.trajectory)
        if position % self._trajectory_every_n_steps:
            return
        entry = {
            "step": step,
            "diffusion_index": diffusion_index,
            "time": round(float(time), 6) if time is not None else None,
            "loss": self._last_report["loss"],
        }
        for key, values in self._last_report.items():
            if key != "loss":
                entry[key] = list(values)
        self.trajectory.append(entry)

    def write_trajectory(self, folder_path, file_name="rmsd_trajectory.csv"):
        """
            Write the trajectory as long-form CSV: one row per step per sample per selection.

            Long form so it can be plotted or grouped without reshaping. Returns None when
            nothing was recorded -- an unguided run never calls the loss inside the loop, so
            it has no trajectory unless it was run with the guidance strength set to zero.
        """
        if not self.trajectory:
            return None
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, file_name)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "diffusion_index", "time", "selection", "sample",
                             "rmsd_angstrom"])
            for entry in self.trajectory:
                for key, values in entry.items():
                    if key in ("step", "diffusion_index", "time", "loss"):
                        continue
                    for sample, value in enumerate(values):
                        writer.writerow([entry["step"], entry["diffusion_index"],
                                         entry["time"], key, sample, value])
        return path

    def wandb_log(self, x_0_hat):
        # Values are cached from the last __call__ rather than recomputed: MultiLossFunction
        # calls wandb_log several times per step and a superposition is not free.
        if not self._last_report:
            return {}
        log = {"rmsd loss": self._last_report["loss"]}
        for key, values in self._last_report.items():
            if key == "loss":
                continue
            log[f"rmsd {key} (A)"] = sum(values) / len(values)
            log[f"rmsd {key} best (A)"] = min(values)
        return log

    def save_state(self, structures, folder_path, **kwargs):
        """Record the final per-sample RMSD next to the structures it describes."""
        if structures is None:
            return
        with torch.no_grad():
            breakdown = self.rmsd_breakdown(structures.detach().to(self._device))
        report = {
            "reference_pdb": self.atom_map.reference_pdb,
            "chain_map": self.atom_map.chain_map,
            "mapped_atoms": len(self.atom_map),
            "aligned_atoms": int(self.alignment_mask.sum()),
            "scored_atoms": int(self.scoring_mask.sum()),
            "scored_atoms_by_entity_kind": {
                kind: int((mask & self.scoring_mask).sum())
                for kind, mask in self.entity_kind_masks.items()
            },
            "rmsd_per_sample": {
                key: [round(float(value), 4) for value in values.cpu().tolist()]
                for key, values in breakdown.items()
            },
        }
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "rmsd_report.json"), "w") as handle:
            json.dump(report, handle, indent=2)

        trajectory_path = self.write_trajectory(folder_path)
        if trajectory_path is not None:
            report["trajectory_csv"] = trajectory_path
            report["trajectory_steps"] = len(self.trajectory)
        return report
