"""
    Correspondence between the model's internal atom ordering and a reference structure.

    The diffusion model lays its atoms out in whatever order the Protenix featurizer
    produces for a given construct: entity by entity, residue by residue, and inside each
    residue in CCD order. A reference PDB has its own order, its own chain names, its own
    residue numbering, and is usually missing atoms the model always emits (disordered
    side chains, unmodelled termini). Lining the two up by position would therefore be
    wrong even when the atom counts happen to agree -- ADP-ribose is the clearest case
    here, where the model and the crystal agree on all 36 atom *names* but not on their
    order.

    So the correspondence is built explicitly, keyed on (chain, residue number, atom
    name), validated on residue name, and cached on disk against the reference file and
    the model ordering it was built for.
"""

import hashlib
import json
import os
import pickle
from typing import Dict, List, Optional, Sequence

import gemmi
import numpy as np
import torch


#: Entity kinds a selection can name, and the atom_array annotation backing each.
ENTITY_KIND_ANNOTATIONS = {
    "protein": "is_protein",
    "ligand": "is_ligand",
    "dna": "is_dna",
    "rna": "is_rna",
}

#: gemmi reports a missing altloc as '\x00' rather than an empty string, so an atom in a
#: single-conformer residue is falsy under a plain truthiness test. Anything that filters
#: on altloc has to compare against this explicitly or it silently drops atoms.
NO_ALTLOC = "\x00"

_SELECTION_KEYS = ("model_chains", "entity_kinds", "atom_names", "reference_residue_ranges")


def _as_dict(value):
    """Config values arrive as nested SimpleNamespaces; accept either those or plain dicts."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Expected a mapping or namespace, got {type(value).__name__}")


def normalize_chain_map(chain_map) -> Dict[str, Dict[str, int]]:
    """
        Normalize the configured chain map to {model_chain_id: {"reference_chain", "residue_offset"}}.

        Two spellings are accepted per entry, so the common case stays terse:

            A0: A                                        # same numbering
            C0: {reference_chain: A, residue_offset: 200} # ligand sits at residue 201 of chain A

        `residue_offset` is added to the model residue id to reach the reference residue
        id. It exists because a non-polymer entity is its own chain in the model, numbered
        from 1, while in a crystal it is usually a high-numbered residue inside a protein
        chain.
    """
    normalized = {}
    for model_chain, target in _as_dict(chain_map).items():
        if isinstance(target, str):
            entry = {"reference_chain": target, "residue_offset": 0}
        else:
            target = _as_dict(target)
            unknown = set(target) - {"reference_chain", "residue_offset"}
            if unknown:
                raise ValueError(
                    f"chain_map entry {model_chain!r} has unknown key(s) {sorted(unknown)}; "
                    "expected 'reference_chain' and optionally 'residue_offset'"
                )
            if "reference_chain" not in target:
                raise ValueError(f"chain_map entry {model_chain!r} is missing 'reference_chain'")
            entry = {
                "reference_chain": str(target["reference_chain"]),
                "residue_offset": int(target.get("residue_offset", 0) or 0),
            }
        normalized[str(model_chain)] = entry
    return normalized


def resolve_chain_map(chain_map: Dict[str, Dict[str, int]], model_chains: Sequence[str]):
    """
        Attach each model chain id to its chain-map entry.

        The featurizer labels chains with a letter plus a copy index -- "A0", "B1", "C0" --
        while a config is more readable written against the plain letters. An exact key
        wins; otherwise the chain id with its trailing digits stripped is tried, and only
        when that leaves the match unambiguous among the model's own chains.
    """
    resolved, unresolved = {}, []
    for model_chain in model_chains:
        if model_chain in chain_map:
            resolved[model_chain] = chain_map[model_chain]
            continue

        stem = model_chain.rstrip("0123456789")
        competing = [other for other in model_chains
                     if other != model_chain and other.rstrip("0123456789") == stem]
        if stem in chain_map and not competing:
            resolved[model_chain] = chain_map[stem]
            continue
        unresolved.append(model_chain)
    return resolved, unresolved


def normalize_selection(selection) -> Dict[str, Optional[list]]:
    """Normalize a selection spec, filling every field with None ("everything")."""
    selection = _as_dict(selection)
    unknown = set(selection) - set(_SELECTION_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown selection key(s) {sorted(unknown)}; expected any of {list(_SELECTION_KEYS)}"
        )

    normalized = {key: None for key in _SELECTION_KEYS}
    for key in _SELECTION_KEYS:
        value = selection.get(key)
        if value is None:
            continue
        if key == "reference_residue_ranges":
            ranges = {
                str(chain): [[int(lo), int(hi)] for lo, hi in bounds]
                for chain, bounds in _as_dict(value).items()
            }
            normalized[key] = ranges or None
        else:
            values = [str(item) for item in value]
            if key == "entity_kinds":
                unknown_kinds = set(values) - set(ENTITY_KIND_ANNOTATIONS)
                if unknown_kinds:
                    raise ValueError(
                        f"Unknown entity kind(s) {sorted(unknown_kinds)}; "
                        f"expected any of {sorted(ENTITY_KIND_ANNOTATIONS)}"
                    )
            normalized[key] = values or None
    return normalized


def model_atom_records(atom_array) -> List[dict]:
    """One record per model atom, in the model's own ordering."""
    kinds = {
        kind: np.asarray(getattr(atom_array, annotation)).astype(bool)
        for kind, annotation in ENTITY_KIND_ANNOTATIONS.items()
        if hasattr(atom_array, annotation)
    }
    records = []
    for index in range(len(atom_array)):
        entity_kind = next((kind for kind, flags in kinds.items() if flags[index]), "unknown")
        records.append({
            "model_index": index,
            "model_chain": str(atom_array.chain_id[index]),
            "model_res_id": int(atom_array.res_id[index]),
            "res_name": str(atom_array.res_name[index]),
            "atom_name": str(atom_array.atom_name[index]),
            "element": str(atom_array.element[index]),
            "entity_kind": entity_kind,
        })
    return records


class EntityDerivedAtomArray:
    """
        The model's atom ordering derived from the entity list alone.

        A Protenix atom array is the authority on the model ordering, but getting one means
        running the featurizer, which means loading the checkpoint. For re-scoring a saved
        structure -- which save_structure_full() wrote in exactly this order, one chain per
        entity copy lettered A, B, C, ... -- the entities are enough. Exposes the same
        annotations the reference map reads, so everything downstream is shared.
    """

    def __init__(self, full_sequences, sequence_types):
        from .io import entity_topology, is_non_polymer_sequence_type

        chain_ids, res_ids, res_names, atom_names, elements, kinds = [], [], [], [], [], []
        for entity_index, (sequence, sequence_type) in enumerate(zip(full_sequences, sequence_types)):
            chain_id = chr(ord("A") + entity_index)
            kind = "ligand" if is_non_polymer_sequence_type(sequence_type) else {
                "proteinChain": "protein", "dnaSequence": "dna", "rnaSequence": "rna",
            }[sequence_type]
            for residue_index, (res_name, entity_atom_names, entity_elements) in enumerate(
                entity_topology(sequence, sequence_type)
            ):
                for atom_name, element in zip(entity_atom_names, entity_elements):
                    chain_ids.append(chain_id)
                    res_ids.append(residue_index + 1)
                    res_names.append(res_name)
                    atom_names.append(atom_name)
                    elements.append(element)
                    kinds.append(kind)

        self.chain_id = np.array(chain_ids, dtype=object)
        self.res_id = np.array(res_ids, dtype=int)
        self.res_name = np.array(res_names, dtype=object)
        self.atom_name = np.array(atom_names, dtype=object)
        self.element = np.array(elements, dtype=object)
        kinds = np.array(kinds, dtype=object)
        for kind, annotation in ENTITY_KIND_ANNOTATIONS.items():
            setattr(self, annotation, kinds == kind)

    def __len__(self):
        return len(self.res_id)


def read_structure_in_model_ordering(pdb_file, atom_array):
    """
        Read a structure into the model's atom ordering.

        Matched on (chain, residue number, atom name) against `atom_array`, not by
        position, so a file that omits or reorders atoms is caught rather than silently
        shifting everything after it. Returns [1, N_atom, 3] and a bool mask of which slots
        the file filled.
    """
    structure = gemmi.read_structure(pdb_file)
    structure.remove_hydrogens()
    positions = {}
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                positions[(chain.name, int(residue.seqid.num), str(atom.name))] = (
                    atom.pos.x, atom.pos.y, atom.pos.z
                )

    coordinates = np.zeros((len(atom_array), 3), dtype=np.float32)
    found = np.zeros(len(atom_array), dtype=bool)
    for index in range(len(atom_array)):
        key = (str(atom_array.chain_id[index]), int(atom_array.res_id[index]),
               str(atom_array.atom_name[index]))
        position = positions.get(key)
        if position is not None:
            coordinates[index] = position
            found[index] = True
    return torch.from_numpy(coordinates).unsqueeze(0), torch.from_numpy(found)


def model_ordering_signature(atom_array) -> str:
    """A digest of the model atom ordering, so a cached map cannot be reused across constructs."""
    digest = hashlib.sha256()
    for index in range(len(atom_array)):
        digest.update(
            f"{atom_array.chain_id[index]}|{atom_array.res_id[index]}|"
            f"{atom_array.res_name[index]}|{atom_array.atom_name[index]}\n".encode()
        )
    return digest.hexdigest()


def read_reference_atoms(reference_pdb: str) -> Dict[tuple, dict]:
    """
        Index a reference structure by (chain, residue number, atom name).

        Hydrogens are dropped -- the model emits none. Where a residue carries alternate
        conformations, the highest-occupancy one wins per atom name; occupancy ties keep
        the first conformer encountered, which is the deterministic choice and matches the
        order the file lists them in.
    """
    structure = gemmi.read_structure(reference_pdb)
    structure.remove_hydrogens()

    atoms: Dict[tuple, dict] = {}
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                key = (chain.name, int(residue.seqid.num), str(atom.name))
                occupancy = float(atom.occ)
                existing = atoms.get(key)
                if existing is not None and existing["occupancy"] >= occupancy:
                    continue
                atoms[key] = {
                    "res_name": str(residue.name),
                    "position": (atom.pos.x, atom.pos.y, atom.pos.z),
                    "occupancy": occupancy,
                    "altloc": str(atom.altloc),
                    "element": str(atom.element.name),
                }
    return atoms


class ReferenceAtomMap:
    """
        An explicit, validated correspondence between model atoms and reference atoms.

        Attributes:
            model_indices: LongTensor [M], indices into the model's atom ordering.
            reference_coordinates: FloatTensor [M, 3], the matching reference positions.
            records: list of M dicts describing each mapped pair.
            unmapped: list of dicts for model atoms with no reference counterpart.
            mismatches: list of dicts for pairs whose residue name disagreed.
    """

    def __init__(self, model_indices, reference_coordinates, records, unmapped, mismatches,
                 chain_map, reference_pdb, n_model_atoms):
        self.model_indices = model_indices
        self.reference_coordinates = reference_coordinates
        self.records = records
        self.unmapped = unmapped
        self.mismatches = mismatches
        self.chain_map = chain_map
        self.reference_pdb = reference_pdb
        self.n_model_atoms = n_model_atoms

    def __len__(self):
        return len(self.records)

    def to(self, device):
        self.model_indices = self.model_indices.to(device)
        self.reference_coordinates = self.reference_coordinates.to(device)
        return self

    def selection_mask(self, selection, device=None) -> torch.Tensor:
        """
            Boolean mask over the *mapped* atoms for one selection spec.

            Every field left as None widens rather than narrows, so an empty spec selects
            everything that was mapped.
        """
        selection = normalize_selection(selection)
        mask = torch.ones(len(self.records), dtype=torch.bool, device=device)

        chains = selection["model_chains"]
        kinds = selection["entity_kinds"]
        atom_names = selection["atom_names"]
        residue_ranges = selection["reference_residue_ranges"]

        for i, record in enumerate(self.records):
            keep = True
            if chains is not None and record["model_chain"] not in chains:
                keep = False
            if keep and kinds is not None and record["entity_kind"] not in kinds:
                keep = False
            if keep and atom_names is not None and record["atom_name"] not in atom_names:
                keep = False
            if keep and residue_ranges is not None:
                bounds = residue_ranges.get(record["reference_chain"])
                keep = bounds is not None and any(
                    lo <= record["reference_res_id"] <= hi for lo, hi in bounds
                )
            mask[i] = keep
        return mask

    def entity_kind_mask(self, kind, device=None) -> torch.Tensor:
        """Boolean mask over the mapped atoms for one entity kind, for reporting breakdowns."""
        return torch.tensor(
            [record["entity_kind"] == kind for record in self.records],
            dtype=torch.bool, device=device,
        )

    def summary(self) -> str:
        lines = [
            f"reference: {self.reference_pdb}",
            f"mapped {len(self.records)} of {self.n_model_atoms} model atoms "
            f"({len(self.unmapped)} unmapped)",
        ]
        by_chain: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for record in self.records:
            key = f"{record['model_chain']} -> {record['reference_chain']}"
            by_chain[key] = by_chain.get(key, 0) + 1
            by_kind[record["entity_kind"]] = by_kind.get(record["entity_kind"], 0) + 1
        for key, count in by_chain.items():
            lines.append(f"  chain {key}: {count} atoms")
        for kind, count in sorted(by_kind.items()):
            lines.append(f"  {kind}: {count} atoms")
        if self.unmapped:
            unmapped_residues = sorted({
                (record["model_chain"], record["model_res_id"], record["res_name"])
                for record in self.unmapped
            })
            lines.append(
                f"  unmapped touches {len(unmapped_residues)} residue(s), "
                f"e.g. {unmapped_residues[:5]}"
            )
        return "\n".join(lines)


def build_reference_atom_map(atom_array, reference_pdb, chain_map,
                            require_all_chains_mapped=True) -> ReferenceAtomMap:
    """
        Build the model <-> reference correspondence.

        Args:
            atom_array: the biotite AtomArray the Protenix featurizer produced for this
                construct; it defines the model's atom ordering.
            reference_pdb: path to the reference structure.
            chain_map: model chain id -> reference chain (see normalize_chain_map).
            require_all_chains_mapped: error out when a model chain has no entry in the
                chain map, rather than dropping it from the correspondence silently.

        Raises:
            ValueError: if a mapped pair disagrees on residue name, if a model chain is
                unmapped, or if nothing at all could be mapped.
    """
    chain_map = normalize_chain_map(chain_map)
    reference_atoms = read_reference_atoms(reference_pdb)
    records = model_atom_records(atom_array)

    model_chains = sorted({record["model_chain"] for record in records})
    resolved_chain_map, missing_chains = resolve_chain_map(chain_map, model_chains)
    if missing_chains and require_all_chains_mapped:
        raise ValueError(
            f"chain_map does not cover model chain(s) {missing_chains}. The model's chains "
            f"for this construct are {model_chains}, and the reference has chains "
            f"{sorted({key[0] for key in reference_atoms})}. Add an entry per model chain."
        )

    mapped, unmapped, mismatches = [], [], []
    for record in records:
        entry = resolved_chain_map.get(record["model_chain"])
        if entry is None:
            unmapped.append({**record, "reason": "chain not in chain_map"})
            continue

        reference_chain = entry["reference_chain"]
        reference_res_id = record["model_res_id"] + entry["residue_offset"]
        reference_atom = reference_atoms.get(
            (reference_chain, reference_res_id, record["atom_name"])
        )
        if reference_atom is None:
            unmapped.append({**record, "reason": "no reference atom with this name"})
            continue

        pair = {
            **record,
            "reference_chain": reference_chain,
            "reference_res_id": reference_res_id,
            "reference_res_name": reference_atom["res_name"],
            "reference_position": reference_atom["position"],
            "reference_occupancy": reference_atom["occupancy"],
            "reference_altloc": reference_atom["altloc"],
        }
        if reference_atom["res_name"] != record["res_name"]:
            mismatches.append(pair)
            continue
        mapped.append(pair)

    if mismatches:
        preview = [
            f"{p['model_chain']}/{p['model_res_id']} {p['res_name']}:{p['atom_name']} vs "
            f"{p['reference_chain']}/{p['reference_res_id']} {p['reference_res_name']}"
            for p in mismatches[:10]
        ]
        raise ValueError(
            f"{len(mismatches)} mapped atom pair(s) disagree on residue name, which means the "
            f"chain map or the residue offsets are wrong. First few:\n  " + "\n  ".join(preview)
        )
    if not mapped:
        raise ValueError(
            f"No model atom could be matched to {reference_pdb}. Check the chain map "
            f"({chain_map}) against the reference chains "
            f"{sorted({key[0] for key in reference_atoms})}."
        )

    model_indices = torch.tensor([pair["model_index"] for pair in mapped], dtype=torch.long)
    reference_coordinates = torch.tensor(
        np.array([pair["reference_position"] for pair in mapped], dtype=np.float32),
        dtype=torch.float32,
    )
    return ReferenceAtomMap(
        model_indices=model_indices,
        reference_coordinates=reference_coordinates,
        records=mapped,
        unmapped=unmapped,
        mismatches=mismatches,
        chain_map=resolved_chain_map,
        reference_pdb=os.path.abspath(reference_pdb),
        n_model_atoms=len(records),
    )


def _cache_key(reference_pdb, chain_map, ordering_signature) -> str:
    stat = os.stat(reference_pdb)
    payload = json.dumps(
        {
            "reference_pdb": os.path.abspath(reference_pdb),
            "reference_size": stat.st_size,
            "reference_mtime_ns": stat.st_mtime_ns,
            "chain_map": normalize_chain_map(chain_map),
            "model_ordering": ordering_signature,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def load_or_build_reference_atom_map(atom_array, reference_pdb, chain_map, cache_dir=None,
                                     require_all_chains_mapped=True, verbose=True):
    """
        Build the correspondence, reusing a cached one when it was built for exactly this
        reference file and model atom ordering.

        The cache key covers the reference path, its size and mtime, the chain map, and a
        digest of the model ordering, so editing the reference or changing the construct
        invalidates it rather than silently reusing a stale map.
    """
    if cache_dir is None:
        return build_reference_atom_map(
            atom_array, reference_pdb, chain_map,
            require_all_chains_mapped=require_all_chains_mapped,
        )

    key = _cache_key(reference_pdb, chain_map, model_ordering_signature(atom_array))
    cache_path = os.path.join(cache_dir, f"reference_atom_map_{key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            atom_map = pickle.load(handle)
        if verbose:
            print(f"[rmsd] reusing cached atom map {cache_path}")
        return atom_map

    atom_map = build_reference_atom_map(
        atom_array, reference_pdb, chain_map,
        require_all_chains_mapped=require_all_chains_mapped,
    )
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(atom_map, handle)
    if verbose:
        print(f"[rmsd] cached atom map -> {cache_path}")
    return atom_map


def check_correspondence(atom_map: ReferenceAtomMap) -> List[dict]:
    """
        Re-derive the identity of every mapped pair from the reference file and report any
        disagreement in residue number, residue name or atom name.

        build_reference_atom_map() already refuses to return a map with residue-name
        disagreements; this re-reads the reference independently so the check does not
        depend on the map having been built correctly in the first place.
    """
    reference_atoms = read_reference_atoms(atom_map.reference_pdb)
    problems = []
    for record in atom_map.records:
        key = (record["reference_chain"], record["reference_res_id"], record["atom_name"])
        reference_atom = reference_atoms.get(key)
        if reference_atom is None:
            problems.append({"record": record, "problem": "reference atom no longer present"})
            continue
        expected_res_id = record["model_res_id"] + atom_map.chain_map[record["model_chain"]]["residue_offset"]
        if record["reference_res_id"] != expected_res_id:
            problems.append({"record": record, "problem": "residue number does not follow the chain map offset"})
        if reference_atom["res_name"] != record["res_name"]:
            problems.append({"record": record, "problem": "residue name differs"})
        if record["reference_position"] != reference_atom["position"]:
            problems.append({"record": record, "problem": "reference position differs"})
    return problems
