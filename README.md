# Guided Protein Structure Prediction

## Overview

This codebase implements a guided protein structure prediction pipeline that incorporates experimental data from three different structural biology modalities to improve AlphaFold3's prediction accuracy. The system uses a diffusion-based approach guided by experimental log-likelihoods to generate protein structures that are consistent with:

- **Cryo-EM**: Electrostatic potential maps from electron microscopy
- **X-ray Crystallography**: Real-space electron density maps (2mFo-DFc or END maps) from crystallographic data  
- **NMR Spectroscopy**: Distance, order parameters, and dihedrals restraints (NOE, dihedral angles, RDC, order parameters)

The pipeline processes experimental data, runs experiment-guided structure prediction, performs structural relaxation using AMBER99 force field, and evaluates results using modality-specific metrics.

A fourth modality, **RMSD**, is an oracle control rather than an experiment: it guides on the
coordinate deviation from a known reference structure. It exists to validate the guidance
loop and to establish the best-case RMSD the sampler can reach when the likelihood is
perfect. See [RMSD guidance](#4-rmsd-guided-structure-prediction-oracle-control).

## Installation

### Environment Setup

1. **Setup the environment:**

   Create a fresh conda environment with Python 3.11:
   ```bash
   conda create -n guided_af3 python=3.11
   conda activate guided_af3
   ```
   Install the core scientific stack:
   ```bash
   pip3 install numpy==1.26.4 scipy==1.15.0 pandas==2.2.0 matplotlib==3.9.0 scikit-learn==1.2.0 scikit-learn-extra==0.3.0 skan==0.13.0 scikit-image==0.24.0 imageio==2.37.0 cvxpy==1.6.6 cvxpylayers==0.1.9
   ```
   Install bioinformatics and structure libraries:
   ```bash
   pip3 install biopython==1.83 biotite==1.0.1 gemmi==0.6.5 rdkit==2023.09.6 dm-tree==0.1.8 py3dmol==2.4.2 modelcif==0.7 loco-hd==0.1.4 pynmrstar==3.3.5 ml-collections==0.1.1
   ```
   Install utilities and logging:
   ```bash
   pip3 install tqdm pyyaml ipywidgets wandb==0.19.4 ipdb==0.13.13 icecream==2.1.4 hydride==1.2.3 pydantic==2.10.6 pdbeccdutils==0.8.5
   ```
   Install PyTorch with CUDA 12.1:
   ```bash
   pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
   ```
   > The pipeline has also been verified end-to-end on `torch==2.6.0+cu124`
   > (NVIDIA H100, Python 3.11). If you use a newer torch, note that a `deepspeed`
   > &le; 0.5.9 left over in the environment will emit a warning about `torch._six`;
   > see [Troubleshooting](#troubleshooting).
   Install AlphaFold-related JAX / TF packages (CPU-only here):
   ```bash
   pip3 install absl-py==1.0.0 dm-haiku==0.0.12 docker==5.0.0 jax==0.4.26 jaxlib==0.4.26 tensorflow-cpu==2.16.1 "pytest<8.5.0" "setuptools<72.0.0"
   ```
   Install Keops
   ```bash
   pip3 install pykeops==2.3 geomloss==0.2.6
   python3
   >>> import pykeops; pykeops.test_torch_bindings() # test keops install
   ```
   Install OpenMM and PDBFixer. Both are **required**: the NMR and X-ray
   preprocessing import them directly (`preprocess_nmr_inputs.py`, `fix_pdb.py`), and
   AlphaFold2's AMBER relaxation needs them too.

   Use `--no-deps`. OpenMM's dependency metadata requests numpy 2.x, which overwrites
   the numpy 1.26.4 pinned above and breaks pandas, tensorflow-cpu, and numba.
   OpenMM 8.2 works correctly against numpy 1.26.4.
   ```bash
   pip3 install --no-deps openmm==8.2.0 pdbfixer==1.12.0
   ```
   These are the versions AlphaFold2's own `requirements.txt` pins. Verify:
   ```bash
   python3 -c "import openmm, pdbfixer; print('openmm', openmm.__version__, 'OK')"
   ```
   PDBFixer manual: https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html

3. **Download Protenix model weights and data:**
   
   This pipeline is built on top of [Protenix](https://github.com/bytedance/Protenix), a PyTorch reproduction of DeepMind's AlphaFold3. Download the required pre-trained model weights and chemical component data files:
   
   ```bash
   # Download model weights (v0.2.0)
   wget -P src/af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt
   
   # Download chemical component dictionary files
   wget -P src/af3-dev/release_data/ https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif
   wget -P src/af3-dev/release_data/ https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl
   ```
   
   For more information, visit the Protenix repository: https://github.com/bytedance/Protenix

### External Dependencies

3. **END RAPID (for X-ray absolute scale maps):**
   
   Download and install the END RAPID script for rendering absolute scale electron density maps (CCP4 8.0 and Phenix 1.21.2):
   ```bash
   wget https://bl831.als.lbl.gov/END/RAPID/end.rapid/Distributions/end.rapid.tar.gz
   tar -xzf end.rapid.tar.gz
   ```
   Setup environment path:
   ```bash
   export PATH=<directory_path>:$PATH
   ```
   Move the script to root:
   ```bash
   cp end.rapid/END_RAPID.com .
   chmod +x END_RAPID.com
   ```
   Installation manual: https://bl831.als.lbl.gov/END/RAPID/end.rapid/Documentation/end.rapid.Manual.htm#InstallationInstructions

4. **Phenix 1.21.2 (for X-ray and Cryo-EM):**
   
   Required for structure refinement and validation metrics.
   
   Download from: http://www.phenix-online.org/

5. **CCP4 8.0 (for X-ray):**
   
   Required for crystallographic computations and map processing.
   
   Download from: http://www.ccp4.ac.uk/

6. **AMBER99 relaxation using AlphaFold2 (required):**

   This is **required, not optional**. `experiment_manager.py` imports
   `src.utils.relaxation` at module load, which does `from alphafold.relax import relax`,
   so *every* entrypoint — cryo-EM, X-ray, and NMR — fails at import without it.

   Two upstream details make the obvious install commands fail:

   - AlphaFold2 no longer ships a `setup.py`, so `python3 setup.py install` errors out.
   - Its `pyproject.toml` declares `py-modules = ["run_alphafold"]` and never declares
     `packages`, so `pip install .` produces a ~24 KB wheel containing only the
     `run_alphafold` CLI. It installs "successfully" and `import alphafold` still fails.

   Put the repository on the environment's import path instead:

   ```bash
   # Clone into the conda env, so it is removed together with the env.
   # Pinned to the revision this pipeline was verified against: AlphaFold2 changed its
   # packaging once already (it dropped setup.py), so do not track a floating main.
   git clone https://github.com/google-deepmind/alphafold.git "$CONDA_PREFIX/opt/alphafold"
   git -C "$CONDA_PREFIX/opt/alphafold" checkout c77e5d2a8961d1a353632c462914ff0a32a950f6

   # Make `import alphafold` work env-wide, from any working directory
   SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
   echo "$CONDA_PREFIX/opt/alphafold" > "$SITE/alphafold.pth"
   ```

   AlphaFold2 also does not ship `stereo_chemical_props.txt`, and AMBER relaxation
   fails without it (`FileNotFoundError` raised from
   `residue_constants.load_stereo_chemical_props`, reached via `amber_minimize`).
   Fetch the exact revision AlphaFold2's own Dockerfile pins:

   ```bash
   wget -q -O "$CONDA_PREFIX/opt/alphafold/alphafold/common/stereo_chemical_props.txt" \
     https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
   ```

   If that host is unreachable, this repository vendors a byte-identical copy:

   ```bash
   cp src/utils/openfold_violations/stereo_chemical_props.txt \
      "$CONDA_PREFIX/opt/alphafold/alphafold/common/"
   ```

   Verify the whole relaxation path:
   ```bash
   python3 -c "from alphafold.relax import relax; from alphafold.common import protein, residue_constants; print('alphafold OK')"
   ```

   AlphaFold2's remaining dependencies (`absl-py`, `dm-haiku`, `jax`, `ml-collections`,
   `tensorflow-cpu`, `docker`) are already satisfied by the steps above. A few are
   intentionally newer than AlphaFold2's `requirements.txt` asks for — `numpy` 1.26.4
   vs 1.24.3, `biopython` 1.83 vs 1.79, `ml-collections` 0.1.1 vs 0.1.0 — and the
   relaxation path works with these. Do **not** run
   `pip install -r requirements.txt` from the AlphaFold2 clone: it would downgrade
   numpy and biopython and break the rest of this environment.

### Troubleshooting

**`ModuleNotFoundError: No module named 'torch._six'`** — an old `deepspeed`
(&le; 0.5.9) is installed and is incompatible with torch &ge; 2.0. This project never
uses deepspeed (`use_deepspeed_evo_attention=False` is hardcoded); the vendored
openfold code now degrades with a warning instead of crashing. To silence it:
```bash
pip3 uninstall -y deepspeed
```

## Usage

### 1. Cryo-EM Guided Structure Prediction

Fits protein structures to electrostatic potential maps using cryo-EM data from the EMDB.

**Command:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_em.py <pdb_id> <emdb_id> <renumbered_file_path> <assembly_identifier> \
    --phenix_setup_sh <phenix_setup_path> \
    --sequences <seq1> <seq2> ... \
    --counts <count1> <count2> ... \
    [OPTIONS]
```

**Required Parameters:**
- `pdb_id`: PDB identifier for the protein structure
- `emdb_id`: EMDB identifier for the EM density map
- `renumbered_file_path`: Path to renumbered and reordered PDB file
- `assembly_identifier`: Identifier for the assembly (e.g., biological assembly name)
- `--phenix_setup_sh`: Path to Phenix setup shell script (e.g., `/path/to/phenix-1.21.2/phenix_env.sh`)
- `--sequences`: Space-separated sequences for each chain in the assembly
- `--counts`: Space-separated integer counts corresponding to each sequence (must match length of sequences)

**Optional Parameters:**
- `--dihedrals_file`: Path to dihedral restraints file
- `--noe_restraints_file`: Path to NOE restraints file  
- `--noe_pdb_file`: Path to NOE reference PDB file
- `--input_directory`: Directory for input files (default: `pipeline_inputs`)
- `--output_directory`: Directory for output files (default: `pipeline_outputs`)
- `--wandb_key`: Weights & Biases API key for experiment tracking
- `--wandb_project`: Weights & Biases project name
- `--device`: Compute device (default: `cuda:0`)

**Example:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_em.py 7dac 30622 pdb7dac_seqaligned_short.pdb amyloid_7dac_short_mmseq2 \
    --phenix_setup_sh  /opt/ccp4-8.0/bin/ccp4.setup-sh \
    --sequences PLVNIYNCSGVQVGDNNYLTMQQT \
    --counts 3 \
    --device cuda:0
```
The renumbered file is the path to the PDB file containing atomic coordinates where the residues were renumbered to match the absolute 1-index of the residues of the sequence. An example `pdb7dac_seqaligned_short.pdb` is included in the repository.

### 2. X-ray Crystallography Guided Structure Prediction

Generates ensemble structures fitted to X-ray crystallographic electron density maps.

**Command:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_xray.py <pdb_id> <chain_id> <region_sub_sequence> \
    --ccp4_setup_sh <ccp4_setup_path> \
    --phenix_setup_sh <phenix_setup_path> \
    [OPTIONS]
```

**Required Parameters:**
- `pdb_id`: PDB identifier for the protein structure
- `chain_id`: Chain identifier within the PDB structure (e.g., `A`, `B`)
- `region_sub_sequence`: Subsequence of amino acids defining the region of interest
- `--ccp4_setup_sh`: Path to CCP4 setup shell script (e.g., `/path/to/ccp4-8.0/bin/ccp4.setup-sh`)
- `--phenix_setup_sh`: Path to Phenix setup shell script

**Optional Parameters:**
- `--input_directory`: Directory for input files (default: `pipeline_inputs`)
- `--output_directory`: Directory for output files (default: `pipeline_outputs`)
- `--map_type`: Type of electron density map to use: `2fofc` (standard and quicker) or `end` (absolute scale END map and slower) (default: `end`)
- `--wandb_key`: Weights & Biases API key for experiment tracking
- `--wandb_project`: Weights & Biases project name  
- `--device`: Compute device (default: `cuda:0`)

**Example:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_xray.py 2izr A SLTGT \
    --ccp4_setup_sh /opt/ccp4-8.0/bin/ccp4.setup-sh \
    --phenix_setup_sh /opt/phenix-1.21.2/phenix_env.sh \
    --map_type end \
    --device cuda:0
```

### 3. NMR Guided Structure Prediction

Fits protein structures to NMR experimental restraints including NOE distances, dihedral angles, RDC, and relaxation data.

There are two input modes:

- **Deposited entry** — give a PDB ID and the restraints, coordinates and sequence are all fetched for you.
- **Custom restraints** — give your own restraint file, a sequence, and an identifier. No deposited entry is needed. See [Custom restraints](#custom-restraints).

#### Mode 1: deposited PDB entry

**Command:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_nmr.py <pdb_id> [OPTIONS]
```

**Required Parameters:**
- `pdb_id`: PDB identifier for the NMR structure

**Optional Parameters:**
- `--input_directory`: Directory containing NMR input files (default: `nmr_pipeline_inputs`)
  - Should contain subdirectories: `pdbs/`, `restraints/`, `metadata/`
- `--output_directory`: Directory for output files (default: `nmr_pipeline_outputs`)
- `--methyl_rdc_file`: Path to methyl RDC (Residual Dipolar Coupling) file
- `--amide_rdc_file`: Path to amide RDC file
- `--amide_relax_file`: Path to amide relaxation (S²) file
- `--methyl_relax_file`: Path to methyl relaxation file
- `--wandb_key`: Weights & Biases API key for experiment tracking
- `--wandb_project`: Weights & Biases project name
- `--device`: Compute device (default: `cuda:0`)

**Example:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_nmr.py 1u0p \
    --input_directory nmr_pipeline_inputs \
    --output_directory nmr_pipeline_outputs \
    --device cuda:0
```

#### Custom restraints

To guide an ensemble from your own data, supply an identifier, the sequence, and a
restraint file instead of a PDB ID. Nothing is fetched, and no deposited structure is
required.

**Command:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_nmr.py \
    --conformation_id <identifier> \
    --sequence <one_letter_sequence> \
    --restraints <path> \
    [OPTIONS]
```

**Required Parameters:**
- `--conformation_id`: Identifier for this run, used for output naming in place of a PDB ID
- `--sequences`: One-letter sequence per unique chain, space-separated (`--sequence` is a single-chain shorthand)
- `--restraints`: Path to a restraint file — either a `.csv` in the [format below](#restraint-file-format), or a `.str`/`.mr` NMR-STAR file which is converted automatically

**Optional Parameters:** all of the Mode 1 options, plus
- `--counts`: Number of copies of each entry in `--sequences`. Defaults to `1` each.
- `--sequence_types`: Molecule type per entry in `--sequences` — `proteinChain`, `rnaSequence` or `dnaSequence`. Defaults to all `proteinChain`.
- `--reference_pdb`: Reference structure to compare against. When supplied, the metrics table gains an `MD` row alongside the guided row; when omitted, metrics are guided-only.

> **The reference structure must retain its hydrogens.** Metrics are computed over the
> intersection of the atom sets of every structure being compared, so a
> hydrogen-stripped reference (a "fixed" or cleaned model) removes protons from the
> guided structures too. Almost every NOE restraint refers to a proton, so the
> evaluation silently shrinks to a handful of restraints and typically reports zero
> violations — which looks like a perfect fit rather than a failed comparison. Use the
> deposited NMR ensemble, and watch for the `only N/M NOE restraints are resolvable`
> warning. Mode 1 does this correctly for you.

**Example:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
python3 run_nmr.py \
    --conformation_id my_conf_A \
    --sequence GYIPEAPRDGQAYVRKDGEWVLLSTFL \
    --restraints /path/to/my_restraints.csv \
    --device cuda:0
```

##### Multiple chains

`--sequences`, `--counts` and `--sequence_types` are parallel lists, following the same
convention as `run_em.py`. A homotrimer is one sequence with a count of three:

```bash
python3 run_nmr.py --conformation_id trimer \
    --sequences GYIPEAPRDGQAYVRKDGEWVLLSTFL \
    --counts 3 \
    --restraints my_restraints.csv
```

A hetero-complex lists each unique chain, and molecule types can be mixed:

```bash
python3 run_nmr.py --conformation_id complex_AB \
    --sequences GYIPEAPRDGQAYVRKDGEWVLLSTFL MKTAYIAKQRQISFVK \
    --counts 2 1 \
    --sequence_types proteinChain proteinChain \
    --restraints my_restraints.csv
```

Chains are created in the order given — sequence 1's copies first, then sequence 2's —
and labelled `A`, `B`, `C`, … in that order. **The `chain1`/`chain2` values in your
restraint file must use those labels**, not the chain names from whatever structure the
restraints originally came from. Inter-chain restraints naming a chain that does not
exist are excluded from guidance with a warning; watch for
`inter-chain restraint(s) reference chains`.

Restraints without `chain1`/`chain2` columns are treated as within-chain and applied to
every chain independently. That is correct for a monomer or a homo-oligomer, where the
chains are identical copies.

**A hetero-complex needs the chain columns.** Without them every restraint is applied to
every chain, so one chain's restraints get scored against another chain's coordinates —
and wherever the residue numbers happen to coincide, a meaningless restraint is enforced
rather than skipped. Add `chain1`/`chain2` naming the chain each restraint belongs to and
each chain is then evaluated only against its own restraints. If chains differ in length
and the columns are missing, the run warns:
`this model has chains of differing length ... but the restraint file has no
'chain1'/'chain2' columns`.

Order-parameter losses (`--methyl_relax_file` and friends) are built from the first
chain's topology and applied to all chains, so they are rejected for constructs whose
chains differ in length. They are fine for homo-oligomers.

Mode 1 (PDB ID) builds a single-chain model. If the deposited entry has more than one
polymer chain it now fails with the equivalent multi-chain command rather than silently
keeping only the first chain.

#### Restraint file format

A restraint CSV is what an NMR-STAR `_mr.str` file becomes after conversion, and is what
both the guidance loss and the metrics read. Columns are looked up by name, so their
order does not matter. Passing a `.str`/`.mr` file to `--restraints` produces this
automatically; the schema is documented for hand-built or externally generated files.

**Required columns**

| Column | Type | Notes |
|---|---|---|
| `type` | string | Only rows equal to `NOE` are used; other rows are ignored |
| `constrain_id` | integer | OR-group id. Rows sharing an id are treated as alternatives, and only the least-violated one is penalised |
| `residue1_num` | integer | 1-based residue index into the supplied sequence |
| `residue1_id` | string | Three-letter uppercase residue name (`ALA`, `LEU`, …) |
| `atom1` | string | NMR atom name — see below |
| `residue2_num` | integer | |
| `residue2_id` | string | |
| `atom2` | string | |
| `lower_bound` | float or `.` | `.` is accepted and treated as `0` |
| `upper_bound` | float | Must be numeric — unlike `lower_bound`, `.` is **not** accepted here |

**Optional columns** — `chain1` and `chain2`. Supply both or neither. If absent, every
restraint is treated as within-chain. If present, rows where the two differ become
inter-chain restraints between the named chains.

**Ignored if present** — `member_id`, `member_logic`, `heavy_atom1`, `heavy_atom2`,
`distance`. The NMR-STAR converter emits these, but nothing reads them.

**Atom naming.** Hydrogens are built on the fly, so `atom1`/`atom2` use NMR naming
conventions: individual protons (`HA`, `HB2`, `HD21`), methyl pseudo-atoms (`MB`, `MD1`,
`MG2`, `ME`), and aromatic or amine pseudo-atoms (`QD`, `QE`, `QZ`). Names ending in `#`
are averaged over their `1`/`2` partners. Unrecognised names fall back to a heavy-atom
lookup, and any restraint that still cannot be resolved is skipped with a warning rather
than aborting the run.

**Minimal example** — the two `constrain_id: 1` rows are OR-alternatives, so satisfying
either one is enough:

```csv
type,constrain_id,residue1_num,residue1_id,atom1,residue2_num,residue2_id,atom2,lower_bound,upper_bound
NOE,1,3,LEU,MD1,17,VAL,HB,.,5.5
NOE,1,3,LEU,MD2,17,VAL,HB,.,5.5
NOE,2,5,TYR,QD,21,ALA,MB,1.8,4.0
```

Malformed files are rejected up front with a message naming the offending column,
rather than failing later inside the loss.

#### NMR output layout

Both modes write:

```
<output_directory>/<id>_nmr_guided/diffusion_process/
├── <id>_ensemble.pdb    # the ensemble: every relaxed structure as one MODEL
├── <id>_metrics.csv     # NOE violation metrics
└── verbose/             # per-structure intermediates (raw, _hyd_added, _colab_relaxed)
```

`<id>_ensemble.pdb` is the primary output. The per-structure files and their
hydrogenated and relaxed derivatives are kept under `verbose/`.

### 4. RMSD Guided Structure Prediction (oracle control)

Guides on the coordinate deviation between the structure being generated and a reference
structure, instead of an experimental measurement. There is nothing to model and nothing to
fit -- the objective is the answer itself. Two uses:

- check that the guidance machinery works end to end: loss -> gradient on the noisy
  coordinates -> denoising step actually moves samples where the loss points;
- measure the best RMSD the sampler can reach, as an upper bound on what any real
  experimental likelihood could achieve on the same construct.

It is not a structure determination method.

#### Running it

One command prepares the inputs and runs the guided/unguided pair:

```bash
python run_rmsd.py \
    --source-pdb path/to/crystal.pdb \
    --pdb-id my_complex \
    --keep-heteroatoms APR \
    --ligand-specifier CCD_APR --ligand-count 1 \
    --msa-source-dir path/to/msa \      # optional; queries the MSA server if omitted
    --run-unguided
```

Preparation and running can also be split:

```bash
# 1. clean the reference and write generated_configurations/<id>_rmsd_guided{,_unguided}.yaml
python -m src.utils.process_pipeline_inputs.preprocess_rmsd_inputs \
    --source-pdb path/to/crystal.pdb --pdb-id my_complex --keep-heteroatoms APR

# 2. check the correspondence, self-consistency and gradients before spending a run
python scripts/verify_rmsd_guidance.py generated_configurations/my_complex_rmsd_guided.yaml

# 3. run
python run_rmsd.py --config generated_configurations/my_complex_rmsd_guided.yaml \
    --unguided-config generated_configurations/my_complex_rmsd_guided_unguided.yaml \
    --run-unguided
```

`scripts/test_rmsd_reference.py` covers the atom map and the ligand-aware entity helpers on
CPU, with no model weights.

#### Input preparation

`preprocess_rmsd_inputs.py` writes the reference that scoring uses:

- the whole main structure is kept -- every polymer chain at full length, plus the
  heteroatom residues named by `--keep-heteroatoms`;
- waters, hydrogens and every other heteroatom are dropped;
- where a residue has alternate conformations, the **highest-occupancy** conformer is
  selected per atom name. This is deliberately not gemmi's
  `remove_alternative_conformations()`, which keeps whichever conformer comes first --
  frequently the minor one.

> gemmi reports "no alternate conformation" as `'\x00'`, not `''`. An atom in an ordinary
> single-conformer residue is therefore **truthy**, and a ligand refined at partial
> occupancy usually carries a real altloc letter -- so a filter written as
> `if not atom.altloc` drops the ligand. Compare against the sentinel explicitly.

It then reads the composition back out of the cleaned file and fills the config in from it:
the entity sequences, the chain map, and the ligand's residue offset. Nothing about the
construct is hardcoded.

#### The atom index map

The correspondence between the model's atom ordering and the reference is built explicitly
(`src/utils/rmsd_reference.py`), keyed on `(chain, residue number, atom name)`, validated on
residue name, and cached against the reference file and the model atom ordering it was built
for. Positional zipping would be wrong even where the atom counts agree: for ADP-ribose the
model and a crystal file agree on all 36 atom *names* but not on their order.

`chain_map` connects the two numbering schemes. Keys may be the featurizer's chain ids
(`A0`, `B1`, `C0`) or the plain letters. A value is either a bare reference chain name, or:

```yaml
chain_map:
  A: A                                          # protein copy 1 -> reference chain A
  B: B                                          # protein copy 2 -> reference chain B
  C: {reference_chain: A, residue_offset: 200}   # ligand: own chain from 1 in the model,
                                                 # residue 201 of chain A in the crystal
```

`residue_offset` is added to the model residue number to reach the reference one. A model
chain with no entry is an error rather than a silently unscored part of the structure
(`require_all_chains_mapped`).

#### Alignment and scoring sets

Superposition and scoring are independent selections, both defaulting to every mapped atom:

```yaml
alignment_selection: null    # null means "everything mapped"
scoring_selection:
  entity_kinds: [ligand]     # superpose on everything, score the ligand alone
                             # -> the ligand's pose in its pocket
```

Selection fields are `model_chains`, `entity_kinds` (`protein`/`ligand`/`dna`/`rna`),
`atom_names`, and `reference_residue_ranges` (`{chain: [[lo, hi], ...]}`). Every field left
null widens rather than narrows.

#### Configuration

Everything is in the config -- see `pipeline_configurations/rmsd_baseline.yaml`:

| key | meaning |
|---|---|
| `loss_function.rmsd_loss_function.reference_pdb` | reference structure to score against |
| `loss_function.rmsd_loss_function.chain_map` | model chain -> reference chain (+ residue offset) |
| `loss_function.rmsd_loss_function.alignment_selection` | atoms to superpose on |
| `loss_function.rmsd_loss_function.scoring_selection` | atoms to score |
| `loss_function.rmsd_loss_function.objective` | `rmsd` (loss is in Angstroms) or `msd` |
| `loss_function.rmsd_loss_function.weight` | weight in the combined loss |
| `loss_function.rmsd_loss_function.cache_dir` | where the built atom index map is cached |
| `diffusion_process.guidance.guidance_direction_scale_factor` | guidance strength |

#### Ligands

Protenix handles a CCD ligand natively; a ligand entity is declared alongside the polymers,
using the same `sequence` field:

```yaml
sequences:
  - sequence: "SMVNSFSGYLKL..."
    count: 2
    sequence_type: "proteinChain"
  - sequence: "CCD_APR"
    count: 1
    sequence_type: "ligand"
```

The ligand is a scored, gradient-carrying part of the structure -- not held static and not
excluded -- and lands in the output PDB as its own chain of HETATM records.

Note that `violation_loss_weight` scores protein stereochemistry only (peptide geometry and
the amino-acid atom14 layout); non-protein chains are skipped. `bond_length_loss_weight`
covers all entity types.

#### RMSD output layout

```
<output_directory>/<id>_rmsd_guided/
├── diffusion_process/
│   ├── <id>_0.pdb ... <id>_N.pdb    # the ensemble, protein chains + ligand chain
│   └── rmsd_report.json             # per-sample RMSD, total and per entity kind
└── metadata.json
<output_directory>/<id>_rmsd_guided_runs_summary.json   # guided vs unguided comparison
```

#### The RMSD trajectory across the diffusion process

Every run records the RMSD at each denoising step the loss is evaluated on and writes
`diffusion_process/rmsd_trajectory.csv` -- long form, one row per step per sample per
selection. The breakdown is computed anyway to form the loss, so this costs nothing;
`track_trajectory` and `trajectory_every_n_steps` control it.

An **unguided** run records no trajectory, because the diffusion loop only calls the loss
when guidance is on. The baseline curve therefore comes from the third config preparation
writes, `<id>_rmsd_guided_monitor.yaml`: guidance machinery on with
`guidance_direction_scale_factor: 0`, so the loss is measured at every step while the
denoising step itself is untouched -- the direction enters the update as
`delta + step_size * direction`, and scaling it to zero leaves `delta` exactly as it was.

```bash
python run_rmsd.py --config generated_configurations/<id>_rmsd_guided.yaml
python run_rmsd.py --config generated_configurations/<id>_rmsd_guided_monitor.yaml

python scripts/plot_rmsd_trajectory.py \
    --runs <id> [<id> ...] --output-dir rmsd_pipeline_outputs/<series name>
```

The plot is one small multiple per dataset, RMSD against denoising step on a log axis, with
the sampler's own noise level on the top axis so the step index can be read as a noise
level. Light and dark versions are written as PNG and PDF.

Note the trajectory covers steps `0 .. N-2`: the loop skips the loss on the final step. The
end-point value for the finished structures is in `rmsd_report.json`.

#### Comparing several datasets

`scripts/collect_rmsd_series.py` gathers a set of runs into one tidy directory -- readable
tables, a wide CSV and a long-form per-sample CSV, the superposed structures, and the
pairwise distances between the datasets' own references:

```bash
python scripts/collect_rmsd_series.py \
    --guided-configs generated_configurations/*_rmsd_guided.yaml \
    --output-dir rmsd_pipeline_outputs/<series name>
```

It looks for each run's unguided counterpart at `<config>_unguided.yaml`, reads the ligand
occupancy out of each reference to label the dataset, and skips anything with no output.

The reference-to-reference distances matter for reading such a table: two runs aimed at two
different targets cannot both reach zero, so how far apart the references are is the floor
for any comparison across datasets.

#### Visualizing against the reference

The structures in `diffusion_process/` are in the **model's own frame**, not the reference
frame -- the RMSD is computed after an internal superposition, which the coordinates do not
carry. Loading them straight into a viewer next to the reference will not overlay them.

`scripts/score_rmsd_output.py --write-superposed <dir>` writes each structure moved into the
reference frame by the whole-structure fit, next to a copy of the reference:

```bash
python scripts/score_rmsd_output.py generated_configurations/<id>_rmsd_guided.yaml \
    --write-superposed <output_directory>/<id>_rmsd_guided/superposed
```

The transform is applied to every atom, so load them without letting the viewer re-fit
anything -- `align`/`super` would compute its own superposition over a different atom set and
report a different number from the one above:

```
# PyMOL
load superposed/reference.pdb
load superposed/<id>_0_superposed.pdb
# no align / super -- the coordinates are already in the reference frame
rms_cur <id>_0_superposed, reference
```

Chain naming differs between the two on purpose, and the chain map is what reconciles them:
the model writes the ligand as its own chain (`C`, residue 1) while a crystal usually has it
inside a protein chain (here `A`, residue 201).

## Experiment Tracking

The pipeline supports experiment tracking via Weights & Biases (wandb). To enable tracking:

1. Create a wandb account at https://wandb.ai
2. Obtain your API key from https://wandb.ai/authorize
3. Pass the API key and project name to any run script:
   ```bash
   --wandb_key <your_api_key> --wandb_project <project_name>
   ```

## Citation
Please cite the following papers if you use this software:
```
@article{maddipatla2025experiment,
  title={Experiment-guided AlphaFold3 resolves accurate protein ensembles},
  author={Maddipatla, Advaith and Bojan Sellam, Nadav and Bojan, Meital and Masalitin, Volodymyr and Vedula, Sanketh and Schanda, Paul M and Marx, Ailie and Bronstein, Alexander M},
  journal={bioRxiv},
  pages={2025--10},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
```
@inproceedings{maddipatla2025inverse,
  title={Inverse problems with experiment-guided AlphaFold},
  author={Maddipatla, Advaith and Sellam, Nadav Bojan and Bojan, Meital and Vedula, Sanketh and Schanda, Paul and Marx, Ailie and Bronstein, Alex M},
  year={2025}
  booktitle={Forty-second International Conference on Machine Learning},
}
```
```
@article{maddipatla2024generative,
  title={Generative modeling of protein ensembles guided by crystallographic electron densities},
  author={Maddipatla, Sai Advaith and Sellam, Nadav Bojan and Vedula, Sanketh and Marx, Ailie and Bronstein, Alex},
  journal={arXiv preprint arXiv:2412.13223},
  year={2024}
}
```
## License

Soon.

## Contact

Correspondence Email: `Alexander.Bronstein@ist.ac.at`
