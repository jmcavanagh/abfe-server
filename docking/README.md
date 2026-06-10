# GID4 docking → pose selection for ABFE

Docks a SMILES to **GID4 (9QDZ)** with AutoDock Vina, samples many poses across
CPU cores, scores every pose with the GID4 composite score, and writes the
single best pose as an SDF ready for ABFE.

## What the score is

For each docked pose, lower is better:

```
composite = vina_affinity − interaction_bonus
```

`interaction_bonus` is the sum of per-residue bonuses for the interactions the
pose makes, detected by **PLIP**. The residues and weights come straight from
the SmileyLlama config YAML (`scores.gid4_dock.parameters`):

| residue | interaction | bonus |
|--------:|-------------|------:|
| 237 (GLU) | salt bridge | 3.0 (distance-scaled 2.5–5.5 Å) |
| 132, 258 | H-bond | 0.1 each |
| 171, 240, 254, 273 | hydrophobic | 0.1 each |

This is a faithful, standalone port of `GID4DockingScoreV3` from the (private)
`smileyllama` package — it reads the **same** YAML so the two never drift, but
it does **not import** that package, so this module can live in its own env.

Only poses within `pose_energy_window` (3 kcal/mol) of the best raw pose are
analysed with PLIP; the molecular-weight / ion penalties are per-molecule
constants (they don't change which pose wins) and are reported, not used for
ranking. The pose with the most negative composite is selected.

## Usage

```bash
conda env create -f docking/environment.yml      # one-time
conda activate docking

python -m docking.run_docking \
    --smiles "CC(N)Cc1ccccc1" \
    --protein data/9qdz.pdb \
    --config smileyllama-gid4/examples/sl_config_gid4_v3_1b_3kcal_sbdist.yml \
    --out runs/lig001
```

Key flags (see `--help`): `--exhaustiveness` (default 32), `--num-modes` (20),
`--n-replicas` (8 independent Vina runs with different seeds), `--total-cpus`
(all cores by default — splits cores across replicas), `--plip-nprocs`.
On a GPU node, install `unidock` and pass `--vina-exec unidock` (it shares
Vina's CLI/output format).

`--config` is optional; omitted, the built-in defaults match
`sl_config_gid4_v3_1b_3kcal_sbdist.yml`.

## Outputs (under `--out`)

| file | contents |
|------|----------|
| `best_pose.sdf` | the selected pose (3D), with score/interaction SD tags — **the ABFE input** |
| `docking_log.txt` | human-readable summary |
| `docking_log.json` | full record: every scored pose, Vina affinity, bonus breakdown, interactions, salt-bridge distance, config used |
| `9qdz_clean.pdb`, `9qdz_receptor.pdbqt`, `ligand.pdbqt`, `vina_out/` | intermediates |

## Environment isolation

This package depends **only** on the docking stack (rdkit, meeko, plip,
dimorphite-dl, vina/obabel, pyyaml) — never on `smileyllama`, openmm, openff, or
the server code. Keep it in the `docking` conda env; the ABFE/mbar and server
code should each get their own env and call this module across a process
boundary (subprocess / CLI / job queue), passing `best_pose.sdf` downstream.
That way nobody's dependency solve can break anyone else's.
