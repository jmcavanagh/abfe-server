"""Command-line entry point for GID4 docking + pose selection.

Example
-------
    conda run -n docking-reward python -m docking.run_docking \
        --smiles "CC(=O)Nc1ccc(CCN)cc1" \
        --protein data/9qdz.pdb \
        --config smileyllama-gid4/examples/sl_config_gid4_v3_1b_3kcal_sbdist.yml \
        --out runs/lig001

Outputs ``best_pose.sdf``, ``docking_log.json``, ``docking_log.txt`` (plus the
intermediate Vina/PLIP files) under ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import dock_smiles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="docking.run_docking",
        description="Dock a SMILES to GID4 (9QDZ) and select the best-scoring pose for ABFE.",
    )
    p.add_argument("--smiles", required=True, help="Ligand SMILES.")
    p.add_argument("--protein", required=True, help="Protein PDB (e.g. data/9qdz.pdb).")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--config", default=None,
                   help="SmileyLlama scoring YAML (defaults to built-in GID4 v3 params).")
    p.add_argument("--vina-exec", default="vina", help="Vina executable (default: vina).")
    p.add_argument("--obabel-exec", default="obabel", help="OpenBabel executable.")
    p.add_argument("--exhaustiveness", type=int, default=32, help="Vina exhaustiveness per replica.")
    p.add_argument("--num-modes", type=int, default=20, help="Poses per Vina replica.")
    p.add_argument("--n-replicas", type=int, default=8,
                   help="Independent Vina runs (different seeds) for pose sampling.")
    p.add_argument("--total-cpus", type=int, default=None,
                   help="Total CPU cores to use (default: all detected).")
    p.add_argument("--plip-nprocs", type=int, default=None,
                   help="Processes for PLIP interaction analysis (default: total cpus).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.protein).is_file():
        print(f"error: protein not found: {args.protein}", file=sys.stderr)
        return 2

    result = dock_smiles(
        smiles=args.smiles,
        protein_pdb=args.protein,
        out_dir=args.out,
        config_yaml=args.config,
        vina_exec=args.vina_exec,
        obabel_exec=args.obabel_exec,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        n_replicas=args.n_replicas,
        total_cpus=args.total_cpus,
        plip_nprocs=args.plip_nprocs,
    )

    print((out := Path(args.out) / "docking_log.txt").read_text())
    if not result.success:
        print(f"DOCKING FAILED: {result.message}", file=sys.stderr)
        return 1
    print(f"Best pose written to: {result.sdf_path}")
    print(f"Full log: {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
