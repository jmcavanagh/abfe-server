"""Protein preparation for GID4 docking.

Produces two artifacts from a raw PDB:
  * a cleaned ``*.pdb`` (waters/ligands stripped, peptide caps and protonation
    variants normalised to ATOM records) used by PLIP for residue-numbered
    interaction detection, and
  * a rigid receptor ``*.pdbqt`` (via OpenBabel) used by Vina.

PLIP only reads ATOM/TER records, so any His/cap atoms stored as HETATM in the
input are rewritten to ATOM here; otherwise those residues would be invisible
to interaction detection.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Set, Tuple

WATER_RESNAMES: Set[str] = {"HOH", "WAT", "TIP", "TIP3", "SOL", "DOD"}

# Residues that are part of the protein chain (standard + protonation variants
# + common terminal caps) and must be kept as ATOM records.
PROTEIN_RESNAMES: Set[str] = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation / tautomer variants
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
    "LYN", "TYM", "ARN", "MSE",
    # terminal caps
    "ACE", "NME", "NMA", "FOR", "NH2",
}


def clean_protein(pdb_in: str | Path, pdb_out: str | Path) -> Tuple[Path, List[str]]:
    """Strip waters and non-protein heteroatoms; normalise protein records to ATOM.

    Returns ``(output_path, dropped_resnames)`` where ``dropped_resnames`` lists
    any non-water HETATM residue names that were removed (ions, cofactors, an
    existing ligand, …) so the caller can warn about them.
    """
    pdb_in, pdb_out = Path(pdb_in), Path(pdb_out)
    out_lines: List[str] = []
    dropped: Set[str] = set()

    with open(pdb_in) as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                resname = line[17:20].strip()
                if resname in WATER_RESNAMES:
                    continue
                if rec == "HETATM":
                    if resname in PROTEIN_RESNAMES:
                        line = "ATOM  " + line[6:]  # reclassify as protein
                    else:
                        dropped.add(resname)
                        continue
                out_lines.append(line.rstrip("\n"))
            elif rec == "TER":
                out_lines.append(line.rstrip("\n"))
            elif rec in ("MODEL", "ENDMDL"):
                # keep only the first model of an NMR/multi-model file
                if rec == "ENDMDL":
                    break

    out_lines.append("END")
    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    pdb_out.write_text("\n".join(out_lines) + "\n")
    return pdb_out, sorted(dropped)


def prepare_receptor_pdbqt(
    clean_pdb: str | Path, pdbqt_out: str | Path, obabel_exec: str = "obabel",
) -> Path:
    """Convert a cleaned protein PDB to a rigid receptor PDBQT via OpenBabel."""
    clean_pdb, pdbqt_out = Path(clean_pdb), Path(pdbqt_out)
    pdbqt_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [obabel_exec, "-ipdb", str(clean_pdb), "-opdbqt", "-O", str(pdbqt_out), "-xr"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not pdbqt_out.is_file():
        raise RuntimeError(
            f"obabel receptor prep failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    return pdbqt_out


def prepare_protein(
    pdb_in: str | Path, workdir: str | Path, obabel_exec: str = "obabel",
) -> Tuple[Path, Path, List[str]]:
    """Run the full protein prep, returning (clean_pdb, receptor_pdbqt, dropped)."""
    workdir = Path(workdir)
    stem = Path(pdb_in).stem
    clean_pdb, dropped = clean_protein(pdb_in, workdir / f"{stem}_clean.pdb")
    receptor = prepare_receptor_pdbqt(clean_pdb, workdir / f"{stem}_receptor.pdbqt", obabel_exec)
    return clean_pdb, receptor, dropped
