"""Ligand preparation: protonation, 3D embedding, PDBQT export.

Ported from ``smileyllama.score.gid4_dock_v3`` so the ligand handed to Vina is
prepared identically to the RL scorer (dimorphite-dl protonation at pH 7.4
optionally patched with a custom SMARTS file, RDKit ETKDG embed + MMFF, then
Meeko PDBQT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy

ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def _patch_dimorphite_smarts(smarts_file: Optional[str]) -> None:
    if not smarts_file or not Path(smarts_file).is_file():
        return
    from dimorphite_dl.protonate.data import PKaData

    def _load_lines_custom():
        with open(smarts_file) as f:
            return [s.strip() for s in f if s.strip() and not s.strip().startswith("#")]

    PKaData._load_lines = staticmethod(_load_lines_custom)
    PKaData._instance = None
    PKaData._data = []


def protonate_smiles(
    smiles: str, ph: float = 7.4, smarts_file: Optional[str] = None,
) -> str:
    """Return the dominant protonation state at ``ph`` (most positive variant).

    Mirrors V3: among dimorphite-dl variants, pick the one with the highest net
    formal charge (favouring a protonated basic amine for the GID4 salt bridge).
    Falls back to the input SMILES on any failure.
    """
    _patch_dimorphite_smarts(smarts_file)
    try:
        from dimorphite_dl import protonate_smiles as _dimorphite

        variants = _dimorphite(smiles, ph_min=ph, ph_max=ph, max_variants=128)
        if not variants:
            return smiles
        best, best_charge = variants[0], -999
        for v in variants:
            mol = Chem.MolFromSmiles(v)
            if mol:
                fc = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                if fc > best_charge:
                    best_charge, best = fc, v
        return best
    except Exception:
        return smiles


def positive_charge_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(max(0, a.GetFormalCharge()) for a in mol.GetAtoms())


def molecular_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.ExactMolWt(mol) if mol is not None else float("nan")


def prepare_ligand_pdbqt(
    smiles: str, out_pdbqt: str | Path, seed: int = 42,
) -> Tuple[bool, str]:
    """Embed and write a ligand PDBQT. Returns ``(success, message)``."""
    out_pdbqt = Path(out_pdbqt)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid SMILES"
    if any(a.GetSymbol() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
        return False, "contains element outside docking-supported set"
    try:
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            return False, "3D embedding failed"
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        pdbqt = PDBQTWriterLegacy.write_string(
            MoleculePreparation(rigid_macrocycles=True)(mol)[0]
        )[0]
        n_atoms = sum(line.startswith("ATOM") for line in pdbqt.split("\n"))
        n_branch = sum(line.startswith("BRANCH") for line in pdbqt.split("\n"))
        if n_atoms > 140:
            return False, f"too many atoms ({n_atoms} > 140)"
        if n_branch > 40:
            return False, f"too many rotatable branches ({n_branch} > 40)"
        if pdbqt.strip() == "":
            return False, "empty PDBQT"
        out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        out_pdbqt.write_text(pdbqt)
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"preparation error: {e}"
