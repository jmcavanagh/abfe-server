"""Per-pose interaction analysis and composite scoring for GID4 docking.

Ported (standalone, no smileyllama import) from
``smileyllama.score.gid4_dock_v3``.  Given a protein PDB and a docked-pose
PDBQT model, PLIP detects hydrogen bonds, hydrophobic contacts, pi-stacking
and salt bridges to each residue; the configured per-residue bonuses are then
subtracted from the Vina docking energy to form a composite (lower = better).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from rdkit import Chem

from .config import ScoringConfig


# ---------------------------------------------------------------------------
# PDBQT -> PLIP-ready ligand records
# ---------------------------------------------------------------------------

def _parse_pdbqt_charges(pdbqt_path: str) -> Dict[int, int]:
    """Extract formal charges from REMARK SMILES / SMILES IDX lines."""
    smiles = None
    idx_nums = []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("REMARK SMILES IDX"):
                idx_nums.extend(line.split()[3:])
            elif line.startswith("REMARK SMILES") and "IDX" not in line:
                smiles = line.split(None, 2)[2].strip()
            elif line.startswith("ATOM"):
                break
    if not smiles or not idx_nums:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    charges = {}
    for i in range(0, len(idx_nums) - 1, 2):
        smiles_1based = int(idx_nums[i])
        pdbqt_serial = int(idx_nums[i + 1])
        rdkit_idx = smiles_1based - 1
        if rdkit_idx < mol.GetNumAtoms():
            fc = mol.GetAtomWithIdx(rdkit_idx).GetFormalCharge()
            if fc != 0:
                charges[pdbqt_serial] = fc
    return charges


def _pdbqt_model_to_hetatm(pdbqt_path: str, model_idx: int = 0, charges=None):
    """Extract one model from a multi-model PDBQT as HETATM LIG records.

    Preserves PDBQT atom-type strings in the element column so OpenBabel keeps
    polar hydrogens, and appends formal-charge annotations so PLIP can detect
    charged groups for salt bridges.

    ``charges`` may be a precomputed {atom_serial: formal_charge} map (e.g. read
    once from the input ligand PDBQT); Vina output files do not always retain
    the ``REMARK SMILES`` lines this is otherwise derived from.
    """
    if charges is None:
        charges = _parse_pdbqt_charges(pdbqt_path)
    lines = []
    current_model = -1
    in_target = False
    has_models = False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                has_models = True
                current_model += 1
                in_target = (current_model == model_idx)
                continue
            if line.startswith("ENDMDL"):
                if in_target:
                    break
                in_target = False
                continue
            if not has_models and model_idx == 0:
                in_target = True
            if in_target and line.startswith(("ATOM", "HETATM")):
                serial = int(line[6:11].strip())
                base = line[:54].ljust(54)
                new_line = "HETATM" + base[6:17] + " LIG X   1" + base[27:]
                elem = line[77:79].strip() if len(line) > 77 else ""
                if not elem:
                    atom_name = line[12:16].strip()
                    elem = re.sub(r"[^A-Za-z]", "", atom_name)
                    elem = elem[0] if elem else "C"
                new_line = new_line.ljust(76) + elem.rjust(2)
                charge = charges.get(serial, 0)
                if charge != 0:
                    new_line += "%d%s" % (abs(charge), "+" if charge > 0 else "-")
                lines.append(new_line)
    return lines


def analyze_pose_interactions(
    protein_pdb_path: str, pdbqt_path: str, model_idx: int, charges=None,
) -> Tuple[Dict[int, Set[str]], Dict[int, float]]:
    """Run PLIP on one protein-ligand pose.

    Returns ``(interactions, sb_distances)`` where ``interactions`` maps a
    residue number to the set of interaction types it makes with the ligand
    ({'hbond', 'hydrophobic', 'saltbridge'}), and ``sb_distances`` maps a
    residue number to its shortest salt-bridge distance.
    """
    from plip.structure.preparation import PDBComplex

    if not os.path.isfile(pdbqt_path):
        return {}, {}
    lig_lines = _pdbqt_model_to_hetatm(pdbqt_path, model_idx, charges=charges)
    if not lig_lines:
        return {}, {}

    prot_lines = []
    with open(protein_pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "TER")):
                prot_lines.append(line.rstrip())

    complex_str = "\n".join(prot_lines + ["TER"] + lig_lines + ["END"]) + "\n"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(complex_str)
            tmp_path = f.name
        mol = PDBComplex()
        mol.load_pdb(tmp_path)
        if not mol.ligands:
            return {}, {}
        for lig in mol.ligands:
            mol.characterize_complex(lig)

        interactions: Dict[int, Set[str]] = {}
        sb_distances: Dict[int, float] = {}
        for _, site in mol.interaction_sets.items():
            for hb in site.hbonds_pdon + site.hbonds_ldon:
                interactions.setdefault(hb.resnr, set()).add("hbond")
            for hp in site.hydrophobic_contacts:
                interactions.setdefault(hp.resnr, set()).add("hydrophobic")
            for ps in site.pistacking:
                interactions.setdefault(ps.resnr, set()).add("hydrophobic")
            for sb in site.saltbridge_lneg + site.saltbridge_pneg:
                interactions.setdefault(sb.resnr, set()).add("saltbridge")
                if sb.resnr not in sb_distances or sb.distance < sb_distances[sb.resnr]:
                    sb_distances[sb.resnr] = sb.distance
        return interactions, sb_distances
    except Exception:
        return {}, {}
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def plip_worker(args):
    """Top-level wrapper so PLIP analysis can run under multiprocessing.Pool.

    ``args`` = (protein_pdb_path, pdbqt_path, model_idx, charges_dict).
    """
    protein_pdb_path, pdbqt_path, model_idx, charges = args
    return analyze_pose_interactions(protein_pdb_path, pdbqt_path, model_idx, charges=charges)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_pose_bonus(
    interactions: Dict[int, Set[str]],
    cfg: ScoringConfig,
    sb_distances: Optional[Dict[int, float]] = None,
    allow_saltbridge: bool = True,
) -> Tuple[float, Dict[Tuple[int, str], float]]:
    """Total interaction bonus for one pose, plus a per-(resnr, type) breakdown.

    Salt bridges are distance-scaled between ``sb_dist_best`` and
    ``sb_dist_max`` when ``sb_dist_best`` is set.
    """
    if sb_distances is None:
        sb_distances = {}
    bonus = 0.0
    breakdown: Dict[Tuple[int, str], float] = {}
    for resnr, type_bonuses in cfg.interaction_bonuses.items():
        if resnr not in interactions:
            continue
        for itype, value in type_bonuses.items():
            if itype == "saltbridge" and not allow_saltbridge:
                continue
            if itype in interactions[resnr]:
                awarded = value
                if itype == "saltbridge" and cfg.sb_dist_best is not None:
                    dist = sb_distances.get(resnr)
                    if dist is not None:
                        scale = (cfg.sb_dist_max - dist) / (cfg.sb_dist_max - cfg.sb_dist_best)
                        awarded = value * max(0.0, min(1.0, scale))
                bonus += awarded
                breakdown[(resnr, itype)] = awarded
    return bonus, breakdown
