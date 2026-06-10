"""Run Vina, sample poses across replicas, and convert poses to SDF."""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
from meeko import PDBQTMolecule, RDKitMolCreate


@dataclass
class Pose:
    """One docked binding mode."""

    replica: int
    model_idx: int          # 0-based index of the MODEL within its replica file
    vina_score: float       # docking affinity (kcal/mol, lower = better)
    pdbqt_path: Path        # multi-model replica file this pose lives in
    pose_id: int = -1       # global id assigned after pooling


def run_vina_replica(
    *, exec_path: str, receptor: Path, ligand: Path, out_pdbqt: Path,
    log_path: Path, box_center, box_size, exhaustiveness: int, num_modes: int,
    seed: int, cpu: int,
) -> Path:
    """Run one Vina docking replica; returns the output PDBQT path."""
    cmd = [
        exec_path,
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(box_center[0]),
        "--center_y", str(box_center[1]),
        "--center_z", str(box_center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
        "--cpu", str(cpu),
        "--out", str(out_pdbqt),
    ]
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        # Non-fatal: a failed replica just contributes no poses.
        with open(log_path, "a") as logf:
            logf.write(f"\n[run_vina_replica] exit code {proc.returncode}\n")
    return out_pdbqt


def sample_poses(
    *, exec_path: str, receptor: Path, ligand: Path, workdir: Path,
    box_center, box_size, exhaustiveness: int, num_modes: int,
    n_replicas: int, cpu_per_replica: int, thresh_for_fail: float,
) -> List[Pose]:
    """Run ``n_replicas`` Vina runs concurrently and pool all valid poses."""
    out_dir = workdir / "vina_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    with ThreadPoolExecutor(max_workers=n_replicas) as pool:
        for r in range(n_replicas):
            out_pdbqt = out_dir / f"replica_{r}.pdbqt"
            log_path = out_dir / f"replica_{r}.log"
            jobs.append(pool.submit(
                run_vina_replica,
                exec_path=exec_path, receptor=receptor, ligand=ligand,
                out_pdbqt=out_pdbqt, log_path=log_path,
                box_center=box_center, box_size=box_size,
                exhaustiveness=exhaustiveness, num_modes=num_modes,
                seed=r + 1, cpu=cpu_per_replica,
            ))
        for j in jobs:
            j.result()

    poses: List[Pose] = []
    for r in range(n_replicas):
        out_pdbqt = out_dir / f"replica_{r}.pdbqt"
        for model_idx, score in _parse_pose_scores(out_pdbqt):
            if math.isnan(score) or score < thresh_for_fail:
                continue
            poses.append(Pose(replica=r, model_idx=model_idx,
                              vina_score=score, pdbqt_path=out_pdbqt))

    poses.sort(key=lambda p: p.vina_score)
    for i, p in enumerate(poses):
        p.pose_id = i
    return poses


def _parse_pose_scores(out_pdbqt: Path) -> List[Tuple[int, float]]:
    """Yield (model_idx, vina_affinity) for each model in a Vina output file."""
    if not out_pdbqt.is_file():
        return []
    results: List[Tuple[int, float]] = []
    model_idx = -1
    saw_models = False
    with open(out_pdbqt) as f:
        for line in f:
            if line.startswith("MODEL"):
                saw_models = True
                model_idx += 1
            elif line.startswith("REMARK VINA RESULT"):
                try:
                    score = float(line.split()[-3])
                except (ValueError, IndexError):
                    continue
                idx = model_idx if saw_models else len(results)
                results.append((idx, score))
    return results


def parse_charges(ligand_pdbqt: Path) -> Dict[int, int]:
    """Read {atom_serial: formal_charge} once from the input ligand PDBQT."""
    from .scoring import _parse_pdbqt_charges

    return _parse_pdbqt_charges(str(ligand_pdbqt))


def _extract_model_block(pdbqt_path: Path, model_idx: int) -> Optional[str]:
    """Return the full PDBQT text of a single model (incl. REMARK/torsion tree)."""
    header: List[str] = []          # REMARKs before the first MODEL (SMILES etc.)
    block: List[str] = []
    current = -1
    in_target = False
    saw_models = False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                saw_models = True
                current += 1
                in_target = (current == model_idx)
                continue
            if line.startswith("ENDMDL"):
                if in_target:
                    break
                in_target = False
                continue
            if not saw_models:
                header.append(line)
                in_target = (model_idx == 0)
            if in_target:
                block.append(line)
    if not block:
        return None
    pre = header if saw_models else []
    return "".join(pre + block)


def pose_to_sdf(
    pose: Pose, sdf_path: Path, properties: Optional[Dict[str, str]] = None,
) -> bool:
    """Write a single pose to SDF, attaching ``properties`` as SD tags."""
    block = _extract_model_block(pose.pdbqt_path, pose.model_idx)
    if block is None:
        return False
    tmp = sdf_path.with_suffix(".pose.pdbqt")
    tmp.write_text(block)
    try:
        pmol = PDBQTMolecule.from_file(str(tmp), skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        mol = next((m for m in mols if m is not None), None)
        if mol is None:
            return False
        if properties:
            for k, v in properties.items():
                mol.SetProp(str(k), str(v))
        sdf_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        return True
    finally:
        tmp.unlink(missing_ok=True)
