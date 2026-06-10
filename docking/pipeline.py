"""End-to-end: dock a SMILES to GID4 (9QDZ), score poses, pick the best one.

Selection objective (per pose, lower = better), ported from
``GID4DockingScoreV3``:

    composite = vina_affinity - interaction_bonus

where ``interaction_bonus`` sums the configured per-residue bonuses for salt
bridges / H-bonds / hydrophobic contacts the pose makes (salt bridges
distance-scaled).  The molecular-weight and ion penalties are per-molecule
constants and do not affect which *pose* wins; they are reported for the
record.  The selected pose is the one whose composite is most negative.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ScoringConfig, load_scoring_config
from .ligand import (
    molecular_weight, positive_charge_count, prepare_ligand_pdbqt, protonate_smiles,
)
from .poses import Pose, parse_charges, pose_to_sdf, sample_poses
from .protein import prepare_protein
from .scoring import compute_pose_bonus, plip_worker


@dataclass
class PoseScore:
    pose: Pose
    bonus: float
    composite: float
    breakdown: Dict[Tuple[int, str], float] = field(default_factory=dict)
    interactions: Dict[int, list] = field(default_factory=dict)
    sb_distances: Dict[int, float] = field(default_factory=dict)


@dataclass
class DockingResult:
    smiles: str
    protonated_smiles: str
    success: bool
    message: str
    best: Optional[PoseScore]
    scored_poses: List[PoseScore]
    n_poses_total: int
    n_poses_qualifying: int
    mw: float
    mw_penalty: float
    positive_charges: int
    allow_saltbridge: bool
    final_score: float          # higher = better (= -(composite + penalties))
    sdf_path: Optional[Path]
    log_path: Optional[Path]


def dock_smiles(
    smiles: str,
    protein_pdb: str | Path,
    out_dir: str | Path,
    *,
    config: Optional[ScoringConfig] = None,
    config_yaml: Optional[str | Path] = None,
    vina_exec: str = "vina",
    obabel_exec: str = "obabel",
    exhaustiveness: int = 32,
    num_modes: int = 20,
    n_replicas: int = 8,
    total_cpus: Optional[int] = None,
    plip_nprocs: Optional[int] = None,
) -> DockingResult:
    """Dock ``smiles`` to ``protein_pdb`` and select the best-scoring pose."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or load_scoring_config(config_yaml, protein_key="9qdz")

    total_cpus = total_cpus or os.cpu_count() or 1
    n_replicas = max(1, min(n_replicas, total_cpus))
    cpu_per_replica = max(1, total_cpus // n_replicas)
    plip_nprocs = plip_nprocs or total_cpus

    log = {"smiles": smiles, "warnings": []}

    # 1. Protein prep -------------------------------------------------------
    clean_pdb, receptor, dropped = prepare_protein(protein_pdb, out_dir, obabel_exec)
    if dropped:
        log["warnings"].append(
            f"Dropped non-protein HETATM residues from receptor: {dropped}"
        )

    # 2. Ligand prep --------------------------------------------------------
    prot_smiles = protonate_smiles(
        smiles, ph=cfg.protonate_ph, smarts_file=cfg.smarts_file,
    ) if cfg.protonate else smiles
    if cfg.protonate and cfg.smarts_file and not Path(cfg.smarts_file).is_file():
        log["warnings"].append(
            f"smarts_file not found ({cfg.smarts_file}); used default dimorphite SMARTS"
        )

    mw = molecular_weight(prot_smiles)
    mw_penalty = max(0.0, (mw - cfg.mw_threshold) / cfg.mw_penalty_divisor)
    pos_charges = positive_charge_count(prot_smiles)
    allow_sb = pos_charges <= cfg.max_positive_charges

    ligand_pdbqt = out_dir / "ligand.pdbqt"
    ok, msg = prepare_ligand_pdbqt(prot_smiles, ligand_pdbqt)
    if not ok:
        return _fail(smiles, prot_smiles, f"ligand prep failed: {msg}",
                     out_dir, log, mw, mw_penalty, pos_charges, allow_sb)

    charges = parse_charges(ligand_pdbqt)

    # 3. Dock (sample poses across replicas) --------------------------------
    poses = sample_poses(
        exec_path=vina_exec, receptor=receptor, ligand=ligand_pdbqt,
        workdir=out_dir, box_center=cfg.box_center, box_size=cfg.box_size,
        exhaustiveness=exhaustiveness, num_modes=num_modes,
        n_replicas=n_replicas, cpu_per_replica=cpu_per_replica,
        thresh_for_fail=cfg.thresh_for_fail,
    )
    if not poses:
        return _fail(smiles, prot_smiles, "docking produced no valid poses",
                     out_dir, log, mw, mw_penalty, pos_charges, allow_sb)

    # 4. Select qualifying poses (within energy window of the best) ---------
    best_raw = poses[0].vina_score
    qualifying = [p for p in poses if p.vina_score <= best_raw + cfg.pose_energy_window]

    # 5. PLIP interaction analysis (parallel) -------------------------------
    plip_args = [(str(clean_pdb), str(p.pdbqt_path), p.model_idx, charges)
                 for p in qualifying]
    nproc = min(plip_nprocs, len(plip_args))
    if nproc <= 1:
        plip_results = [plip_worker(a) for a in plip_args]
    else:
        with mp.Pool(nproc) as pool:
            plip_results = pool.map(plip_worker, plip_args)

    # 6. Composite per pose -------------------------------------------------
    scored: List[PoseScore] = []
    for pose, (interactions, sb_dists) in zip(qualifying, plip_results):
        bonus, breakdown = compute_pose_bonus(
            interactions, cfg, sb_distances=sb_dists, allow_saltbridge=allow_sb,
        )
        scored.append(PoseScore(
            pose=pose, bonus=bonus, composite=pose.vina_score - bonus,
            breakdown=breakdown,
            interactions={r: sorted(t) for r, t in interactions.items()},
            sb_distances=sb_dists,
        ))
    scored.sort(key=lambda s: s.composite)
    best = scored[0]

    # 7. Penalised final score (reporting; constant across poses) -----------
    composite_with_penalty = best.composite + mw_penalty
    if cfg.ion_penalty > 0 and pos_charges > cfg.max_positive_charges:
        composite_with_penalty += cfg.ion_penalty
    final_score = -composite_with_penalty

    # 8. Write best pose SDF + log -----------------------------------------
    sdf_path = out_dir / "best_pose.sdf"
    sb237 = best.sb_distances.get(237)
    props = {
        "SMILES": smiles,
        "protonated_SMILES": prot_smiles,
        "vina_affinity_kcal_mol": f"{best.pose.vina_score:.3f}",
        "interaction_bonus": f"{best.bonus:.3f}",
        "composite_score": f"{best.composite:.3f}",
        "final_score": f"{final_score:.3f}",
        "salt_bridge_237": "yes" if 237 in best.interactions
                           and "saltbridge" in best.interactions[237] else "no",
        "salt_bridge_237_dist": f"{sb237:.2f}" if sb237 is not None else "NA",
        "molecular_weight": f"{mw:.2f}",
        "mw_penalty": f"{mw_penalty:.3f}",
    }
    sdf_ok = pose_to_sdf(best.pose, sdf_path, properties=props)
    if not sdf_ok:
        log["warnings"].append("failed to write best_pose.sdf")

    result = DockingResult(
        smiles=smiles, protonated_smiles=prot_smiles, success=True, message="ok",
        best=best, scored_poses=scored, n_poses_total=len(poses),
        n_poses_qualifying=len(qualifying), mw=mw, mw_penalty=mw_penalty,
        positive_charges=pos_charges, allow_saltbridge=allow_sb,
        final_score=final_score,
        sdf_path=sdf_path if sdf_ok else None, log_path=out_dir / "docking_log.json",
    )
    _write_log(result, cfg, log, out_dir)
    return result


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _fail(smiles, prot_smiles, message, out_dir, log, mw, mw_penalty,
          pos_charges, allow_sb) -> DockingResult:
    result = DockingResult(
        smiles=smiles, protonated_smiles=prot_smiles, success=False, message=message,
        best=None, scored_poses=[], n_poses_total=0, n_poses_qualifying=0,
        mw=mw, mw_penalty=mw_penalty, positive_charges=pos_charges,
        allow_saltbridge=allow_sb, final_score=float("nan"),
        sdf_path=None, log_path=Path(out_dir) / "docking_log.json",
    )
    _write_log(result, None, log, Path(out_dir))
    return result


def _itype_abbrev(itype: str) -> str:
    return {"saltbridge": "sb", "hbond": "hb", "hydrophobic": "hp"}.get(itype, itype[:2])


def _write_log(result: DockingResult, cfg: Optional[ScoringConfig], log: dict, out_dir: Path):
    """Write docking_log.json (full) + docking_log.txt (human) + poses.csv."""
    poses_table = []
    for s in result.scored_poses:
        row = {
            "pose_id": s.pose.pose_id,
            "replica": s.pose.replica,
            "model_idx": s.pose.model_idx,
            "vina_affinity": round(s.pose.vina_score, 3),
            "interaction_bonus": round(s.bonus, 3),
            "composite": round(s.composite, 3),
        }
        for (resnr, itype), val in sorted(s.breakdown.items()):
            row[f"{_itype_abbrev(itype)}_{resnr}"] = round(val, 3)
        row["interactions"] = {r: t for r, t in sorted(s.interactions.items())}
        poses_table.append(row)

    payload = {
        "smiles": result.smiles,
        "protonated_smiles": result.protonated_smiles,
        "success": result.success,
        "message": result.message,
        "n_poses_total": result.n_poses_total,
        "n_poses_qualifying": result.n_poses_qualifying,
        "molecular_weight": round(result.mw, 3) if not math.isnan(result.mw) else None,
        "mw_penalty": round(result.mw_penalty, 3),
        "positive_charges": result.positive_charges,
        "saltbridge_allowed": result.allow_saltbridge,
        "warnings": log.get("warnings", []),
    }
    if result.best is not None:
        b = result.best
        payload["best_pose"] = {
            "pose_id": b.pose.pose_id,
            "vina_affinity": round(b.pose.vina_score, 3),
            "interaction_bonus": round(b.bonus, 3),
            "composite": round(b.composite, 3),
            "final_score": round(result.final_score, 3),
            "bonus_breakdown": {f"{_itype_abbrev(it)}_{r}": round(v, 3)
                                for (r, it), v in sorted(b.breakdown.items())},
            "salt_bridge_237_dist": round(b.sb_distances[237], 3)
                                    if 237 in b.sb_distances else None,
            "interactions": {r: t for r, t in sorted(b.interactions.items())},
        }
    if cfg is not None:
        payload["scoring_config"] = {
            "box_center": list(cfg.box_center),
            "box_size": list(cfg.box_size),
            "interaction_bonuses": {str(r): v for r, v in cfg.interaction_bonuses.items()},
            "sb_dist_best": cfg.sb_dist_best,
            "sb_dist_max": cfg.sb_dist_max,
            "pose_energy_window": cfg.pose_energy_window,
            "protonate": cfg.protonate,
            "protonate_ph": cfg.protonate_ph,
            "smarts_file": cfg.smarts_file,
        }
    payload["all_scored_poses"] = poses_table

    (out_dir / "docking_log.json").write_text(json.dumps(payload, indent=2))

    # Human-readable summary
    lines = [
        "GID4 docking log",
        "=" * 60,
        f"SMILES (input)      : {result.smiles}",
        f"SMILES (protonated) : {result.protonated_smiles}",
        f"Success             : {result.success}  ({result.message})",
        f"Poses sampled       : {result.n_poses_total} "
        f"({result.n_poses_qualifying} within energy window)",
        f"MW / penalty        : {result.mw:.2f} / {result.mw_penalty:.3f}",
        f"Positive charges    : {result.positive_charges} "
        f"(salt bridge {'allowed' if result.allow_saltbridge else 'DISALLOWED'})",
    ]
    if result.best is not None:
        b = result.best
        lines += [
            "-" * 60,
            "BEST POSE",
            f"  pose id           : {b.pose.pose_id} "
            f"(replica {b.pose.replica}, model {b.pose.model_idx})",
            f"  vina affinity     : {b.pose.vina_score:.3f} kcal/mol",
            f"  interaction bonus : {b.bonus:.3f}",
            f"  composite         : {b.composite:.3f}",
            f"  final score       : {result.final_score:.3f}  (higher = better)",
            "  bonus breakdown   :",
        ]
        if b.breakdown:
            for (resnr, itype), val in sorted(b.breakdown.items()):
                lines.append(f"      {itype:11s} res {resnr}: +{val:.3f}")
        else:
            lines.append("      (none)")
        if 237 in b.sb_distances:
            lines.append(f"  salt bridge 237   : {b.sb_distances[237]:.2f} A")
    if log.get("warnings"):
        lines += ["-" * 60, "WARNINGS"]
        lines += [f"  - {w}" for w in log["warnings"]]
    (out_dir / "docking_log.txt").write_text("\n".join(lines) + "\n")
