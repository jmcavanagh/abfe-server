"""GID4 docking + pose selection for the ABFE server.

Docks a SMILES to GID4 (9QDZ) with AutoDock Vina, samples many poses across
CPU replicas, scores each pose with a faithful port of the SmileyLlama
``GID4DockingScoreV3`` composite (Vina affinity minus PLIP-detected
salt-bridge / H-bond / hydrophobic interaction bonuses to the relevant
residues), and emits the best pose as an SDF for downstream ABFE.

This package depends only on the docking stack (rdkit, meeko, plip,
dimorphite-dl, vina/obabel) — *not* on the smileyllama package — so it can live
in its own conda environment, isolated from the ABFE/server code.
"""

from .config import ScoringConfig, load_scoring_config
from .pipeline import DockingResult, PoseScore, dock_smiles

__all__ = [
    "ScoringConfig",
    "load_scoring_config",
    "DockingResult",
    "PoseScore",
    "dock_smiles",
]
