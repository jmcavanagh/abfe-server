"""Load docking/scoring parameters from a SmileyLlama config YAML.

The pose-selection score used here is a faithful port of the per-pose
composite computed by ``GID4DockingScoreV3`` in the (private) smileyllama
package.  Rather than depend on that package, we read the *same* parameter
block out of its config YAML so the two stay in sync.

Only the ``scores.gid4_dock`` block matters for choosing a pose: the other
RL terms (iminer drug-likeness, logS) are per-molecule constants and do not
change between poses of one molecule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Protonation SMARTS shipped alongside this module. Used whenever the path in
# the config YAML (often a cluster-absolute path) is not present on disk.
BUNDLED_SMARTS = Path(__file__).parent / "hiqbind.smarts"


def resolve_smarts_file(smarts_file: Optional[str | Path]) -> Optional[str]:
    """Prefer an existing configured SMARTS path, else the bundled copy."""
    if smarts_file and Path(smarts_file).is_file():
        return str(smarts_file)
    if BUNDLED_SMARTS.is_file():
        return str(BUNDLED_SMARTS)
    return str(smarts_file) if smarts_file else None


# Same defaults as smileyllama.score.gid4_dock_v3.DEFAULT_INTERACTION_BONUSES,
# used only if the YAML omits the block.
DEFAULT_INTERACTION_BONUSES: Dict[int, Dict[str, float]] = {
    237: {"saltbridge": 3.0},
    132: {"hbond": 0.2},
    258: {"hbond": 0.2},
    171: {"hydrophobic": 0.2},
    240: {"hydrophobic": 0.2},
    254: {"hydrophobic": 0.2},
    273: {"hydrophobic": 0.2},
}


@dataclass
class ScoringConfig:
    """All knobs needed to dock against GID4 (9QDZ) and score poses."""

    box_center: Tuple[float, float, float] = (8.62, 1.56, 0.12)
    box_size: Tuple[float, float, float] = (30.0, 30.0, 30.0)

    # interaction_bonuses[resnr][interaction_type] = bonus (subtracted from the
    # docking energy, i.e. makes the composite better / more negative).
    interaction_bonuses: Dict[int, Dict[str, float]] = field(
        default_factory=lambda: {int(k): dict(v) for k, v in DEFAULT_INTERACTION_BONUSES.items()}
    )

    # Salt-bridge distance scaling: the full saltbridge bonus is awarded at
    # <= sb_dist_best, linearly fading to 0 at sb_dist_max. None disables scaling.
    sb_dist_best: Optional[float] = 2.5
    sb_dist_max: float = 5.5

    # A pose qualifies for PLIP analysis if its docking energy is within
    # pose_energy_window kcal/mol of the best pose for that molecule.
    pose_energy_window: float = 3.0

    # Molecular-weight penalty (per-molecule, constant across poses; reported
    # only). penalty = max(0, (MW - mw_threshold) / mw_penalty_divisor).
    mw_threshold: float = 400.0
    mw_penalty_divisor: float = 50.0

    # Salt bridges are only credited for molecules with <= this many positive
    # formal charges (an over-protonated cation gets no salt-bridge bonus).
    max_positive_charges: int = 1
    ion_penalty: float = 0.0

    # Protonation (dimorphite-dl) settings.
    protonate: bool = True
    protonate_ph: float = 7.4
    smarts_file: Optional[str] = None

    thresh_for_fail: float = -20.0

    @property
    def bonus_keys(self) -> List[Tuple[int, str]]:
        return sorted(
            (resnr, itype)
            for resnr, types in self.interaction_bonuses.items()
            for itype in types
        )


def load_scoring_config(
    yaml_path: Optional[str | Path] = None,
    protein_key: str = "9qdz",
    score_name: str = "gid4_dock",
) -> ScoringConfig:
    """Build a :class:`ScoringConfig` from a SmileyLlama config YAML.

    Parameters
    ----------
    yaml_path
        Path to a ``sl_config_*.yml``.  If None, returns built-in defaults
        (which match ``sl_config_gid4_v3_1b_3kcal_sbdist.yml``).
    protein_key
        Which box center to read (``9qdz`` or ``7slz``).
    score_name
        Key under ``scores:`` holding the docking score block.
    """
    cfg = ScoringConfig()

    if yaml_path is not None:
        with open(yaml_path) as f:
            doc = yaml.safe_load(f)
        params = doc.get("scores", {}).get(score_name, {}).get("parameters", {})

        center = params.get(f"box_center_{protein_key}")
        if center is not None:
            cfg.box_center = tuple(float(x) for x in center)
        if "box_size" in params:
            cfg.box_size = tuple(float(x) for x in params["box_size"])

        if "interaction_bonuses" in params:
            cfg.interaction_bonuses = {
                int(resnr): {str(it): float(v) for it, v in types.items()}
                for resnr, types in params["interaction_bonuses"].items()
            }

        for attr in (
            "sb_dist_best", "sb_dist_max", "pose_energy_window",
            "mw_threshold", "mw_penalty_divisor", "max_positive_charges",
            "ion_penalty", "protonate", "protonate_ph", "smarts_file",
            "thresh_for_fail",
        ):
            if attr in params:
                setattr(cfg, attr, params[attr])

    # Fall back to the bundled hiqbind.smarts when the configured path (often a
    # cluster-absolute path) is absent on this machine.
    cfg.smarts_file = resolve_smarts_file(cfg.smarts_file)
    return cfg
