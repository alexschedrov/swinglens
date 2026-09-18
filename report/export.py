import csv
import dataclasses
import json
from pathlib import Path

from swing.state import SwingState

_CSV_FIELDS = [
    "frame_idx", "timestamp", "phase",
    "pelvis_rotation", "torso_rotation", "lead_arm_angle",
    "wrist_x", "wrist_y", "wrist_z", "head_x", "head_y", "head_z", "com_x", "com_y", "com_z",
    "angular_velocity_pelvis", "angular_velocity_torso", "angular_velocity_lead_arm",
]

def export_states(states: list[SwingState], output: str) -> None:
    """Serialize a per-frame state trajectory to JSON or CSV, format chosen by
    --output's extension."""
    path = Path(output)
    if path.suffix == ".json":
        _export_json(states, path)
    elif path.suffix == ".csv":
        _export_csv(states, path)
    else:
        raise ValueError(f"unsupported export format {path.suffix!r}: use .json or .csv")

def _export_json(states: list[SwingState], path: Path) -> None:
    path.write_text(json.dumps([dataclasses.asdict(s) for s in states], indent=2))

def _export_csv(states: list[SwingState], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for s in states:
            writer.writerow({
                "frame_idx": s.frame_idx,
                "timestamp": s.timestamp,
                "phase": s.phase,
                "pelvis_rotation": s.pelvis_rotation,
                "torso_rotation": s.torso_rotation,
                "lead_arm_angle": s.lead_arm_angle,
                "wrist_x": s.wrist_position[0] if s.wrist_position else None,
                "wrist_y": s.wrist_position[1] if s.wrist_position else None,
                "wrist_z": s.wrist_position[2] if s.wrist_position else None,
                "head_x": s.head_position[0] if s.head_position else None,
                "head_y": s.head_position[1] if s.head_position else None,
                "head_z": s.head_position[2] if s.head_position else None,
                "com_x": s.com_proxy[0] if s.com_proxy else None,
                "com_y": s.com_proxy[1] if s.com_proxy else None,
                "com_z": s.com_proxy[2] if s.com_proxy else None,
                "angular_velocity_pelvis": s.angular_velocities.pelvis,
                "angular_velocity_torso": s.angular_velocities.torso,
                "angular_velocity_lead_arm": s.angular_velocities.lead_arm,
            })
