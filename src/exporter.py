"""Sauvegarde / chargement du world model (JSON riche + CSV façon Pupil Labs)."""

import csv
import json

import numpy as np
from scipy.spatial.transform import Rotation

from transforms import transform_to_dict


def save(world_model, path):
    diag = world_model.diagnostics()
    shift = getattr(world_model, "origin_shift", None)
    tags = {}
    for tid, T in sorted(world_model.poses.items()):
        if shift is not None:
            T = T.copy()
            T[:3, 3] = T[:3, 3] + shift     # origine = centre ou coin du tag réf
        entry = transform_to_dict(T)
        entry["diagnostics"] = diag.get(tid, {})   # qualité / contraintes du tag
        tags[str(tid)] = entry
    data = {
        "reference_id": world_model.reference_id,
        "units": "meters",
        "frame": "origine = tag de référence (centre ou coin selon config tag.origin)",
        "tags": tags,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def save_csv(world_model, path):
    """Exporte le world model au format `head_pose_tracker_model.csv` (Pupil Labs) :
    une ligne par marqueur, rotation (VECTEUR de rotation, en degrés) + translation
    (en mm). Le tag de référence est à (0,0,0). Origine = centre (ou coin si configuré)."""
    shift = getattr(world_model, "origin_shift", None)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["marker_id", "rotation_x", "rotation_y", "rotation_z",
                    "translation_x", "translation_y", "translation_z"])
        for tid, T in sorted(world_model.poses.items()):
            rvec_deg = np.degrees(Rotation.from_matrix(T[:3, :3]).as_rotvec())
            t = T[:3, 3] + (shift if shift is not None else 0.0)
            t_mm = t * 1000.0                                   # mètres -> millimètres
            w.writerow([tid] + [f"{v:.6f}" for v in (*rvec_deg, *t_mm)])
    return path


def load(path):
    """Recharge un world model exporté -> {tag_id: np.array 4x4}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        int(tid): np.array(entry["matrix"])
        for tid, entry in data["tags"].items()
    }
