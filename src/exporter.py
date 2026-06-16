"""Sauvegarde / chargement du world model en JSON."""

import json

import numpy as np

from transforms import transform_to_dict


def save(world_model, path):
    diag = world_model.diagnostics()
    tags = {}
    for tid, T in sorted(world_model.poses.items()):
        entry = transform_to_dict(T)
        entry["diagnostics"] = diag.get(tid, {})   # qualité / contraintes du tag
        tags[str(tid)] = entry
    data = {
        "reference_id": world_model.reference_id,
        "units": "meters",
        "frame": "le tag de référence est l'origine du monde",
        "tags": tags,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load(path):
    """Recharge un world model exporté -> {tag_id: np.array 4x4}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        int(tid): np.array(entry["matrix"])
        for tid, entry in data["tags"].items()
    }
