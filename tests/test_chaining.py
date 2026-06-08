"""Test du chaînage multi-vues sur une scène simulée (vérité terrain connue)."""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transforms import make_transform, invert  # noqa: E402
from world_model import WorldModel              # noqa: E402


def pose(xyz, deg):
    return make_transform(Rotation.from_euler("xyz", deg, degrees=True).as_matrix(), xyz)


# Vérité terrain : poses MONDE des tags (référence = tag 0 à l'origine).
TRUTH = {
    0: np.eye(4),
    1: pose([1, 0, 0], [0, 0, 30]),
    2: pose([1, 1, 0], [0, 45, 0]),
    3: pose([2, 1, 0.5], [20, 0, 0]),
}


class Det:
    def __init__(self, tid, T):
        self.tag_id = tid
        self.T_cam_tag = T
        self.decision_margin = 50.0


def view(cam_pose, ids):
    """Ce que voit une caméra : T_cam_tag = inv(T_world_cam) @ T_world_tag."""
    return [Det(i, invert(cam_pose) @ TRUTH[i]) for i in ids]


def test_chaining_exact():
    w = WorldModel(reference_id=0)
    # Vue A : ref(0)+tag1 | Vue B : tag1+tag2 | Vue C : tag2+tag3
    # -> le tag 3 n'est JAMAIS vu avec la référence, il est localisé par chaînage.
    w.add_frame(view(pose([0, 0, -3], [0, 0, 0]), [0, 1]))
    w.add_frame(view(pose([1, 0, -3], [0, 0, 0]), [1, 2]))
    w.add_frame(view(pose([2, 1, -3], [0, 0, 0]), [2, 3]))

    assert sorted(w.poses) == [0, 1, 2, 3]
    for tid, T_truth in TRUTH.items():
        err = np.linalg.norm(w.poses[tid][:3, 3] - T_truth[:3, 3])
        assert err < 1e-9, f"tag {tid}: erreur {err}"


if __name__ == "__main__":
    w = WorldModel(reference_id=0)
    w.add_frame(view(pose([0, 0, -3], [0, 0, 0]), [0, 1]))
    w.add_frame(view(pose([1, 0, -3], [0, 0, 0]), [1, 2]))
    w.add_frame(view(pose([2, 1, -3], [0, 0, 0]), [2, 3]))

    print("Tags localises:", sorted(w.poses))
    ok = True
    for tid, T_truth in TRUTH.items():
        err = np.linalg.norm(w.poses[tid][:3, 3] - T_truth[:3, 3])
        print(f"  tag {tid}: erreur position = {err * 1000:.6f} mm")
        ok = ok and err < 1e-9
    print("RESULTAT:", "OK - chainage exact (tag 3 via 0->1->2->3)" if ok else "ECHEC")
    sys.exit(0 if ok else 1)
