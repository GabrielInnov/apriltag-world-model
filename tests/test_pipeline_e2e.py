"""Test de la chaîne complète : image 2 tags -> détection -> world model -> export JSON."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import TagDetector      # noqa: E402
from world_model import WorldModel    # noqa: E402
from exporter import save, load       # noqa: E402


def make_marker(aruco_dict, tag_id, px):
    m = cv2.aruco.generateImageMarker(aruco_dict, tag_id, px)
    q = px // 4
    return cv2.copyMakeBorder(m, q, q, q, q, cv2.BORDER_CONSTANT, value=255)


def main():
    W, H = 1600, 900
    fx = fy = 1200.0
    cam_params = (fx, fy, W / 2.0, H / 2.0)
    tag_size = 0.10
    marker_px = 240

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    ref_id, other_id = 0, 5
    m_ref = make_marker(aruco_dict, ref_id, marker_px)
    m_oth = make_marker(aruco_dict, other_id, marker_px)

    img = np.full((H, W), 255, dtype=np.uint8)
    mh, mw = m_ref.shape
    y = (H - mh) // 2
    # ref à gauche, l'autre à droite (séparation connue en pixels)
    x_ref, x_oth = 300, 900
    img[y:y + mh, x_ref:x_ref + mw] = m_ref
    img[y:y + mh, x_oth:x_oth + mw] = m_oth

    detector = TagDetector("tag36h11", tag_size, cam_params)
    dets = detector.detect(img)
    print(f"Tags détectés : {sorted(d.tag_id for d in dets)}")

    world = WorldModel(reference_id=ref_id)
    world.add_frame(dets)
    print(f"Tags localisés dans le monde : {sorted(world.poses)}")

    # Distance attendue entre les deux tags = écart des centres en px * (Z / fx)
    Z = fx * tag_size / marker_px
    px_centres = (x_oth + mw / 2) - (x_ref + mw / 2)
    expected_dx = px_centres * Z / fx
    measured_dx = np.linalg.norm(world.poses[other_id][:3, 3] - world.poses[ref_id][:3, 3])
    print(f"  distance ref->{other_id} mesurée : {measured_dx:.4f} m (attendue {expected_dx:.4f} m)")

    out = os.path.join(os.path.dirname(__file__), "..", "world_model_test.json")
    save(world, out)
    reloaded = load(out)
    print(f"  export/reload OK : {sorted(reloaded)} -> {os.path.basename(out)}")

    ok = (
        {d.tag_id for d in dets} == {ref_id, other_id}
        and sorted(world.poses) == [ref_id, other_id]
        and abs(measured_dx - expected_dx) < 0.02 * expected_dx
        and sorted(reloaded) == [ref_id, other_id]
    )
    print("RESULTAT:", "OK - chaine complete fonctionnelle" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
