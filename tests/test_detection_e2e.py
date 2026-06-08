"""Test end-to-end de la détection + estimation de pose sur une image synthétique.

Génère un vrai marqueur tag36h11 (via OpenCV ArUco), le place fronto-parallèle
à une distance connue, puis vérifie que TagDetector le détecte et retrouve la pose.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import TagDetector  # noqa: E402


def build_scene():
    # --- Caméra synthétique ---
    W, H = 1280, 1024
    fx = fy = 1000.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # --- Marqueur tag36h11 id=7 ---
    tag_id = 7
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker_px = 400
    marker = cv2.aruco.generateImageMarker(aruco_dict, tag_id, marker_px)
    # Zone blanche (quiet zone) indispensable à la détection.
    quiet = marker_px // 4
    marker = cv2.copyMakeBorder(
        marker, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255
    )

    # --- Image : fond blanc + marqueur centré ---
    img = np.full((H, W), 255, dtype=np.uint8)
    mh, mw = marker.shape
    y0, x0 = (H - mh) // 2, (W - mw) // 2
    img[y0:y0 + mh, x0:x0 + mw] = marker

    # Taille physique : on choisit 0.10 m pour le bord noir (marker_px pixels).
    tag_size = 0.10
    # Distance théorique : le bord noir fait marker_px px à fx => Z = fx * size / px.
    expected_z = fx * tag_size / marker_px

    return img, K, (fx, fy, cx, cy), tag_id, tag_size, expected_z


def main():
    img, K, cam_params, tag_id, tag_size, expected_z = build_scene()
    detector = TagDetector("tag36h11", tag_size, cam_params)
    dets = detector.detect(img)

    print(f"Détections : {len(dets)}")
    if not dets:
        print("RESULTAT: ECHEC - aucun tag détecté")
        return 1

    d = dets[0]
    t = d.T_cam_tag[:3, 3]
    print(f"  id détecté        : {d.tag_id} (attendu {tag_id})")
    print(f"  translation (m)   : x={t[0]:+.4f} y={t[1]:+.4f} z={t[2]:+.4f}")
    print(f"  Z attendu (m)     : {expected_z:.4f}")

    ok = (
        d.tag_id == tag_id
        and abs(t[0]) < 0.01            # tag centré -> x ~ 0
        and abs(t[1]) < 0.01            # tag centré -> y ~ 0
        and abs(t[2] - expected_z) < 0.02 * expected_z  # profondeur à 2 %
    )
    print("RESULTAT:", "OK - detection + pose correctes" if ok else "ECHEC - pose hors tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
