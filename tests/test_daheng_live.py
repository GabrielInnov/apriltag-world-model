"""Test matériel headless : capture Daheng -> dé-distorsion -> détection AprilTag.

Aucune fenêtre : capture quelques images, affiche la résolution réelle et les
tags détectés. À lancer avec la Daheng branchée et un (ou plusieurs) tag(s)
tag36h11 dans le champ.
"""

import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from camera import create_camera   # noqa: E402
from detector import TagDetector   # noqa: E402


def load_yaml(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml(os.path.join(ROOT, "config.yaml"))
    intr = load_yaml(os.path.join(ROOT, cfg["camera"]["intrinsics"]))
    cm = intr["camera_matrix"]
    fx, fy, cx, cy = cm["fx"], cm["fy"], cm["cx"], cm["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr["distortion"], dtype=np.float64)

    # Force la Daheng (pas de fallback webcam) pour ce test matériel.
    cam_cfg = dict(cfg["camera"])
    cam_cfg["source"] = "daheng"
    camera = create_camera(cam_cfg)

    detector = TagDetector(cfg["tag"]["family"], cfg["tag"]["size_m"], (fx, fy, cx, cy))

    try:
        frame = None
        # On lit plusieurs images pour laisser l'auto-exposition converger.
        for i in range(30):
            f = camera.read()
            if f is not None:
                frame = f
        if frame is None:
            print("RESULTAT: ECHEC - aucune image reçue de la Daheng")
            return 1

        h, w = frame.shape[:2]
        print(f"Image capturée : {w} x {h} px")
        print(f"Calibration cx,cy = {cx:.1f},{cy:.1f}  (centre image = {w/2:.1f},{h/2:.1f})")
        if abs(cx - w / 2) > 0.15 * w or abs(cy - h / 2) > 0.15 * h:
            print("  ⚠️ centre optique éloigné du centre image : vérifie image_width/height "
                  "et que la résolution de capture = celle de la calibration.")

        undist = cv2.undistort(frame, K, dist)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray)

        print(f"Tags détectés : {len(dets)}")
        for d in sorted(dets, key=lambda x: x.tag_id):
            t = d.T_cam_tag[:3, 3]
            print(f"  id={d.tag_id:<4} dist={np.linalg.norm(t):.3f} m  "
                  f"(x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f})  marge={d.decision_margin:.0f}")

        # Sauve une image annotée pour inspection visuelle hors-ligne.
        out = os.path.join(ROOT, "daheng_capture.png")
        cv2.imwrite(out, undist)
        print(f"Image dé-distordue sauvegardée : {out}")

        print("RESULTAT:", "OK - capture Daheng + detection fonctionnelles"
              if dets else "CAPTURE OK mais AUCUN TAG détecté (montre un tag36h11)")
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    sys.exit(main())
