"""Détection d'AprilTags et estimation de pose caméra -> tag.

Affinage de précision :
  - coins raffinés au SOUS-PIXEL (cv2.cornerSubPix) ;
  - MOYENNAGE TEMPOREL des coins quand la caméra est quasi immobile (bruit ÷√N) ;
  - pose recalculée depuis ces coins affinés (solvePnP IPPE_SQUARE) -> profite aussi
    aux arêtes / au PnP / au bundle adjustment, et choisit la meilleure des deux
    solutions (atténue l'ambiguïté de retournement).
"""

import cv2
import numpy as np
from pupil_apriltags import Detector

from transforms import make_transform

# Coins du tag dans son repère (z=0), ordre pupil_apriltags (= ordre attendu par
# SOLVEPNP_IPPE_SQUARE). Unité : demi-côté.
_OBJ_UNIT = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float64)
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)


class Detection:
    """Une détection de tag dans une image."""

    def __init__(self, tag_id, T_cam_tag, corners, center, decision_margin):
        self.tag_id = tag_id
        self.T_cam_tag = T_cam_tag          # pose 4x4 : repère caméra -> repère tag
        self.corners = corners              # 4x2 px
        self.center = center                # 2 px
        self.decision_margin = decision_margin  # qualité de la détection


class TagDetector:
    def __init__(self, family, tag_size, camera_params, det_cfg=None):
        det_cfg = det_cfg or {}
        self.detector = Detector(
            families=family,
            nthreads=det_cfg.get("nthreads", 4),
            quad_decimate=det_cfg.get("quad_decimate", 1.0),
            decode_sharpening=det_cfg.get("decode_sharpening", 0.25),
        )
        self.tag_size = tag_size
        self.camera_params = camera_params  # (fx, fy, cx, cy)
        self.subpixel = det_cfg.get("subpixel", True)
        self.temporal_avg = det_cfg.get("temporal_avg", True)
        self._alpha = 0.8        # poids de l'historique (moyennage temporel)
        self._motion_px = 2.0    # déplacement coin max (px) pour considérer "immobile"
        self._smooth = {}        # tag_id -> coins lissés (4x2)
        self._obj = _OBJ_UNIT * (tag_size / 2.0)   # coins 3D dans le repère du tag

    def _K(self):
        fx, fy, cx, cy = self.camera_params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def detect(self, gray):
        """`gray` : image en niveaux de gris, déjà DÉ-DISTORDUE."""
        # Pas d'estimation de pose par pupil : on la recalcule nous-mêmes (solvePnP),
        # ce qui évite ses messages "more than one new minima" et accélère.
        results = self.detector.detect(gray, estimate_tag_pose=False)
        K = self._K()
        h, w = gray.shape[:2]
        win = 5
        margin = win + 2                       # marge de sécurité pour cornerSubPix
        detections = []
        smooth_next = {}
        for r in results:
            corners = np.asarray(r.corners, dtype=np.float32)

            # Affinage sous-pixel — sauf si un coin est trop près du bord (sinon
            # la fenêtre de recherche sort de l'image -> erreur OpenCV).
            if self.subpixel and (corners[:, 0] > margin).all() \
                    and (corners[:, 0] < w - margin).all() \
                    and (corners[:, 1] > margin).all() \
                    and (corners[:, 1] < h - margin).all():
                refined = corners.reshape(-1, 1, 2).copy()
                try:
                    cv2.cornerSubPix(gray, refined, (win, win), (-1, -1), _SUBPIX_CRIT)
                    corners = refined.reshape(4, 2)
                except cv2.error:
                    pass

            if self.temporal_avg:                               # moyennage si quasi immobile
                prev = self._smooth.get(r.tag_id)
                if prev is not None and np.abs(corners - prev).max() < self._motion_px:
                    corners = self._alpha * prev + (1.0 - self._alpha) * corners
                smooth_next[r.tag_id] = corners

            # Pose depuis les coins affinés (IPPE_SQUARE = meilleure des 2 solutions).
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    self._obj, corners.astype(np.float64), K, None,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except cv2.error:
                ok = False
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            detections.append(
                Detection(r.tag_id, make_transform(R, tvec.ravel()),
                          corners, r.center, r.decision_margin)
            )

        self._smooth = smooth_next      # ne garde que les tags vus (reset si revu autrement)
        return detections
