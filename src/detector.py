"""Détection d'AprilTags et estimation de pose caméra -> tag."""

from pupil_apriltags import Detector

from transforms import make_transform


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

    def detect(self, gray):
        """`gray` : image en niveaux de gris, déjà DÉ-DISTORDUE."""
        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )
        detections = []
        for r in results:
            T = make_transform(r.pose_R, r.pose_t)
            detections.append(
                Detection(r.tag_id, T, r.corners, r.center, r.decision_margin)
            )
        return detections
