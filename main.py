"""Boucle principale : capture -> détection -> chaînage -> visualisation/export.

Pilotage par le panneau de boutons à droite de la vue caméra, ou au clavier
(fenêtre caméra au focus) :
    q  quitter (sauvegarde automatique)   s  sauvegarder maintenant
    r  réinitialiser la carte             c  calibration on/off
    v  afficher/masquer la vue 3D         +/-  exposition
"""

import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from camera import create_camera          # noqa: E402
from detector import TagDetector          # noqa: E402
from world_model import WorldModel        # noqa: E402
from visualizer import Visualizer         # noqa: E402
from exporter import save                 # noqa: E402


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_intrinsics(path):
    data = load_yaml(path)
    cm = data["camera_matrix"]
    fx, fy, cx, cy = cm["fx"], cm["fy"], cm["cx"], cm["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(data.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    calib_width = data.get("image_width")  # résolution de calibration (px)
    return K, dist, (fx, fy, cx, cy), calib_width


def scale_intrinsics(K, scale):
    """Met à l'échelle les intrinsèques pour une résolution différente de la
    calibration (ex. binning). La distorsion (coords normalisées) est inchangée."""
    Ks = K.copy()
    Ks[0] *= scale  # fx, cx
    Ks[1] *= scale  # fy, cy
    fx, fy, cx, cy = Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]
    return Ks, (fx, fy, cx, cy)


def draw_overlay(frame, detections, K_disp, tag_size, reference_id, scale=1.0,
                 localized=()):
    """Dessine, sur l'image d'affichage (dé-distordue, échelle `scale`), le contour,
    l'id et les axes 3D de chaque tag.

    `K_disp` : intrinsèques à l'échelle d'affichage (projection -> pixels affichés).
    Les coins/centre proviennent de la détection en résolution pleine : on les
    multiplie par `scale`.
    `localized` : ids des tags présents dans le world model. Code couleur :
        rouge = tag de référence · vert = localisé (dans la carte) · jaune = juste détecté.
    """
    half = tag_size / 2.0
    axes = np.float32([[0, 0, 0], [half, 0, 0], [0, half, 0], [0, 0, half]])
    no_dist = np.zeros(5)  # image déjà dé-distordue
    for d in detections:
        if d.tag_id == reference_id:
            color, tag = (0, 0, 255), "REF"          # rouge
        elif d.tag_id in localized:
            color, tag = (0, 220, 0), "MAP"           # vert : dans le world model
        else:
            color, tag = (0, 255, 255), "vu"          # jaune : détecté seulement

        corners = (d.corners * scale).astype(int)
        cv2.polylines(frame, [corners.reshape(-1, 1, 2)], True, color, 2)
        for c in corners:
            cv2.circle(frame, tuple(c), 4, color, -1)
        cx, cy = (d.center * scale).astype(int)
        cv2.putText(frame, f"{tag} id={d.tag_id}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rvec, _ = cv2.Rodrigues(d.T_cam_tag[:3, :3])
        tvec = d.T_cam_tag[:3, 3]
        pts, _ = cv2.projectPoints(axes, rvec, tvec, K_disp, no_dist)
        pts = pts.reshape(-1, 2).astype(int)
        o = tuple(pts[0])
        cv2.line(frame, o, tuple(pts[1]), (0, 0, 255), 2)   # X rouge
        cv2.line(frame, o, tuple(pts[2]), (0, 255, 0), 2)   # Y vert
        cv2.line(frame, o, tuple(pts[3]), (255, 0, 0), 2)   # Z bleu
    return frame


class Menu:
    """Panneau de boutons cliquables à DROITE de la vue caméra (ajouté à côté de
    la vidéo, sans en modifier la taille). Évite de dépendre du focus clavier."""

    PANEL_W = 210
    BTN_H = 46

    def __init__(self):
        self._click = None
        self._rects = {}    # id -> (x, y, w, h) en coordonnées de l'image composite
        self._panel = None  # panneau mis en cache (redessiné seulement si l'état change)
        self._key = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    def render(self, video, mapping_on, viz_open):
        """Renvoie l'image composite [vidéo | panneau de boutons]. Le panneau est
        mis en cache et n'est redessiné que lorsque son état change."""
        h, w = video.shape[:2]
        key = (h, w, mapping_on, viz_open)
        if key != self._key:
            self._panel, self._rects = self._build_panel(h, w, mapping_on, viz_open)
            self._key = key
        return np.hstack([video, self._panel])

    def _build_panel(self, h, w, mapping_on, viz_open):
        panel = np.full((h, self.PANEL_W, 3), 38, np.uint8)
        items = [
            ("save", "Enregistrer", (60, 60, 60)),
            ("calib", f"Calib: {'ON' if mapping_on else 'OFF'}",
             (0, 110, 0) if mapping_on else (70, 70, 70)),
            ("viz", "Masquer 3D" if viz_open else "Afficher vue 3D", (95, 70, 0)),
            ("reset", "Reset carte", (60, 60, 60)),
            ("exp_down", "Expo -", (60, 60, 60)),
            ("exp_up", "Expo +", (60, 60, 60)),
            ("quit", "Quitter", (0, 0, 130)),
        ]
        x0, bw, y = 12, self.PANEL_W - 24, 16
        rects = {}
        for bid, text, bg in items:
            cv2.rectangle(panel, (x0, y), (x0 + bw, y + self.BTN_H), bg, -1)
            cv2.rectangle(panel, (x0, y), (x0 + bw, y + self.BTN_H), (205, 205, 205), 1)
            cv2.putText(panel, text, (x0 + 10, y + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            rects[bid] = (w + x0, y, bw, self.BTN_H)  # décalé de la largeur vidéo
            y += self.BTN_H + 10
        return panel, rects

    def poll(self):
        """Renvoie l'id du bouton cliqué (et consomme le clic), sinon None."""
        if self._click is None:
            return None
        cx, cy = self._click
        self._click = None
        for bid, (x, y, w, h) in self._rects.items():
            if x <= cx <= x + w and y <= cy <= y + h:
                return bid
        return None


def main():
    cfg = load_yaml(os.path.join(HERE, "config.yaml"))
    K, dist, cam_params, calib_width = load_intrinsics(
        os.path.join(HERE, cfg["camera"]["intrinsics"])
    )

    camera = create_camera(cfg["camera"])
    detector = TagDetector(
        cfg["tag"]["family"], cfg["tag"]["size_m"], cam_params, cfg.get("detection")
    )
    reference_id = cfg["tag"]["reference_id"]
    min_obs = cfg["mapping"]["min_observations"]
    mapping_on = cfg["mapping"].get("enabled", True)
    world = WorldModel(reference_id, min_obs, mapping=mapping_on)
    tag_size = cfg["tag"]["size_m"]
    viz = None  # la vue 3D s'ouvre à la demande (bouton "Afficher vue 3D")
    if cfg["visualization"].get("autostart", False):
        viz = Visualizer(tag_size, reference_id)
    export_path = os.path.join(HERE, cfg["export"]["path"])

    print("Vue caméra : panneau de boutons à droite "
          "(Enregistrer / Calib / Afficher vue 3D / Reset / Expo / Quitter).")
    print("Touches équivalentes (fenêtre caméra au focus) : "
          "[q] quitter [s] save [r] reset [c] calib [v] vue 3D [+/-] expo")
    window_name = "AprilTag scan"
    menu = Menu()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, menu.on_mouse)
    # Cartes de dé-distorsion précalculées (cv2.undistort les recalcule à chaque
    # appel, très coûteux en 5 MP). Deux jeux :
    #   - détection : pleine résolution, appliqué au GRIS (1 canal => moins cher
    #     qu'un remap couleur), intrinsèques inchangées (poses valides).
    #   - affichage : dé-distorsion + downscale en une seule passe (couleur).
    det_map1 = det_map2 = None
    disp_map1 = disp_map2 = None
    disp_scale = 1.0
    K_disp = K
    last_ver = -1
    base = None          # vidéo + overlay + HUD, recalculée à chaque NOUVELLE image
    exp_us = camera.get_exposure()  # en cache : évite un appel SDK par image
    try:
        while True:
            frame, ver = camera.latest()
            if frame is None:
                if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
                    break
                continue

            if det_map1 is None:
                h, w = frame.shape[:2]
                # Ajuste les intrinsèques si la résolution diffère de la calibration
                # (binning, downscale...). La distorsion reste valable.
                if calib_width and abs(w / calib_width - 1.0) > 1e-3:
                    scale = w / calib_width
                    K, cam_params = scale_intrinsics(K, scale)
                    detector.camera_params = cam_params
                    print(f"[intrinsics] résolution {w}px vs calib {calib_width}px "
                          f"-> intrinsèques x{scale:.3f}")
                det_map1, det_map2 = cv2.initUndistortRectifyMap(
                    K, dist, None, K, (w, h), cv2.CV_16SC2
                )
                disp_scale = min(1.0, 1280.0 / w)  # fenêtre 2D allégée
                dw, dh = int(round(w * disp_scale)), int(round(h * disp_scale))
                K_disp = K * disp_scale
                K_disp[2, 2] = 1.0
                disp_map1, disp_map2 = cv2.initUndistortRectifyMap(
                    K, dist, None, K_disp, (dw, dh), cv2.CV_16SC2
                )

            # Détection/intégration + rendu UNIQUEMENT sur une image neuve (sinon on
            # réutilise `base` : la boucle reste fluide et réactive aux entrées).
            if ver != last_ver:
                last_ver = ver
                gray = cv2.remap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                 det_map1, det_map2, cv2.INTER_LINEAR)
                detections = detector.detect(gray)
                world.add_frame(detections)
                base = cv2.remap(frame, disp_map1, disp_map2, cv2.INTER_LINEAR)
                draw_overlay(base, detections, K_disp, tag_size,
                             reference_id, disp_scale, localized=world.poses)
                status = f"tags localises: {len(world.poses)} | vus: {len(detections)}"
                if exp_us:
                    status += f" | expo: {exp_us:.0f} us"
                cv2.putText(base, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            composite = menu.render(base, world.mapping, viz is not None)
            cv2.imshow(window_name, composite)
            # La vue 3D est optionnelle : si elle est fermée (X / q), on la masque
            # simplement (on NE quitte PAS l'application).
            if viz is not None and not viz.update(world.poses, world.last_camera_pose):
                viz.close()
                viz = None

            # Une action vient soit d'un clic sur le menu, soit du clavier.
            key = cv2.waitKey(1) & 0xFF
            action = menu.poll()
            if action == "quit" or key in (ord("q"), 27):  # bouton, q ou Échap
                break
            # Fermeture de la fenêtre OpenCV via le bouton X.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if action == "save" or key == ord("s"):
                save(world, export_path)
                print(f"[export] {export_path} ({len(world.poses)} tags)")
            if action == "reset" or key == ord("r"):
                world = WorldModel(reference_id, min_obs, mapping=world.mapping)
                print("[reset] world model réinitialisé")
            if action == "calib" or key == ord("c"):
                world.mapping = not world.mapping
                print(f"[calibration] {'ON (ajout des tags)' if world.mapping else 'OFF (carte figee, localisation seule)'}")
            if action == "viz" or key == ord("v"):
                if viz is None:
                    viz = Visualizer(tag_size, reference_id)  # ouvre la fenêtre 3D
                else:
                    viz.close()
                    viz = None
                    print("[viz 3D] fermée")
            # Réglage live de l'exposition (-20 % / +25 %).
            exp_delta = None
            if action == "exp_down" or key == ord("-"):
                exp_delta = 0.8
            elif action == "exp_up" or key in (ord("+"), ord("=")):
                exp_delta = 1.25
            if exp_delta is not None and exp_us:
                applied = camera.set_exposure(exp_us * exp_delta)
                if applied:
                    exp_us = applied
                    print(f"[expo] {applied:.0f} us")
                else:
                    print("[expo] non réglable")
    finally:
        save(world, export_path)
        print(f"[export final] {export_path} ({len(world.poses)} tags)")
        camera.close()
        cv2.destroyAllWindows()
        if viz:
            viz.close()


if __name__ == "__main__":
    main()
