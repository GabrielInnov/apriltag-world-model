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
from exporter import save, save_csv       # noqa: E402


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

    def render(self, video, mapping_on, viz_open, trace_on):
        """Renvoie l'image composite [vidéo | panneau de boutons]. Le panneau est
        mis en cache et n'est redessiné que lorsque son état change."""
        h, w = video.shape[:2]
        key = (h, w, mapping_on, viz_open, trace_on)
        if key != self._key:
            self._panel, self._rects = self._build_panel(h, w, mapping_on, viz_open, trace_on)
            self._key = key
        return np.hstack([video, self._panel])

    def _build_panel(self, h, w, mapping_on, viz_open, trace_on):
        panel = np.full((h, self.PANEL_W, 3), 38, np.uint8)
        items = [
            ("save", "Enregistrer", (60, 60, 60)),
            ("calib", f"Calib: {'ON' if mapping_on else 'OFF'}",
             (0, 110, 0) if mapping_on else (70, 70, 70)),
            ("viz", "Masquer 3D" if viz_open else "Afficher vue 3D", (95, 70, 0)),
            ("trace", f"Trace: {'ON' if trace_on else 'OFF'}",
             (90, 60, 60) if trace_on else (60, 60, 60)),
            ("refine", "Affiner (BA)", (0, 90, 90)),
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


def run_bundle_adjustment(world):
    """Lance le bundle adjustment si possible (assez d'images clés) et affiche le
    gain. Appelé automatiquement avant chaque sauvegarde si `auto_refine`."""
    if len(world.keyframes) < 3:
        return
    print(f"[BA] optimisation ({len(world.keyframes)} images cles)...")
    res = world.refine()
    if res:
        print(f"[BA] reproj {res[0]:.2f} -> {res[1]:.2f} px")


def print_diagnostics(world, worst=8):
    """Résumé console du diagnostic par tag, les plus FAIBLES d'abord (erreur de
    reprojection élevée / peu d'angles), pour repérer ceux à re-filmer."""
    diag = world.diagnostics()
    if not diag:
        return
    rows = sorted(diag.items(),
                  key=lambda kv: (kv[1]["reproj_px"] is None, -(kv[1]["reproj_px"] or 0)))
    print(f"[diagnostic] {len(diag)} tags | "
          f"{sum(1 for d in diag.values() if d['frozen'])} figés "
          f"| tags les plus faibles :")
    print("   id |  reproj | bord | parallax(Z) | obs | vues | sauts | voisins | figé")
    for tid, d in rows[:worst]:
        rp = f"{d['reproj_px']:.2f}px" if d["reproj_px"] is not None else "  -  "
        br = f"{d['img_radius']:.2f}" if d["img_radius"] is not None else "  - "
        px = d.get("parallax_deg", 0.0)
        flag = "!Z" if px < 10 else ("~Z" if px < 25 else "okZ")   # fiabilité de la profondeur
        print(f"  {tid:3d} | {rp:>7} | {br:>4} | {px:5.1f}deg {flag:>3} | "
              f"{d['observations']:3d} | {d['viewpoints']:4d} | {str(d['hops_to_ref']):>5} "
              f"| {d['neighbors']:7d} | {'oui' if d['frozen'] else 'non'}")


def select_intrinsics(cam_cfg, kind):
    """Choisit le fichier de calibration selon la caméra réellement ouverte
    (intrinsics_daheng / intrinsics_webcam / ... sinon repli sur `intrinsics`)."""
    return cam_cfg.get(f"intrinsics_{kind}") or cam_cfg["intrinsics"]


def main():
    cfg = load_yaml(os.path.join(HERE, "config.yaml"))

    # On ouvre la caméra D'ABORD, puis on charge LES intrinsèques correspondantes.
    camera = create_camera(cfg["camera"])
    intr_path = select_intrinsics(cfg["camera"], getattr(camera, "kind", None))
    print(f"[intrinsics] caméra '{getattr(camera, 'kind', '?')}' -> {intr_path}")
    K, dist, cam_params, calib_width = load_intrinsics(os.path.join(HERE, intr_path))

    detector = TagDetector(
        cfg["tag"]["family"], cfg["tag"]["size_m"], cam_params, cfg.get("detection")
    )
    reference_id = cfg["tag"]["reference_id"]
    mp = cfg["mapping"]
    min_obs = mp.get("min_observations", 1)
    mapping_on = mp.get("enabled", True)
    auto_refine = mp.get("auto_refine", True)   # BA auto à la sauvegarde
    tag_size = cfg["tag"]["size_m"]
    # Critères de qualité (filtre + confirmation + pondération) -> pose fiable.
    quality = dict(
        min_margin=mp.get("min_decision_margin", 0.0),
        min_tag_px=mp.get("min_tag_pixels", 0.0),
        rot_coherence=mp.get("rot_coherence_min", 0.0),
        max_trans_std=mp.get("max_trans_std_m", float("inf")),
        refine_iters=mp.get("refine_iters", 0),
        freeze_enabled=mp.get("freeze_confident", False),
        freeze_px=mp.get("freeze_reproj_px", 1.0),
        freeze_min_obs=mp.get("freeze_min_obs", 15),
        freeze_min_views=mp.get("freeze_min_views", 3),
    )
    # Origine rapportée : centre (défaut) ou un coin du tag de référence.
    s = tag_size / 2.0
    corner = {"center": (0, 0, 0), "top_left": (-s, s, 0), "top_right": (s, s, 0),
              "bottom_left": (-s, -s, 0), "bottom_right": (s, -s, 0)}
    origin_shift = -np.array(corner.get(cfg["tag"].get("origin", "center"), (0, 0, 0)),
                             dtype=float)
    world = WorldModel(reference_id, min_obs, mapping=mapping_on,
                       tag_size=tag_size, **quality)
    world.origin_shift = origin_shift
    trace_on = cfg["visualization"].get("trace", True)  # trace caméra en 3D
    viz = None  # la vue 3D s'ouvre à la demande (bouton "Afficher vue 3D")
    if cfg["visualization"].get("autostart", False):
        viz = Visualizer(tag_size, reference_id, show_trajectory=trace_on, origin_shift=origin_shift)
    export_path = os.path.join(HERE, cfg["export"]["path"])
    csv_path = os.path.splitext(export_path)[0] + ".csv"   # format Pupil Labs

    print("Vue caméra : panneau de boutons à droite "
          "(Enregistrer / Calib / Afficher vue 3D / Reset / Expo / Quitter).")
    print("Touches équivalentes (fenêtre caméra au focus) : "
          "[q] quitter [s] save [r] reset [c] calib [v] vue 3D [t] trace [a] affiner [+/-] expo")
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
    ref_seen = False     # le tag de référence a-t-il déjà été vu ? (sinon : aucune carte)
    composite = None     # image affichée (vidéo + panneau), redessinée si besoin
    last_state = None    # état du menu -> ne redessine que s'il change
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
                # K (résolution de détection) -> nécessaire au PnP multi-tags.
                world.camera_matrix = K

            # Détection/intégration + rendu UNIQUEMENT sur une image neuve (sinon on
            # réutilise `base` : la boucle reste fluide et réactive aux entrées).
            new_frame = ver != last_ver
            if new_frame:
                last_ver = ver
                gray = cv2.remap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                 det_map1, det_map2, cv2.INTER_LINEAR)
                detections = detector.detect(gray)
                world.add_frame(detections)
                if any(d.tag_id == reference_id for d in detections):
                    ref_seen = True
                base = cv2.remap(frame, disp_map1, disp_map2, cv2.INTER_LINEAR)
                draw_overlay(base, detections, K_disp, tag_size,
                             reference_id, disp_scale, localized=world.poses)
                status = f"tags localises: {len(world.poses)} | vus: {len(detections)}"
                errs = [world.tag_error[t] for t in world.poses if t in world.tag_error]
                if errs:
                    status += f" | reproj moy: {sum(errs) / len(errs):.2f} px"
                status += f" | img cles: {len(world.keyframes)}"
                if world.frozen:
                    status += f" | figes: {len(world.frozen)}"
                if exp_us:
                    status += f" | expo: {exp_us:.0f} us"
                cv2.putText(base, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Avertissement : tag de référence jamais vu -> rien ne peut se localiser.
                if not ref_seen:
                    cv2.putText(base, f"!! tag REF {reference_id} jamais vu -> aucune carte",
                                (10, base.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                # Position de la caméra par rapport au tag de référence (origine).
                # last_camera_pose = T_world_cam ; sa translation = X, Y, Z (m).
                cp = world.last_camera_pose
                if cp is not None:
                    x, y, z = cp[:3, 3] + world.origin_shift   # origine = centre ou coin
                    d = float(np.linalg.norm((x, y, z)))
                    pos_txt = (f"cam/REF{reference_id} (m): "
                               f"X={x:+.3f} Y={y:+.3f} Z={z:+.3f}  d={d:.3f}")
                    pos_col = (0, 255, 255)
                else:
                    pos_txt = f"cam/REF{reference_id}: tag de reference non localise"
                    pos_col = (0, 165, 255)
                cv2.putText(base, pos_txt, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, pos_col, 2)

            # On ne reconstruit l'image composite (coûteux : copie + hstack) QUE si
            # une nouvelle image est arrivée ou si l'état du menu a changé. Sinon
            # la boucle ne ferait que brasser des allocations -> saturation mémoire.
            state = (world.mapping, viz is not None, trace_on)
            if new_frame or state != last_state or composite is None:
                last_state = state
                composite = menu.render(base, *state)
                cv2.imshow(window_name, composite)
            # La vue 3D est optionnelle : si elle est fermée (X / q), on la masque
            # simplement (on NE quitte PAS l'application).
            if viz is not None and not viz.update(world.poses, world.last_camera_pose,
                                                  world.tag_error, world.frozen):
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
                if auto_refine:
                    run_bundle_adjustment(world)
                save(world, export_path)
                save_csv(world, csv_path)
                print(f"[export] {export_path} + {os.path.basename(csv_path)} "
                      f"({len(world.poses)} tags)")
                print_diagnostics(world)
            if action == "reset" or key == ord("r"):
                world = WorldModel(reference_id, min_obs, mapping=world.mapping,
                                   tag_size=tag_size, **quality)
                world.camera_matrix = K
                world.origin_shift = origin_shift
                print("[reset] world model réinitialisé")
            if action == "calib" or key == ord("c"):
                world.mapping = not world.mapping
                print(f"[calibration] {'ON (ajout des tags)' if world.mapping else 'OFF (carte figee, localisation seule)'}")
            if action == "viz" or key == ord("v"):
                if viz is None:
                    viz = Visualizer(tag_size, reference_id, show_trajectory=trace_on, origin_shift=origin_shift)
                else:
                    viz.close()
                    viz = None
                    print("[viz 3D] fermée")
            if action == "trace" or key == ord("t"):
                trace_on = not trace_on
                if viz is not None:
                    viz.set_trajectory(trace_on)
                print(f"[trace] {'ON' if trace_on else 'OFF'}")
            if action == "refine" or key == ord("a"):
                print(f"[bundle adjustment] optimisation ({len(world.keyframes)} images cles)...")
                res = world.refine()
                if res:
                    print(f"[bundle adjustment] reproj {res[0]:.2f} -> {res[1]:.2f} px")
                else:
                    print("[bundle adjustment] pas assez de donnees "
                          "(filme plusieurs tags sous des angles varies)")
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
        if auto_refine:
            run_bundle_adjustment(world)
        save(world, export_path)
        save_csv(world, csv_path)
        print(f"[export final] {export_path} + {os.path.basename(csv_path)} "
              f"({len(world.poses)} tags)")
        print_diagnostics(world)
        camera.close()
        cv2.destroyAllWindows()
        if viz:
            viz.close()


if __name__ == "__main__":
    main()
