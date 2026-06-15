"""Visualisation 3D temps réel du world model (Open3D).

Objectif : visualiser la carte de tags ET se localiser dans l'espace.
  - Chaque tag = un carré coloré dans son plan + un repère (X rouge, Y vert,
    Z bleu). Le tag de référence (origine du monde) est doré et plus marqué.
  - La caméra = un cône de visée (frustum) orange + un petit repère, qui montre
    où elle se trouve et dans quelle direction elle regarde.
  - Une trace bleue suit le déplacement de la caméra dans la carte.
  - Une grille de sol (plan du tag de référence) donne un repère d'échelle.

Si Open3D n'est pas installé, la visu est silencieusement désactivée.
"""

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False


# Palette épurée : couleur unique pour les tags, le tag de référence ressort.
_TAG_COLOR = [0.32, 0.62, 0.85]   # bleu doux : tags ordinaires
_REF_COLOR = [0.95, 0.60, 0.15]   # orange : tag de référence (origine)
_FROZEN_COLOR = [0.10, 0.85, 0.20]  # vert : tag verrouillé (ne bougera plus)
_CAM_COLOR = [0.35, 0.85, 0.45]   # vert : caméra
_TRAJ_COLOR = [0.85, 0.86, 0.92]  # blanc cassé : trajectoire


class Visualizer:
    def __init__(self, tag_size=0.1, reference_id=0, show_trajectory=True):
        self.enabled = _HAS_O3D
        self.tag_size = tag_size
        self.reference_id = reference_id
        self.show_trajectory = show_trajectory   # trace de la caméra affichée ?
        self._force_rebuild = False              # forcer un rebuild au prochain update
        if not self.enabled:
            print("[viz] open3d non installé : visualisation 3D désactivée.")
            return
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("AprilTag World Model", width=1024, height=768)
        # `q` (ou Échap) dans la fenêtre 3D demande l'arrêt du programme.
        self.alive = True
        self.vis.register_key_callback(ord("Q"), self._request_close)
        self.vis.register_key_callback(256, self._request_close)  # Échap (GLFW)
        self.vis.register_key_callback(ord("T"), self._toggle_trajectory)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.07, 0.08, 0.10])
        # Rendu SANS éclairage : couleurs "à plat", identiques sous tous les angles.
        # Évite que les faces arrière des trièdres apparaissent en noir et masquent
        # les tags (artefact d'éclairage qui "suit" l'objet quand on tourne la vue).
        opt.light_on = False
        opt.mesh_show_back_face = True   # repères visibles des deux côtés
        try:
            opt.line_width = 3.0         # contours/trajectoire plus lisibles
        except Exception:
            pass

        # Scène épurée : seul le petit repère du monde (origine) est statique.
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.tag_size * 0.8
        )
        self.vis.add_geometry(world_frame)

        # Navigation clavier déterministe (la souris d'Open3D est peu pratique).
        self._zoom = 0.7
        self._register_nav_keys()
        self._apply_initial_view()

        print("[viz 3D] Navigation CLAVIER :")
        print("   fleches : pivoter  |  W/S ou molette : zoom  |  I/J/K/L : deplacer")
        print("   F : vue d'ensemble  |  R : recentrer  |  T : trace on/off  |  Q : quitter")

        self._dynamic = []
        self._traj = []          # positions monde successives de la caméra
        self._frame = 0
        self._last_n = -1
        self._max_n = 0          # nb max de tags déjà vus (déclenche le re-cadrage)
        self._rebuild_every = 12  # reconstruire la scène au plus tous les N appels

    # -------------------------------------------------------------- navigation
    def _register_nav_keys(self):
        ROT, PAN = 12.0, 25.0  # incréments par appui
        binds = {
            265: self._nav_rotate(0, -ROT), 264: self._nav_rotate(0, ROT),   # ↑ ↓
            263: self._nav_rotate(-ROT, 0), 262: self._nav_rotate(ROT, 0),   # ← →
            ord("W"): self._nav_zoom(0.9), ord("S"): self._nav_zoom(1 / 0.9),
            ord("I"): self._nav_pan(0, PAN), ord("K"): self._nav_pan(0, -PAN),
            ord("J"): self._nav_pan(PAN, 0), ord("L"): self._nav_pan(-PAN, 0),
            ord("F"): self._nav_fit(),  # vue d'ensemble : cadre toute la scène
        }
        for key, cb in binds.items():
            self.vis.register_key_callback(key, cb)

    def _nav_fit(self):
        def cb(vis):
            vis.reset_view_point(True)         # ajuste pour tout voir
            vc = vis.get_view_control()
            vc.set_front([0.6, -0.6, 0.5])
            vc.set_up([0.0, 0.0, 1.0])
            return False
        return cb

    def _nav_rotate(self, dx, dy):
        def cb(vis):
            vis.get_view_control().rotate(dx, dy)
            return False
        return cb

    def _nav_zoom(self, factor):
        def cb(vis):
            self._zoom = min(2.0, max(0.02, self._zoom * factor))
            vis.get_view_control().set_zoom(self._zoom)
            return False
        return cb

    def _nav_pan(self, dx, dy):
        def cb(vis):
            vis.get_view_control().translate(dx, dy)
            return False
        return cb

    def _apply_initial_view(self):
        vc = self.vis.get_view_control()
        vc.set_lookat([0.0, 0.0, 0.0])      # centré sur le tag de référence
        vc.set_front([0.6, -0.6, 0.5])      # vue 3/4 (isométrique)
        vc.set_up([0.0, 0.0, 1.0])          # Z monde vers le haut
        vc.set_zoom(self._zoom)

    # ------------------------------------------------------------------ utils
    def _request_close(self, _vis):
        self.alive = False
        return False

    def _toggle_trajectory(self, _vis=None):
        """Affiche/masque la trace de la caméra (touche T)."""
        self.set_trajectory(not self.show_trajectory)
        return False

    def set_trajectory(self, on):
        """Active/désactive la trace (les points restent accumulés : réactiver
        réaffiche tout le tracé). Force un rebuild pour un effet immédiat."""
        if self.enabled and on != self.show_trajectory:
            self.show_trajectory = on
            self._force_rebuild = True

    def _color_for(self, tag_id, errors=None):
        if tag_id == self.reference_id:
            return _REF_COLOR
        # Coloration par erreur de reprojection : vert (<=1 px) -> rouge (>=5 px).
        if errors and tag_id in errors:
            x = min(1.0, max(0.0, (errors[tag_id] - 1.0) / 4.0))
            return [x, 1.0 - x, 0.15]          # RGB : vert = précis, rouge = douteux
        return _TAG_COLOR

    def _tag_corners(self, T):
        h = self.tag_size / 2.0
        verts = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)
        return (T[:3, :3] @ verts.T).T + T[:3, 3]

    def _tag_square(self, T, color):
        """Carré PLEIN à la taille réelle du tag (style Pupil Core). Sans éclairage
        + double-face, donc couleur unie visible des deux côtés (pas de face noire)."""
        verts = self._tag_corners(T)
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
        )
        m.paint_uniform_color(color)
        return m

    def _tag_outline(self, T, color):
        """Contour du carré, pour des bords nets par-dessus la surface pleine."""
        verts = self._tag_corners(T)
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        ls = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector2iVector(lines)
        )
        ls.paint_uniform_color([min(1.0, c * 1.4) for c in color])  # bord plus clair
        return ls

    def _camera_frustum(self, T):
        """Cône de visée : sommet = centre optique, base = plan image (+Z)."""
        d = self.tag_size * 1.5
        w, hh = d * 0.6, d * 0.45
        pts = np.array([
            [0, 0, 0], [-w, -hh, d], [w, -hh, d], [w, hh, d], [-w, hh, d],
        ], float)
        pts = (T[:3, :3] @ pts.T).T + T[:3, 3]
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        ls = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines)
        )
        ls.paint_uniform_color(_CAM_COLOR)
        return ls

    def _trajectory(self):
        pts = np.array(self._traj, float)
        lines = np.array([[i, i + 1] for i in range(len(pts) - 1)])
        ls = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines)
        )
        ls.paint_uniform_color(_TRAJ_COLOR)
        return ls

    # ----------------------------------------------------------------- update
    def update(self, poses, camera_pose=None, tag_errors=None, frozen=()):
        """Met à jour la scène 3D. `tag_errors` (id->px) colore les tags du vert
        (précis) au rouge (douteux) ; les tags `frozen` (verrouillés) ont un contour
        blanc. Retourne False si la fenêtre doit se fermer."""
        if not self.enabled:
            return True

        # Accumule la trajectoire (à chaque frame, si la caméra a bougé).
        if camera_pose is not None:
            p = camera_pose[:3, 3]
            if not self._traj or np.linalg.norm(p - self._traj[-1]) > self.tag_size * 0.04:
                self._traj.append(p.copy())
                if len(self._traj) > 3000:
                    self._traj.pop(0)

        # Reconstruire la géométrie est coûteux : on ne le fait que lorsqu'un
        # nouveau tag apparaît, ou périodiquement. Entre deux, on se contente de
        # rendre la scène pour garder la fenêtre fluide et réactive aux touches.
        self._frame += 1
        rebuild = (len(poses) != self._last_n
                   or self._frame % self._rebuild_every == 0
                   or self._force_rebuild)

        if rebuild:
            self._force_rebuild = False
            for g in self._dynamic:
                self.vis.remove_geometry(g, reset_bounding_box=False)
            self._dynamic = []

            # Tags : carré plein + contour. Couleur = qualité (vert->rouge), SAUF
            # les tags figés : toujours VERTS (verrouillés, ils ne bougeront plus),
            # avec un contour blanc pour les repérer.
            for tid, T in poses.items():
                if tid in frozen:
                    color = _FROZEN_COLOR
                else:
                    color = self._color_for(tid, tag_errors)
                square = self._tag_square(T, color)
                outline = self._tag_outline(T, color)
                if tid in frozen:
                    outline.paint_uniform_color([1.0, 1.0, 1.0])   # bordure = verrouillé
                for g in (square, outline):
                    self.vis.add_geometry(g, reset_bounding_box=False)
                    self._dynamic.append(g)

            # Caméra : uniquement le cône de visée (direction du regard).
            if camera_pose is not None:
                frustum = self._camera_frustum(camera_pose)
                self.vis.add_geometry(frustum, reset_bounding_box=False)
                self._dynamic.append(frustum)

            if self.show_trajectory and len(self._traj) >= 2:
                traj = self._trajectory()
                self.vis.add_geometry(traj, reset_bounding_box=False)
                self._dynamic.append(traj)

            # Vue d'ensemble automatique : re-cadre seulement sur une vraie
            # découverte (nouveau record de tags), pour toujours tout voir sans
            # faire sauter la vue sur les fluctuations. Touche F pour forcer.
            if len(poses) > self._max_n:
                self._max_n = len(poses)
                self.vis.reset_view_point(True)
                vc = self.vis.get_view_control()
                vc.set_front([0.6, -0.6, 0.5])
                vc.set_up([0.0, 0.0, 1.0])

            self._last_n = len(poses)

        # poll_events() renvoie False quand la fenêtre est fermée (bouton X).
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive and self.alive

    def close(self):
        if self.enabled:
            self.vis.destroy_window()
