"""Graphe de poses : chaînage multi-vues des tags vers un repère de référence.

Idée :
  - Quand deux tags i et j sont vus dans la MÊME image, on connaît leur pose
    relative T_i_j = inv(T_cam_i) @ T_cam_j, INDÉPENDANTE de la pose caméra.
  - Chaque paire observée = une arête du graphe (noeuds = tags).
  - On propage depuis le tag de référence (BFS) pour obtenir la pose monde de
    chaque tag atteignable, même jamais vu en même temps que la référence.
  - Les observations répétées d'une arête sont moyennées (SO(3) + translation).
"""

from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from transforms import invert, make_transform

# Coins du tag dans SON repère (z=0), dans l'ordre renvoyé par pupil_apriltags
# (vérifié par reprojection). Unité : demi-côté (à multiplier par tag_size/2).
_TAG_CORNERS_UNIT = np.array(
    [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float64
)


class WorldModel:
    def __init__(self, reference_id, min_observations=1, mapping=True, tag_size=None):
        self.reference_id = reference_id
        self.min_observations = min_observations
        self.mapping = mapping           # True = on enrichit la carte ; False = figée
        self.tag_size = tag_size         # côté du tag (m), pour le PnP multi-tags
        self.camera_matrix = None        # K (3x3), fourni par main une fois connu
        # Moyennage des arêtes par accumulateurs CUMULATIFS (méthode de Markley) :
        # par arête on garde B = Σ q·qᵀ (4x4), la somme des translations et le
        # nombre d'observations. La rotation moyenne = vecteur propre dominant de
        # B. Tout est mis à jour en O(1)/observation et calculé PAR LOTS (un seul
        # appel scipy par image), ce qui évite l'explosion en N² × moyennage.
        self._edge_B = {}                # (i, j) -> 4x4 Σ q·qᵀ
        self._edge_tsum = {}             # (i, j) -> Σ translation
        self._edge_n = {}                # (i, j) -> nombre d'observations
        self.poses = {}                  # tag_id -> T_world_tag (4x4)
        self.last_camera_pose = None     # T_world_cam (4x4) pour la visu
        self._avg_cache = {}             # (i, j) -> moyenne mise en cache
        self._dirty = set()              # arêtes modifiées depuis le dernier solve

    def add_frame(self, detections):
        """Intègre une image.

        Si `mapping` est actif : enregistre les arêtes et recalcule la carte.
        Sinon : la carte est figée (aucun tag ajouté), on met seulement à jour
        la pose caméra par rapport aux tags déjà localisés (mode localisation).
        """
        if self.mapping:
            if len(detections) >= 2:
                self._integrate(detections)   # nouvelles arêtes (>= 2 tags vus)
            self.solve()                       # garde le tag de référence localisé
        self._update_camera_pose(detections)

    def _integrate(self, detections):
        """Ajoute toutes les poses relatives entre tags vus ensemble, par lots.

        Pour N tags : poses relatives T_i_j = inv(T_cam_i) @ T_cam_j calculées en
        une passe (einsum), conversion en quaternions en UN appel scipy, puis mise
        à jour des accumulateurs. Les rotations dégénérées (det<=0) sont ignorées.
        """
        ids = [d.tag_id for d in detections]
        N = len(ids)
        Ts = np.stack([np.asarray(d.T_cam_tag, dtype=np.float64) for d in detections])
        invT = np.stack([invert(T) for T in Ts])
        rel = np.einsum("iab,jbc->ijac", invT, Ts)        # (N, N, 4, 4)
        R = rel[:, :, :3, :3].reshape(N * N, 3, 3)
        t = rel[:, :, :3, 3].reshape(N * N, 3)

        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        good = (ii != jj).reshape(-1)                      # exclut i == j
        good &= np.isfinite(R).all(axis=(1, 2))
        good &= np.linalg.det(R) > 1e-6                    # exclut réflexions
        if not good.any():
            return

        quats = np.empty((N * N, 4))
        quats[good] = Rotation.from_matrix(R[good]).as_quat()   # un seul appel scipy
        qqt = np.einsum("ka,kb->kab", quats, quats)        # Σ q·qᵀ par observation
        for k in np.nonzero(good)[0]:
            key = (ids[k // N], ids[k % N])
            if key not in self._edge_B:
                self._edge_B[key] = np.zeros((4, 4))
                self._edge_tsum[key] = np.zeros(3)
                self._edge_n[key] = 0
            self._edge_B[key] += qqt[k]
            self._edge_tsum[key] += t[k]
            self._edge_n[key] += 1
            self._dirty.add(key)

    def _averaged_edges(self):
        # Recalcule la moyenne des seules arêtes modifiées, PAR LOTS : eigh
        # vectorisé sur les B + un seul appel scipy pour les quaternions.
        dirty = [k for k in self._dirty if self._edge_n.get(k, 0) > 0]
        if dirty:
            Bs = np.stack([self._edge_B[k] for k in dirty])     # (M, 4, 4)
            _, V = np.linalg.eigh(Bs)                           # symétriques
            Rs = Rotation.from_quat(V[:, :, -1]).as_matrix()    # (M, 3, 3)
            for k, Rk in zip(dirty, Rs):
                self._avg_cache[k] = make_transform(Rk, self._edge_tsum[k] / self._edge_n[k])
        self._dirty.clear()
        return {
            key: avg
            for key, avg in self._avg_cache.items()
            if self._edge_n.get(key, 0) >= self.min_observations
        }

    def solve(self):
        """Parcours BFS depuis la référence pour calculer toutes les poses monde."""
        avg = self._averaged_edges()
        adj = defaultdict(list)
        for (i, j) in avg:
            adj[i].append(j)

        poses = {self.reference_id: np.eye(4)}
        queue = deque([self.reference_id])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt in poses:
                    continue
                poses[nxt] = poses[cur] @ avg[(cur, nxt)]
                queue.append(nxt)

        self.poses = poses
        return poses

    def _update_camera_pose(self, detections):
        """Estime T_world_cam (pose caméra dans le repère du tag de référence).

        - >= 2 tags localisés visibles : PnP conjoint sur TOUS leurs coins
          (cv2.solvePnP) -> pose stable, sans ambiguïté de retournement.
        - 1 seul : repli sur T_world_tag @ inv(T_cam_tag).
        - 0 : on conserve la dernière pose connue.
        """
        visible = [d for d in detections if d.tag_id in self.poses]
        if not visible:
            return

        if len(visible) >= 2 and self.camera_matrix is not None and self.tag_size:
            pose = self._pnp_camera_pose(visible)
            if pose is not None:
                self.last_camera_pose = pose
                return

        # Repli mono-tag : T_world_cam = T_world_tag @ T_tag_cam.
        best = max(visible, key=lambda d: d.decision_margin)
        self.last_camera_pose = self.poses[best.tag_id] @ invert(best.T_cam_tag)

    def _pnp_camera_pose(self, visible):
        """Pose caméra par PnP conjoint sur les coins de plusieurs tags mappés.
        Les coins image sont dans l'image dé-distordue (distorsion nulle)."""
        corners3d = _TAG_CORNERS_UNIT * (self.tag_size / 2.0)
        obj, img = [], []
        for d in visible:
            R, t = self.poses[d.tag_id][:3, :3], self.poses[d.tag_id][:3, 3]
            obj.append(corners3d @ R.T + t)        # coins du tag -> repère monde
            img.append(np.asarray(d.corners, dtype=np.float64))
        obj = np.vstack(obj)
        img = np.vstack(img)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, self.camera_matrix, None, flags=cv2.SOLVEPNP_SQPNP
            )
        except cv2.error:
            return None
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return invert(make_transform(R, tvec.ravel()))   # T_cam_world -> T_world_cam

    def stats(self):
        return {
            "tags_localises": len(self.poses),
            "aretes": len(self._edge_n),
            "reference_id": self.reference_id,
            "reference_vue": self.reference_id in self.poses
            and any(self.reference_id in k for k in self._edge_n),
        }
