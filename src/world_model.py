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

import numpy as np

from transforms import invert, average_transforms


class WorldModel:
    MAX_OBS_PER_EDGE = 200  # borne mémoire par arête

    def __init__(self, reference_id, min_observations=1, mapping=True):
        self.reference_id = reference_id
        self.min_observations = min_observations
        self.mapping = mapping           # True = on enrichit la carte ; False = figée
        self.edges = defaultdict(list)   # (i, j) -> liste de T_i_j observées
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
            for di in detections:
                for dj in detections:
                    if di.tag_id == dj.tag_id:
                        continue
                    key = (di.tag_id, dj.tag_id)
                    T_i_j = invert(di.T_cam_tag) @ dj.T_cam_tag
                    obs = self.edges[key]
                    obs.append(T_i_j)
                    if len(obs) > self.MAX_OBS_PER_EDGE:
                        obs.pop(0)
                    self._dirty.add(key)
            self.solve()
        self._update_camera_pose(detections)

    def _averaged_edges(self):
        # Ne recalcule la moyenne (SO(3), coûteuse) que pour les arêtes vues
        # dans l'image courante ; les autres sont servies depuis le cache.
        for key in self._dirty:
            self._avg_cache[key] = average_transforms(self.edges[key])
        self._dirty.clear()
        return {
            key: avg
            for key, avg in self._avg_cache.items()
            if len(self.edges[key]) >= self.min_observations
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
        """Estime T_world_cam à partir d'un tag déjà localisé dans le monde."""
        best = None
        for d in detections:
            if d.tag_id in self.poses:
                if best is None or d.decision_margin > best.decision_margin:
                    best = d
        if best is not None:
            # T_world_cam = T_world_tag @ T_tag_cam
            self.last_camera_pose = self.poses[best.tag_id] @ invert(best.T_cam_tag)

    def stats(self):
        return {
            "tags_localises": len(self.poses),
            "aretes": len(self.edges),
            "reference_id": self.reference_id,
            "reference_vue": self.reference_id in self.poses
            and any(self.reference_id in k for k in self.edges),
        }
