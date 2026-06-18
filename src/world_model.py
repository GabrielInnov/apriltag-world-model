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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from transforms import invert, make_transform


def _pose_to_params(T):
    """Pose 4x4 -> 6 paramètres (rvec, tvec) pour l'optimisation."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return np.concatenate([rvec.ravel(), T[:3, 3]])


def _params_to_pose(p):
    R, _ = cv2.Rodrigues(p[:3])
    return make_transform(R, p[3:6])


def _angle_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))

# Coins du tag dans SON repère (z=0), dans l'ordre renvoyé par pupil_apriltags
# (vérifié par reprojection). Unité : demi-côté (à multiplier par tag_size/2).
_TAG_CORNERS_UNIT = np.array(
    [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float64
)


def _apparent_size(corners):
    """Taille apparente d'un tag = longueur moyenne de ses côtés (px), ou None
    si les coins sont absents/invalides (cas des détections synthétiques)."""
    if corners is None:
        return None
    c = np.asarray(corners, dtype=float)
    if c.shape != (4, 2) or not np.isfinite(c).all():
        return None
    return float(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean())


class WorldModel:
    def __init__(self, reference_id, min_observations=1, mapping=True, tag_size=None,
                 min_margin=0.0, min_tag_px=0.0, rot_coherence=0.0,
                 max_trans_std=float("inf"), refine_iters=0,
                 freeze_enabled=False, freeze_px=1.0, freeze_min_obs=15,
                 freeze_min_views=3):
        self.reference_id = reference_id
        self.min_observations = min_observations
        self.mapping = mapping           # True = on enrichit la carte ; False = figée
        self.tag_size = tag_size         # côté du tag (m), pour le PnP multi-tags
        self.camera_matrix = None        # K (3x3), fourni par main une fois connu
        self.refine_iters = refine_iters  # itérations de relaxation du graphe (0 = BFS pur)
        # Verrouillage des tags confiants : une fois un tag "vert" (faible erreur de
        # reprojection) et assez observé, sa pose est FIGÉE (ancre fixe). Évite qu'il
        # soit dégradé par des observations bruitées ultérieures.
        self.freeze_enabled = freeze_enabled
        self.freeze_px = freeze_px           # erreur de reprojection max pour figer (px)
        self.freeze_min_obs = freeze_min_obs # nb d'images min vu avant de figer
        self.freeze_min_views = freeze_min_views  # nb de points de vue distincts requis
        self.frozen = set()                  # tags dont la pose est verrouillée (définitif)
        self._tag_seen = defaultdict(int)    # tag_id -> nb d'images où il a été vu
        self._tag_radius = {}                # tag_id -> position image (0=centre, ~1=coin)
        # Diversité de points de vue PAR TAG (indépendante du plafond d'images clés
        # du BA) : on accumule les directions de visée distinctes de chaque tag.
        self._tag_bearings = defaultdict(list)  # tag_id -> [directions de visée]
        self._view_cos_thr = np.cos(np.radians(8.0))  # 8° entre deux vues distinctes
        # Critères de QUALITÉ (par défaut neutres -> aucun filtrage ; activés par
        # main via la config). Un tag n'est "confirmé" (ajouté à la carte) que si
        # une arête vers la carte est fiable : assez d'observations + cohérente.
        self.min_margin = min_margin         # decision_margin mini d'une détection
        self.min_tag_px = min_tag_px         # taille apparente mini (px)
        self.rot_coherence = rot_coherence   # cohérence rotationnelle mini (0..1)
        self.max_trans_std = max_trans_std   # écart-type max de translation (m)
        # Moyennage des arêtes par accumulateurs CUMULATIFS (méthode de Markley) :
        # par arête on garde B = Σ q·qᵀ (4x4), la somme des translations et le
        # nombre d'observations. La rotation moyenne = vecteur propre dominant de
        # B. Tout est mis à jour en O(1)/observation et calculé PAR LOTS (un seul
        # appel scipy par image), ce qui évite l'explosion en N² × moyennage.
        self._edge_B = {}                # (i, j) -> 4x4 Σ w·q·qᵀ (pondéré)
        self._edge_tsum = {}             # (i, j) -> Σ w·translation
        self._edge_t2 = {}               # (i, j) -> Σ w·|translation|²  (variance)
        self._edge_w = {}                # (i, j) -> Σ w  (somme des poids)
        self._edge_n = {}                # (i, j) -> nombre d'observations
        self.poses = {}                  # tag_id -> T_world_tag (4x4)
        self.tag_error = {}              # tag_id -> erreur de reprojection (px, lissée)
        self.last_camera_pose = None     # T_world_cam (4x4) pour la visu
        # Décalage de l'origine rapportée (sortie) : 0 = centre du tag de référence.
        # Pour mettre l'origine sur un coin, on ajoute ce vecteur aux translations.
        self.origin_shift = np.zeros(3)
        # Images clés pour le bundle adjustment (vues diverses mémorisées).
        self.keyframes = []              # [{cam: T_cam_world, obs: [(tag_id, corners)]}]
        self.max_keyframes = 30
        self._kf_trans = tag_size or 0.02   # seuil de diversité en translation (m)
        self._kf_rot = 8.0                  # seuil de diversité en rotation (deg)
        self._avg_cache = {}             # (i, j) -> moyenne mise en cache
        self._edge_trusted = {}          # (i, j) -> arête jugée fiable ?
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
        """Ajoute les poses relatives entre tags vus ensemble (par lots), en
        FILTRANT et PONDÉRANT par qualité.

        - Filtre : on ignore les détections de faible `decision_margin` ou de
          trop petite taille apparente (tag lointain/peu fiable).
        - Pondération : poids de l'observation = moyenne géométrique des tailles
          apparentes (les tags vus gros/près pèsent plus dans la moyenne).
        Les rotations dégénérées (det<=0) sont aussi ignorées.
        """
        ids = [d.tag_id for d in detections]
        N = len(ids)
        sizes = [_apparent_size(getattr(d, "corners", None)) for d in detections]
        margins = [float(getattr(d, "decision_margin", 0.0)) for d in detections]
        ok = np.array([
            mg >= self.min_margin and (sz is None or sz >= self.min_tag_px)
            for sz, mg in zip(sizes, margins)
        ])
        # Poids = taille apparente ; jamais nul (sinon division par 0 plus loin).
        wdet = np.array([sz if sz else 1.0 for sz in sizes], dtype=float)
        if ok.sum() < 2:
            return

        Ts = np.stack([np.asarray(d.T_cam_tag, dtype=np.float64) for d in detections])
        invT = np.stack([invert(T) for T in Ts])
        rel = np.einsum("iab,jbc->ijac", invT, Ts)        # (N, N, 4, 4)
        R = rel[:, :, :3, :3].reshape(N * N, 3, 3)
        t = rel[:, :, :3, 3].reshape(N * N, 3)

        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        iif, jjf = ii.reshape(-1), jj.reshape(-1)
        good = (iif != jjf) & ok[iif] & ok[jjf]            # paires de tags fiables
        good &= np.isfinite(R).all(axis=(1, 2))
        good &= np.linalg.det(R) > 1e-6                    # exclut réflexions
        if not good.any():
            return

        quats = np.empty((N * N, 4))
        quats[good] = Rotation.from_matrix(R[good]).as_quat()   # un seul appel scipy
        qqt = np.einsum("ka,kb->kab", quats, quats)
        wpair = np.sqrt(wdet[iif] * wdet[jjf])             # poids par paire
        for k in np.nonzero(good)[0]:
            key = (ids[iif[k]], ids[jjf[k]])
            w = wpair[k]
            if key not in self._edge_B:
                self._edge_B[key] = np.zeros((4, 4))
                self._edge_tsum[key] = np.zeros(3)
                self._edge_t2[key] = 0.0
                self._edge_w[key] = 0.0
                self._edge_n[key] = 0
            self._edge_B[key] += w * qqt[k]
            self._edge_tsum[key] += w * t[k]
            self._edge_t2[key] += w * float(t[k] @ t[k])
            self._edge_w[key] += w
            self._edge_n[key] += 1
            self._dirty.add(key)

    def _averaged_edges(self):
        # Moyenne des arêtes modifiées (par lots) + test de CONFIRMATION : une
        # arête n'est "fiable" que si elle a assez d'observations, une rotation
        # cohérente (vecteurs propres alignés) et une translation peu dispersée.
        dirty = [k for k in self._dirty if self._edge_n.get(k, 0) > 0]
        if dirty:
            Bs = np.stack([self._edge_B[k] for k in dirty])     # (M, 4, 4)
            evals, V = np.linalg.eigh(Bs)                       # symétriques
            Rs = Rotation.from_quat(V[:, :, -1]).as_matrix()    # (M, 3, 3)
            for idx, k in enumerate(dirty):
                ws = self._edge_w[k]
                if ws <= 0:
                    continue
                tmean = self._edge_tsum[k] / ws
                coherence = evals[idx, -1] / ws                 # 1 = parfaitement aligné
                var = max(0.0, self._edge_t2[k] / ws - float(tmean @ tmean))
                self._avg_cache[k] = make_transform(Rs[idx], tmean)
                self._edge_trusted[k] = (
                    self._edge_n[k] >= self.min_observations
                    and coherence >= self.rot_coherence
                    and var <= self.max_trans_std ** 2
                )
        self._dirty.clear()
        return {k: v for k, v in self._avg_cache.items()
                if self._edge_trusted.get(k, False)}

    def solve(self):
        """Calcule les poses monde des tags.

        1. BFS depuis la référence : connectivité + estimation initiale (warm-start
           sur les poses précédentes si dispo).
        2. Si `refine_iters > 0` : RELAXATION du graphe de poses utilisant TOUTES
           les arêtes fiables (fermeture de boucle) -> annule la dérive du BFS.
        """
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
                # warm-start : réutilise la pose précédente si elle existe.
                poses[nxt] = self.poses.get(nxt, poses[cur] @ avg[(cur, nxt)])
                queue.append(nxt)

        # Les tags figés gardent leur pose verrouillée (ancres fixes).
        for t in self.frozen:
            if t in self.poses:
                poses[t] = self.poses[t]

        if self.refine_iters > 0 and len(poses) > 2:
            poses = self._relax(poses, avg)

        self.poses = poses
        self._update_frozen()
        return poses

    def _update_frozen(self):
        """Verrouille les tags VRAIMENT bien contraints, et libère ceux qui se
        révèlent incohérents.

        Critères de gel (durcis pour éviter de figer un tag mal placé) :
          - vu assez souvent (`freeze_min_obs`) ;
          - vu sous PLUSIEURS points de vue distincts (`freeze_min_views`, compté
            par tag, sans limite globale) -> pose bien triangulée, retournement levé ;
          - faible erreur de reprojection (`freeze_px`).
        Un tag figé le reste DÉFINITIVEMENT (ancre fixe). C'est donc le critère de
        gel qui doit être sûr -> on ne fige que des tags vraiment bien contraints.
        """
        if not self.freeze_enabled:
            return
        for t in list(self.poses):
            if t == self.reference_id or t in self.frozen:
                continue
            if (self._tag_seen.get(t, 0) >= self.freeze_min_obs
                    and len(self._tag_bearings.get(t, ())) >= self.freeze_min_views
                    and self.tag_error.get(t, 1e9) <= self.freeze_px):
                self.frozen.add(t)

    def _relax(self, poses, avg):
        """Relaxation du graphe de poses (Jacobi pondéré, vectorisé).

        Chaque arête fiable (i, j) donne une estimation de la pose de j à partir
        de celle de i : T_world_j = T_world_i @ T_i_j. On fait la moyenne pondérée
        (rotation : Markley) de toutes ces estimations pour chaque tag, en itérant.
        La référence reste fixe. Utiliser TOUTES les arêtes = fermeture de boucle.
        """
        tags = list(poses.keys())
        idx = {t: k for k, t in enumerate(tags)}
        ref_k = idx[self.reference_id]
        # Indices à NE PAS bouger : référence + tags figés (ancres fixes).
        fixed = {ref_k} | {idx[t] for t in self.frozen if t in idx}
        edges = [(i, j) for (i, j) in avg
                 if i in idx and j in idx and j != self.reference_id]
        if not edges:
            return poses

        src = np.array([idx[i] for (i, j) in edges])
        dst = np.array([idx[j] for (i, j) in edges])
        Tedge = np.stack([avg[(i, j)] for (i, j) in edges])          # (E,4,4) = T_i_j
        w = np.array([self._edge_w[(i, j)] for (i, j) in edges])     # poids = fiabilité
        M = np.stack([poses[t] for t in tags]).astype(np.float64)    # (Nt,4,4)
        Nt = len(tags)

        for _ in range(self.refine_iters):
            est = M[src] @ Tedge                                     # (E,4,4) estim. de j
            q = Rotation.from_matrix(est[:, :3, :3]).as_quat()       # un appel scipy
            qqt = np.einsum("ea,eb->eab", q, q) * w[:, None, None]
            B = np.zeros((Nt, 4, 4)); np.add.at(B, dst, qqt)
            Ts = np.zeros((Nt, 3)); np.add.at(Ts, dst, est[:, :3, 3] * w[:, None])
            Ws = np.zeros(Nt); np.add.at(Ws, dst, w)
            upd = np.array([k for k in np.where(Ws > 0)[0] if k not in fixed])
            if len(upd) == 0:
                break
            _, V = np.linalg.eigh(B[upd])
            Rnew = Rotation.from_quat(V[:, :, -1]).as_matrix()       # un appel scipy
            tnew = Ts[upd] / Ws[upd, None]
            Mnew = M.copy()
            Mnew[upd, :3, :3] = Rnew
            Mnew[upd, :3, 3] = tnew
            move = np.linalg.norm(Mnew[upd, :3, 3] - M[upd, :3, 3], axis=1).max()
            M = Mnew
            if move < 1e-6:                                          # convergé
                break

        return {t: M[idx[t]] for t in tags}

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

        pose = None
        if len(visible) >= 2 and self.camera_matrix is not None and self.tag_size:
            pose = self._pnp_camera_pose(visible)
        if pose is None:
            # Repli mono-tag : T_world_cam = T_world_tag @ T_tag_cam.
            best = max(visible, key=lambda d: d.decision_margin)
            pose = self.poses[best.tag_id] @ invert(best.T_cam_tag)

        self.last_camera_pose = pose
        self._update_tag_errors(visible, pose)
        if self.mapping:
            self._maybe_keyframe(visible, pose)

    def _maybe_keyframe(self, visible, T_world_cam):
        """Mémorise une image clé si elle apporte un point de vue NOUVEAU (pour le
        bundle adjustment). On ne garde que des vues diverses, bornées en nombre."""
        if len(self.keyframes) >= self.max_keyframes:
            return
        obs = [(d.tag_id, np.asarray(d.corners, dtype=np.float64))
               for d in visible
               if getattr(d, "corners", None) is not None and d.tag_id in self.poses]
        if len(obs) < 2:
            return
        Tcw = invert(T_world_cam)
        for kf in self.keyframes:                      # vue trop proche d'une existante ?
            if (np.linalg.norm(Tcw[:3, 3] - kf["cam"][:3, 3]) < self._kf_trans
                    and _angle_deg(Tcw[:3, :3], kf["cam"][:3, :3]) < self._kf_rot):
                return
        self.keyframes.append({"cam": Tcw, "obs": obs})

    def _note_viewpoint(self, tag_id, cam_pos):
        """Mémorise une direction de visée du tag si elle est nouvelle (> 8° des
        précédentes). Le nombre de directions = diversité de points de vue du tag,
        indépendante du plafond d'images clés du BA."""
        v = cam_pos - self.poses[tag_id][:3, 3]
        n = np.linalg.norm(v)
        if n < 1e-9:
            return
        v = v / n
        dirs = self._tag_bearings[tag_id]
        if all(float(v @ u) < self._view_cos_thr for u in dirs):  # assez différente
            if len(dirs) < 30:
                dirs.append(v)

    def _update_tag_errors(self, visible, T_world_cam):
        """Erreur de REPROJECTION par tag : on reprojette les coins 3D du tag (via
        la carte + la pose caméra) et on compare aux coins détectés. Indicateur de
        qualité (px) : faible = le tag est bien placé dans la carte. Lissé (EMA)."""
        if self.camera_matrix is None or not self.tag_size:
            return
        Tcw = invert(T_world_cam)
        Rcw, tcw = Tcw[:3, :3], Tcw[:3, 3]
        cam_pos = T_world_cam[:3, 3]
        corners3d = _TAG_CORNERS_UNIT * (self.tag_size / 2.0)
        for d in visible:
            self._tag_seen[d.tag_id] += 1
            self._note_viewpoint(d.tag_id, cam_pos)        # diversité d'angles (gel)
            corners = getattr(d, "corners", None)
            if corners is None:
                continue
            # Position image du tag (0 = centre, ~1 = coin) -> indicateur "bord de champ".
            ctr = np.asarray(corners).mean(axis=0)
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            radius = float(np.hypot(ctr[0] - cx, ctr[1] - cy) / np.hypot(cx, cy))
            prev_r = self._tag_radius.get(d.tag_id)
            self._tag_radius[d.tag_id] = radius if prev_r is None else 0.8 * prev_r + 0.2 * radius
            Tw = self.poses[d.tag_id]
            cw = corners3d @ Tw[:3, :3].T + Tw[:3, 3]      # coins -> monde
            cc = cw @ Rcw.T + tcw                           # -> repère caméra
            if np.any(cc[:, 2] <= 1e-6):                    # derrière la caméra
                continue
            proj = (cc @ self.camera_matrix.T)
            proj = proj[:, :2] / proj[:, 2:3]               # -> pixels
            err = float(np.linalg.norm(proj - np.asarray(corners), axis=1).mean())
            prev = self.tag_error.get(d.tag_id)
            self.tag_error[d.tag_id] = err if prev is None else 0.8 * prev + 0.2 * err

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

    def refine(self):
        """Bundle adjustment : optimise GLOBALEMENT les poses des tags et des
        images clés pour minimiser l'erreur de reprojection des coins (méthode
        Pupil Labs). Le tag de référence reste fixe (origine). Robuste (perte de
        Huber). Renvoie (err_avant_px, err_apres_px) ou None. À lancer à la demande.
        """
        if self.camera_matrix is None or not self.tag_size or len(self.keyframes) < 3:
            return None
        tag_list = sorted({tid for kf in self.keyframes for (tid, _) in kf["obs"]
                           if tid in self.poses and tid != self.reference_id})
        if not tag_list:
            return None
        tag_idx = {t: i for i, t in enumerate(tag_list)}
        ntag, nkf = len(tag_list), len(self.keyframes)
        K = self.camera_matrix
        corners_unit = _TAG_CORNERS_UNIT * (self.tag_size / 2.0)

        # Observations : src 0 = référence (fixe), src k+1 = tag_list[k].
        obs_kf, obs_src, obs_uv = [], [], []
        for ki, kf in enumerate(self.keyframes):
            for (tid, corners) in kf["obs"]:
                if tid == self.reference_id:
                    src = 0
                elif tid in tag_idx:
                    src = tag_idx[tid] + 1
                else:
                    continue
                obs_kf.append(ki); obs_src.append(src); obs_uv.append(corners)
        if len(obs_kf) < ntag + nkf:
            return None
        obs_kf = np.array(obs_kf); obs_src = np.array(obs_src)
        obs_uv = np.stack(obs_uv)                                  # (No, 4, 2)

        x0 = np.concatenate(
            [np.concatenate([_pose_to_params(self.poses[t]) for t in tag_list]),
             np.concatenate([_pose_to_params(kf["cam"]) for kf in self.keyframes])])

        def residuals(x):
            tagP = x[:ntag * 6].reshape(ntag, 6)
            camP = x[ntag * 6:].reshape(nkf, 6)
            Rt = np.array([cv2.Rodrigues(p[:3])[0] for p in tagP])
            Rc = np.array([cv2.Rodrigues(p[:3])[0] for p in camP])
            WC = np.empty((ntag + 1, 4, 3))
            WC[0] = corners_unit                                   # référence
            WC[1:] = np.einsum("tab,pb->tpa", Rt, corners_unit) + tagP[:, None, 3:6]
            wc = WC[obs_src]                                        # (No,4,3) monde
            cc = np.einsum("oab,opb->opa", Rc[obs_kf], wc) + camP[obs_kf, None, 3:6]
            uvh = np.einsum("ab,opb->opa", K, cc)
            z = np.where(np.abs(uvh[..., 2:3]) < 1e-6, 1e-6, uvh[..., 2:3])
            return (uvh[..., :2] / z - obs_uv).ravel()

        from scipy.sparse import lil_matrix
        No = len(obs_kf)
        S = lil_matrix((No * 8, (ntag + nkf) * 6), dtype=int)
        for o in range(No):
            r = slice(o * 8, o * 8 + 8)
            if obs_src[o] > 0:
                c = (obs_src[o] - 1) * 6
                S[r, c:c + 6] = 1
            c = ntag * 6 + obs_kf[o] * 6
            S[r, c:c + 6] = 1

        def px(res):
            return float(np.linalg.norm(res.reshape(-1, 2), axis=1).mean())

        before = px(residuals(x0))
        try:
            sol = least_squares(residuals, x0, jac_sparsity=S, method="trf",
                                x_scale="jac", max_nfev=200)
        except Exception:
            return (before, before)
        after = px(sol.fun)
        if after >= before:                       # pas d'amélioration -> on ne touche rien
            return (before, before)
        tagP = sol.x[:ntag * 6].reshape(ntag, 6)
        camP = sol.x[ntag * 6:].reshape(nkf, 6)
        for i, t in enumerate(tag_list):
            self.poses[t] = _params_to_pose(tagP[i])
        for ki in range(nkf):
            self.keyframes[ki]["cam"] = _params_to_pose(camP[ki])
        return (before, after)

    def diagnostics(self):
        """Diagnostic par tag (qualité / contraintes) :
          - observations : nb d'images où le tag a été vu ;
          - viewpoints   : nb d'angles de vue distincts ;
          - reproj_px    : erreur de reprojection (px) ;
          - hops_to_ref  : distance (sauts) au tag de référence dans le graphe ;
          - neighbors    : nb de tags voisins (arêtes fiables) ;
          - frozen       : pose verrouillée ?
        Permet de repérer les tags faibles (peu vus / un seul angle / loin / élevé)."""
        adj = defaultdict(list)
        for (i, j), trusted in self._edge_trusted.items():
            if trusted:
                adj[i].append(j)
        hops = {self.reference_id: 0}
        queue = deque([self.reference_id])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in hops:
                    hops[nxt] = hops[cur] + 1
                    queue.append(nxt)
        diag = {}
        for t in self.poses:
            err = self.tag_error.get(t)
            r = self._tag_radius.get(t)
            diag[t] = {
                "observations": int(self._tag_seen.get(t, 0)),
                "viewpoints": len(self._tag_bearings.get(t, ())),
                "parallax_deg": round(self._parallax_deg(t), 1),  # angle max de visée -> Z fiable
                "reproj_px": round(err, 3) if err is not None else None,
                "img_radius": round(r, 2) if r is not None else None,  # 0=centre, ~1=bord
                "hops_to_ref": hops.get(t),
                "neighbors": len(adj.get(t, [])),
                "frozen": t in self.frozen,
            }
        return diag

    def _parallax_deg(self, tag_id):
        """Angle MAX entre les directions de visée du tag (bras de levier de
        triangulation). Faible -> Z peu observable ; grand -> Z fiable."""
        dirs = self._tag_bearings.get(tag_id, [])
        if len(dirs) < 2:
            return 0.0
        B = np.array(dirs)
        mindot = float(np.clip((B @ B.T).min(), -1.0, 1.0))
        return float(np.degrees(np.arccos(mindot)))

    def stats(self):
        return {
            "tags_localises": len(self.poses),
            "aretes": len(self._edge_n),
            "reference_id": self.reference_id,
            "reference_vue": self.reference_id in self.poses
            and any(self.reference_id in k for k in self._edge_n),
        }
