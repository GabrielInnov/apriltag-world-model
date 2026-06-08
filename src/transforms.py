"""Utilitaires de transformations rigides SE(3) (matrices 4x4)."""

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(R, t):
    """Construit une matrice homogène 4x4 à partir d'une rotation 3x3 et d'une translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R).reshape(3, 3)
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert(T):
    """Inverse d'une transformation rigide (plus stable que np.linalg.inv)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def orthonormalize(R):
    """Projette une matrice 3x3 sur la rotation propre (SO(3)) la plus proche.
    Corrige les petites erreurs numériques ; force un déterminant +1."""
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:           # réflexion -> on rétablit une rotation propre
        U[:, -1] = -U[:, -1]
        Rp = U @ Vt
    return Rp


def average_transforms(transforms):
    """Moyenne robuste d'une liste de transformations 4x4.

    - translation : moyenne arithmétique
    - rotation    : moyenne sur SO(3) (via quaternions, scipy)

    Les détections dégénérées (rotation au déterminant <= 0 ou non finie, fréquentes
    sur des poses AprilTag ambiguës) sont ignorées pour ne pas corrompre le résultat
    ni faire planter scipy.
    """
    transforms = list(transforms)
    if not transforms:
        raise ValueError("Liste de transformations vide")

    rots, ts = [], []
    for T in transforms:
        R = T[:3, :3]
        if np.all(np.isfinite(R)) and np.linalg.det(R) > 1e-6:
            rots.append(orthonormalize(R))
            ts.append(T[:3, 3])

    if not rots:
        # Aucune rotation exploitable : pose neutre à la translation moyenne.
        mean_t = np.mean([T[:3, 3] for T in transforms], axis=0)
        return make_transform(np.eye(3), mean_t)

    mean_t = np.mean(ts, axis=0)
    if len(rots) == 1:
        return make_transform(rots[0], mean_t)
    mean_R = Rotation.from_matrix(np.array(rots)).mean().as_matrix()
    return make_transform(mean_R, mean_t)


def transform_to_dict(T):
    """Sérialise une transformation 4x4 pour l'export JSON."""
    R = T[:3, :3]
    t = T[:3, 3]
    quat = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    return {
        "matrix": T.tolist(),
        "translation_m": t.tolist(),
        "quaternion_xyzw": quat.tolist(),
        "euler_xyz_deg": euler.tolist(),
    }
