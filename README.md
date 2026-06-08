# AprilTag World Model

Construction d'une carte 3D d'AprilTags à partir d'une caméra (Daheng).
Un tag sert d'**origine du monde** ; tous les autres sont localisés par rapport
à lui, par **chaînage multi-vues** (graphe de poses).

## Principe

Pour chaque image :
1. Détection des AprilTags + estimation de la pose `caméra → tag` (`pupil-apriltags` + calibration).
2. Pour chaque paire de tags vus **ensemble**, calcul de la pose relative
   `T_i_j = inv(T_cam_i) · T_cam_j` (indépendante de la position de la caméra).
3. Chaque paire = une arête d'un graphe (nœuds = tags). Observations moyennées.
4. Parcours BFS depuis le tag de référence → pose monde de chaque tag atteignable.

Un tag B jamais vu en même temps que la référence A est quand même localisé si une
chaîne existe (A voit C, C voit B…).

## Installation

```powershell
cd C:\Users\Gabriel.S\apriltag-world-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Caméra Daheng** : `gxipy` n'est pas sur PyPI. Installe le *Galaxy SDK* Daheng,
puis le wheel fourni :
```powershell
pip install "C:\Program Files\Daheng Imaging\GalaxySDK\Development\Samples\Python\gxipy\dist\gxipy-X.X.X-py3-none-any.whl"
```
Sans `gxipy`, le projet bascule automatiquement sur la **webcam** (test).

## Configuration

Édite [config.yaml](config.yaml) :
- `tag.family`, `tag.size_m` (taille réelle du bord noir, en mètres), `tag.reference_id`
- `camera.source` : `auto` | `daheng` | `webcam` | chemin d'un fichier vidéo

Renseigne ta calibration dans [calibration/camera_intrinsics.yaml](calibration/camera_intrinsics.yaml).

## Utilisation

```powershell
python main.py
```
Touches : `q` quitter (sauvegarde auto) · `s` sauvegarder · `r` réinitialiser.

Le résultat est écrit dans `world_model.json` : pour chaque tag, sa matrice 4×4,
sa translation (m), son quaternion et ses angles d'Euler, dans le repère du tag
de référence.

## Structure

| Fichier | Rôle |
|---|---|
| `src/camera.py` | Capture Daheng + fallback webcam/vidéo |
| `src/detector.py` | Détection AprilTag + pose caméra→tag |
| `src/transforms.py` | Utilitaires SE(3), moyenne SO(3) |
| `src/world_model.py` | Graphe de poses, chaînage BFS, moyennage |
| `src/visualizer.py` | Visu 3D temps réel (Open3D) |
| `src/exporter.py` | Sauvegarde/chargement JSON |
| `main.py` | Boucle live |

## Pistes d'amélioration

- Optimisation globale du graphe de poses (g2o / GTSAM / `scipy.optimize`) pour
  réduire la dérive sur les longues chaînes.
- Filtrage des détections par `decision_margin` / taille apparente.
- Pondération des arêtes par la qualité d'observation.
