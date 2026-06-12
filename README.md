# AprilTag World Model

Cartographie 3D d'AprilTags et **localisation de la caméra** dans l'espace.
Un tag sert d'**origine du monde** ; tous les autres sont placés par rapport à lui
par **chaînage multi-vues** (graphe de poses), puis la pose de la caméra est
estimée en temps réel dans cette carte.

But final : **connaître la position (X, Y, Z) de la caméra** par rapport au tag de
référence, en direct.

## Principe (pipeline)

Pour chaque image :

1. **Détection** des AprilTags (`pupil-apriltags`), coins raffinés au **sous-pixel**
   et moyennés temporellement quand la caméra est immobile.
2. **Pose `caméra → tag`** recalculée par PnP planaire (`solvePnP IPPE_SQUARE`),
   qui choisit la meilleure des deux solutions (atténue l'ambiguïté de retournement).
3. **Arêtes** : pour deux tags vus ensemble, pose relative
   `T_i_j = inv(T_cam_i) · T_cam_j` (indépendante de la position de la caméra).
4. **Moyennage** des observations par arête (translation : moyenne pondérée ;
   rotation : méthode de Markley), avec **filtre qualité** et **pondération** par
   la taille apparente.
5. **Confirmation** : un tag n'est ajouté à la carte que si son arête est fiable
   (assez d'observations, rotation cohérente, translation peu dispersée).
6. **Placement** : parcours BFS depuis la référence, puis **relaxation du graphe
   de poses** (utilise toutes les arêtes → fermeture de boucle, réduit la dérive).
7. **Bundle adjustment** (à la demande) : optimisation globale par minimisation de
   l'erreur de reprojection des coins.
8. **Localisation caméra** : PnP multi-tags (`solvePnP SQPNP`) sur tous les tags
   mappés visibles → pose stable.

Un tag jamais vu en même temps que la référence est quand même localisé si une
chaîne existe (A voit C, C voit B…).

## Installation

```powershell
cd <chemin>\apriltag-world-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Caméra Daheng** (optionnelle) : `gxipy` n'est pas sur PyPI. Installe le
*Galaxy SDK* Daheng, puis le wheel fourni :
```powershell
pip install "C:\Program Files\Daheng Imaging\GalaxySDK\...\gxipy-X.X.X-py3-none-any.whl"
```
Sans `gxipy`, le projet bascule automatiquement sur une **webcam**.

## Configuration

Tout se règle dans [config.yaml](config.yaml).

**`tag`** — `family`, `size_m` (taille réelle du bord noir **en mètres**, critique
pour l'échelle), `reference_id` (le tag origine, qui **doit être filmé**).

**`camera`** — `source` : `auto` | `daheng` | `webcam` | chemin d'une vidéo.
- Daheng : `exposure_us` (µs ou `auto`), `binning` (1 ou 2).
- Webcam : `webcam_index`, `webcam_width` / `webcam_height` (force la résolution =
  celle de la calibration).
- `intrinsics` : fichier de calibration (`fx, fy, cx, cy`, distorsion, résolution).
  Les intrinsèques sont **auto-ajustées** si la résolution réelle diffère.

**`detection`** — `quad_decimate` (vitesse/précision), `subpixel`, `temporal_avg`.

**`mapping`** — `enabled` (calibration on/off), `min_observations`,
`min_decision_margin`, `min_tag_pixels`, `rot_coherence_min`, `max_trans_std_m`
(filtre/confirmation), `refine_iters` (relaxation du graphe), `freeze_confident` /
`freeze_reproj_px` / `freeze_min_obs` (verrouillage des tags confiants).

**`visualization`** — `autostart` (ouvrir la 3D au démarrage), `trace`.

Renseigne ta calibration dans [calibration/](calibration/) (une par caméra).

## Utilisation

```powershell
python main.py
```

La **vue caméra** s'ouvre avec un **panneau de boutons à droite**. La vue **3D**
(Open3D) s'ouvre à la demande.

**Boutons** : Enregistrer · Calib ON/OFF · Afficher/Masquer 3D · **Affiner (BA)** ·
Reset carte · Expo −/+ · Quitter.

**Touches** (fenêtre caméra) : `q` quitter (sauvegarde auto) · `s` sauvegarder ·
`r` reset · `c` calibration on/off · `v` vue 3D · `t` trace · `a` affiner (BA) ·
`+/−` exposition.

**Navigation 3D** : flèches = pivoter · `W`/`S` ou molette = zoom · `I`/`J`/`K`/`L` =
déplacer · `F` = vue d'ensemble · `R` = recentrer · `T` = trace.

### Lecture à l'écran
- **Vue caméra** : contour rouge = tag de référence, **vert = cartographié**,
  jaune = vu mais pas encore intégré. HUD : nb de tags, erreur de reprojection
  moyenne, images clés, tags figés, **position caméra X/Y/Z** vs référence.
- **Vue 3D** : tags colorés du **vert** (précis) au **rouge** (erreur de reprojection
  élevée) ; **contour blanc = tag verrouillé** ; cône orange = caméra ; trace cyan =
  trajectoire.

### Mode calibration vs localisation
- **Calib ON** : on construit/affine la carte.
- **Calib OFF** : carte figée, la caméra se localise dedans (suivi X/Y/Z).

Le résultat est écrit dans `world_model.json` : pour chaque tag, sa matrice 4×4,
sa translation (m), son quaternion et ses angles d'Euler, dans le repère du tag
de référence.

## Conseils pour de bons résultats

- Le **tag de référence doit être filmé** (sinon avertissement « jamais vu », rien
  ne se localise).
- Renseigner **`tag.size_m`** exactement (échelle de toute la carte).
- Filmer chaque tag **sous plusieurs angles**, aussi **gros/net** que possible,
  bien éclairé, caméra lente (le moyennage temporel se déclenche à l'arrêt).
- Cliquer **« Affiner » (BA)** en fin de scan pour la meilleure précision.
- Une **bonne calibration** (couvrant tout le cadre) abaisse le plancher d'erreur.

## Méthodes

| Étape | Méthode |
|---|---|
| Pose d'un tag | PnP planaire / IPPE (décomposition d'homographie) |
| Arête entre tags | Composition SE(3) (annulation de la caméra) |
| Moyennage des rotations | Markley (vecteur propre dominant de `Σ w·q·qᵀ`) |
| Confiance d'une arête | `λ_max / Σw` (cohérence) + variance de translation |
| Init de la carte | BFS + composition le long d'un arbre couvrant |
| Cohérence globale | relaxation du graphe de poses (Jacobi pondéré, fermeture de boucle) |
| Raffinage final | bundle adjustment (moindres carrés non linéaires, Levenberg–Marquardt) |
| Pose caméra | PnP multi-tags (SQPnP) |
| Stabilité | verrouillage des tags confiants (ancres fixes) |

## Structure

| Fichier | Rôle |
|---|---|
| `src/camera.py` | Capture Daheng / webcam (thread async, binning, expo, résolution) |
| `src/detector.py` | Détection AprilTag, coins sous-pixel + temporel, pose IPPE |
| `src/transforms.py` | Utilitaires SE(3), orthonormalisation, moyenne SO(3) |
| `src/world_model.py` | Graphe de poses, moyennage Markley, confirmation, relaxation, bundle adjustment, PnP caméra, verrouillage |
| `src/visualizer.py` | Visu 3D temps réel (Open3D), navigation clavier |
| `src/exporter.py` | Sauvegarde / chargement JSON |
| `main.py` | Boucle live, menu, HUD, position X/Y/Z |

## Pistes d'amélioration

- Rejet explicite de l'ambiguïté de retournement via `pose_err`.
- Bundle adjustment **automatique** en arrière-plan (sur Calib OFF / par lots).
- Pondération par incertitude (covariance) plutôt que par taille apparente.
- Diagnostic par tag exporté (observations, reprojection, distance à la référence).
