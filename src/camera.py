"""Abstraction de capture caméra : Daheng (gxipy) avec fallback webcam / fichier vidéo."""

import os
import sys
import threading
import time

import cv2

# Racines d'installation possibles du Galaxy SDK Daheng.
_DAHENG_ROOTS = [
    r"C:\Program Files\Daheng Imaging\GalaxySDK",
    r"C:\Program Files (x86)\Daheng Imaging\GalaxySDK",
]


def _register_daheng_dll_path():
    """Rend les DLL natives Daheng chargeables par gxipy.

    gxipy charge `GxIAPI.dll` via `WinDLL('GxIAPI.dll', winmode=0)`, ce qui ignore
    `os.add_dll_directory`. GxIAPI.dll dépend en plus des DLL GenICam. On déclare
    donc tous les dossiers de DLL puis on pré-charge GxIAPI par chemin absolu :
    une fois en mémoire, gxipy la retrouve par son nom de base.
    """
    import ctypes

    arch = "Win64" if sys.maxsize > 2 ** 32 else "Win32"
    genicam_arch = "Win64_x64" if arch == "Win64" else "Win32_i86"

    for root in _DAHENG_ROOTS:
        api_dir = os.path.join(root, "APIDll", arch)
        if not os.path.isdir(api_dir):
            continue
        dep_dirs = [
            api_dir,
            os.path.join(root, "GenICam", "bin", genicam_arch),
            os.path.join(root, "GenTL", arch),
        ]
        for d in dep_dirs:
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except (OSError, AttributeError):
                    pass
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

        for name in ("DxImageProc.dll", "GxIAPI.dll"):
            p = os.path.join(api_dir, name)
            if os.path.isfile(p):
                try:
                    ctypes.WinDLL(p, winmode=0x00000008)  # LOAD_WITH_ALTERED_SEARCH_PATH
                except OSError:
                    pass
        return api_dir
    return None


class Camera:
    """Interface commune. Retourne des images BGR (convention OpenCV)."""

    kind = None   # "daheng" | "webcam" | "video" : sert à choisir la calibration

    def read(self):
        raise NotImplementedError

    def set_exposure(self, microseconds):
        """Règle le temps d'exposition (µs). No-op si non supporté."""
        pass

    def get_exposure(self):
        """Temps d'exposition courant (µs) ou None si inconnu."""
        return None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class OpenCVCamera(Camera):
    """Webcam (index entier) ou fichier vidéo (chemin)."""

    def __init__(self, source=0, width=None, height=None):
        # Webcam (index entier) sur Windows : DirectShow est plus stable que le
        # backend MSMF par défaut (qui provoque des "can't grab frame -2147024809").
        if isinstance(source, int) and sys.platform == "win32":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)   # repli backend par défaut
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source vidéo : {source!r}")
        # Force la résolution de capture (pour coller à la calibration).
        # MJPG d'abord : beaucoup de webcams n'atteignent le 1080p qu'en MJPG.
        if width and height:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera] Webcam {w}x{h}.")
        self.source = source
        self.kind = "video" if isinstance(source, str) else "webcam"

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class DahengCamera(Camera):
    """Caméra Daheng via le SDK gxipy (acquisition continue)."""

    kind = "daheng"

    def __init__(self, index=1, timeout_ms=1000, exposure_us=None, binning=1):
        _register_daheng_dll_path()
        # gxipy est "vendored" à la racine du projet (src/..) : on l'ajoute au path.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        import gxipy as gx  # importé tardivement : optionnel

        self._gx = gx
        self.timeout_ms = timeout_ms
        self.manager = gx.DeviceManager()
        dev_num, _ = self.manager.update_device_list()
        if dev_num == 0:
            raise RuntimeError("Aucune caméra Daheng détectée")
        self.cam = self.manager.open_device_by_index(index)
        # Mode libre (pas de trigger) + acquisition continue.
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
        # Exposition : soit tout en auto, soit FIXE (expo + gain figés). Sans figer
        # le gain, l'auto-gain compense la luminosité et changer l'expo ne se voit pas.
        if exposure_us in (None, "auto"):
            self._enable_auto(gx)
        else:
            self.set_exposure(exposure_us)
        # Binning : réduit la résolution capteur (2 = 2x2 => 4x moins de pixels,
        # capture et traitement bien plus rapides). Intrinsèques ré-ajustées côté main.
        if binning and binning > 1:
            for setter in (lambda: self.cam.BinningHorizontal.set(binning),
                           lambda: self.cam.BinningVertical.set(binning)):
                try:
                    setter()
                except Exception:
                    pass
        # Peu de buffers : on veut l'image la PLUS RÉCENTE, pas une file qui
        # accumule du retard quand le traitement est plus lent que l'acquisition.
        try:
            self.cam.data_stream[0].set_acquisition_buffer_number(2)
        except Exception:
            pass
        self.cam.stream_on()

    def _enable_auto(self, gx):
        """Active exposition/gain/balance des blancs automatiques (défensif :
        chaque réglage est ignoré si le modèle ne l'expose pas)."""
        for setter in (
            lambda: self.cam.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS),
            lambda: self.cam.GainAuto.set(gx.GxAutoEntry.CONTINUOUS),
            lambda: self.cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS),
        ):
            try:
                setter()
            except Exception:
                pass

    def set_exposure(self, microseconds):
        """Fixe le temps de pose (µs) et fige expo + gain (sinon l'auto-gain
        compense la luminosité et le réglage d'expo n'a aucun effet visible).
        Borné aux limites du capteur ; ignoré si le modèle ne l'expose pas."""
        for setter in (
            lambda: self.cam.ExposureAuto.set(self._gx.GxAutoEntry.OFF),
            lambda: self.cam.GainAuto.set(self._gx.GxAutoEntry.OFF),
        ):
            try:
                setter()
            except Exception:
                pass
        try:
            rng = self.cam.ExposureTime.get_range()
            us = float(microseconds)
            us = max(rng["min"], min(rng["max"], us))
            self.cam.ExposureTime.set(us)
            return us
        except Exception:
            return None

    def get_exposure(self):
        try:
            return float(self.cam.ExposureTime.get())
        except Exception:
            return None

    def read(self):
        # Vide la file : on jette les images en retard et on attend la prochaine
        # image fraîche -> latence ~1 frame au lieu d'un retard qui s'accumule.
        try:
            self.cam.data_stream[0].flush_queue()
        except Exception:
            pass
        raw = self.cam.data_stream[0].get_image(self.timeout_ms)
        if raw is None:
            return None
        rgb = raw.convert("RGB")
        if rgb is None:
            return None
        arr = rgb.get_numpy_array()
        if arr is None:
            return None
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def close(self):
        try:
            self.cam.stream_off()
            self.cam.close_device()
        except Exception:
            pass


class AsyncCamera(Camera):
    """Décorateur qui lit la caméra dans un thread de fond et conserve toujours
    la dernière image. La boucle principale récupère cette image SANS bloquer,
    donc l'UI et les touches restent réactives quel que soit le temps de capture."""

    def __init__(self, camera):
        self._cam = camera
        self.kind = getattr(camera, "kind", None)
        self._lock = threading.Lock()
        self._frame = None
        self._version = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                f = self._cam.read()  # peut bloquer (attente d'image fraîche)
            except Exception:
                f = None
            if f is not None:
                with self._lock:
                    self._frame = f
                    self._version += 1
            else:
                time.sleep(0.01)  # caméra muette/déconnectée : ne pas mitrailler

    def read(self):
        with self._lock:
            return self._frame

    def latest(self):
        """(image_ou_None, numéro_de_version) — la version change à chaque
        nouvelle image, pour ne traiter que les images réellement neuves."""
        with self._lock:
            return self._frame, self._version

    def set_exposure(self, microseconds):
        return self._cam.set_exposure(microseconds)

    def get_exposure(self):
        return self._cam.get_exposure()

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cam.close()


def create_camera(cfg):
    """Fabrique une caméra à partir de la config (clé `camera`)."""
    source = cfg.get("source", "auto")

    exposure_us = cfg.get("exposure_us", "auto")
    binning = cfg.get("binning", 1)

    if source in ("auto", "daheng"):
        try:
            cam = DahengCamera(exposure_us=exposure_us, binning=binning)
            exp = cam.get_exposure()
            mode = "auto" if exposure_us in (None, "auto") else "manuelle"
            print(f"[camera] Caméra Daheng ouverte (expo {mode}"
                  + (f", {exp:.0f} µs" if exp else "") + ").")
            return AsyncCamera(cam)  # lecture en thread -> UI réactive
        except Exception as e:
            if source == "daheng":
                raise
            print(f"[camera] Daheng indisponible ({e}). Fallback webcam/vidéo.")

    # Chemin de fichier vidéo explicite : pas de thread (sinon lecture trop rapide).
    if isinstance(source, str) and source not in ("auto", "webcam"):
        print(f"[camera] Lecture du fichier vidéo : {source}")
        return OpenCVCamera(source)

    idx = cfg.get("webcam_index", 0)
    print(f"[camera] Ouverture de la webcam index {idx}.")
    return AsyncCamera(OpenCVCamera(idx, cfg.get("webcam_width"),
                                    cfg.get("webcam_height")))
