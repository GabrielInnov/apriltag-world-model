"""Vérifie que toutes les dépendances et modules du projet s'importent."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, "OK", ""))
    except Exception as e:
        results.append((name, "ECHEC", f"{type(e).__name__}: {e}"))


check("numpy", lambda: __import__("numpy"))
check("scipy", lambda: __import__("scipy"))
check("cv2 (opencv)", lambda: __import__("cv2"))
check("yaml", lambda: __import__("yaml"))
check("pupil_apriltags", lambda: __import__("pupil_apriltags"))
check("open3d", lambda: __import__("open3d"))

check("src.transforms", lambda: __import__("transforms"))
check("src.detector", lambda: __import__("detector"))
check("src.world_model", lambda: __import__("world_model"))
check("src.exporter", lambda: __import__("exporter"))
check("src.camera", lambda: __import__("camera"))
check("src.visualizer", lambda: __import__("visualizer"))

width = max(len(n) for n, _, _ in results)
all_ok = True
for name, status, detail in results:
    all_ok = all_ok and status == "OK"
    print(f"  {name.ljust(width)}  {status}  {detail}")
print("RESULTAT:", "TOUT OK" if all_ok else "DES IMPORTS ONT ECHOUE")
sys.exit(0 if all_ok else 1)
