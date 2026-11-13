"""Menjalankan notebook forecasting berurutan (1 jam -> 30 menit -> 15 menit) dengan kernel skripsi-gpu.

Notebook sumber tidak diubah; salinan yang sudah dieksekusi disimpan di folder ini.
Pemakaian: python hasil_eksekusi/jalankan_semua.py [nama_notebook ...]
"""
import os, sys, time, ctypes, traceback
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "hasil_eksekusi")
LOG = os.path.join(OUT, "run_log.txt")
NOTEBOOKS = sys.argv[1:] or ["skripsi_forecast_v3.ipynb", "skripsi_forecast_30m.ipynb", "skripsi_forecast_15m.ipynb"]

os.chdir(ROOT)
os.environ["PYTHONUTF8"] = "1"; os.environ["PYTHONIOENCODING"] = "utf-8"
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # cegah sleep selama proses ini hidup
except Exception: pass

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")

rc_all = 0
for name in NOTEBOOKS:
    nb = nbformat.read(os.path.join(ROOT, name), as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="skripsi-gpu", resources={"metadata": {"path": ROOT}})
    log(f"START {name}"); t0 = time.time(); rc = 0
    try:
        client.execute()
    except CellExecutionError as e:
        rc = 1; rc_all = 1; log(f"GAGAL {name}:\n{str(e)[-3000:]}")
    except Exception:
        rc = 2; rc_all = 1; log(f"GAGAL {name}:\n{traceback.format_exc()[-3000:]}")
    os.makedirs(OUT, exist_ok=True)
    nbformat.write(nb, os.path.join(OUT, name))
    log(f"END {name} rc={rc} {(time.time() - t0) / 60:.1f} min")
log(f"SELESAI rc={rc_all}")
sys.exit(rc_all)
