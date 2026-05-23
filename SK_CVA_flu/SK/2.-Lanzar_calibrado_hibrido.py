# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__   = "10 de mayo de 2026"

"""
2.-Lanzar_calibrado_hibrido.py
------------------------------
Launches N independent PSO + Nelder-Mead calibration runs.

For each run:
  1. Creates RESULTADOS/run_XX/ with FICHAS/, ERRORES/, GUESSES/
  2. Creates evaluaciones.db for that run
  3. Launches the simulator in the background
  4. Launches NM in the background (if nelder_mead.activo = true)
  5. Launches PSO and waits for it to finish
  6. Writes PSO_FIN -> NM performs the final adjustment and writes FIN
     (If NM is not active, writes FIN directly)
  7. Waits for NM and the simulator to stop
  8. Saves resumen_run_XX.yaml to RESULTADOS/
  9. Merges evaluaciones.db into evaluaciones_global.db
 10. Deletes run_XX/ completely

Usage:
    python 2.-Lanzar_calibrado_hibrido.py [--runs N] [--paralelo]
"""

import argparse
import os
import signal
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yaml


# =============================================================================
# BASE PATHS
# =============================================================================

BASE = os.path.dirname(os.path.abspath(__file__))

ruta_cfg_calibrado  = os.path.join(BASE, "0.-Configuracion", "cfg_calibrado.yaml")
ruta_cfg_parametros = os.path.join(BASE, "0.-Configuracion", "cfg_parametros.yaml")

with open(ruta_cfg_calibrado) as f:
    cfg = yaml.safe_load(f)
with open(ruta_cfg_parametros) as f:
    cfg_params = yaml.safe_load(f)

nombre_parametros = list(cfg_params.keys())
numero_objetivos  = cfg['numero_objetivos']
simulador         = cfg['simulador']
nm_activo         = cfg.get('nelder_mead', {}).get('activo', False)

DIR_RESULTADOS = os.path.join(BASE, "RESULTADOS")
DIR_MEJORES    = os.path.join(DIR_RESULTADOS, "MEJORES")
RUTA_GLOBAL_DB = os.path.join(DIR_RESULTADOS, "evaluaciones_global.db")

DIR_PSO       = os.path.join(BASE, "PSO")
DIR_NM        = os.path.join(BASE, "NM")
DIR_SIMULADOR = os.path.join(BASE, "SIMULADOR")

PYTHON = sys.executable

TIMEOUT_ULTIMO_AJUSTE = 600  # maximum seconds for NM's final adjustment


# =============================================================================
# DATABASE
# =============================================================================

def _schema_sql():
    columnas = ["id TEXT PRIMARY KEY", "semilla INTEGER"]
    for nombre in nombre_parametros:
        columnas.append(f"{nombre} REAL")
    for i in range(numero_objetivos):
        columnas.append(f"error_{i+1} REAL")
    columnas.append("timestamp TEXT")
    return f"CREATE TABLE IF NOT EXISTS evaluaciones ({', '.join(columnas)})"


def crear_db(ruta):
    conn = sqlite3.connect(ruta)
    conn.execute(_schema_sql())
    conn.commit()
    conn.close()


def mejor_de_db(ruta):
    try:
        conn  = sqlite3.connect(ruta)
        cols  = ', '.join(nombre_parametros)
        fila  = conn.execute(
            f"SELECT {cols}, error_1 FROM evaluaciones ORDER BY error_1 ASC LIMIT 1"
        ).fetchone()
        conn.close()
        if fila is None:
            return None, None
        return dict(zip(nombre_parametros, fila[:-1])), fila[-1]
    except Exception:
        return None, None


def n_evaluaciones_db(ruta):
    try:
        conn = sqlite3.connect(ruta)
        n    = conn.execute("SELECT COUNT(*) FROM evaluaciones").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def merge_db(ruta_run):
    conn = sqlite3.connect(RUTA_GLOBAL_DB)
    conn.execute(_schema_sql())
    conn.execute(f"ATTACH DATABASE '{ruta_run}' AS run_db")
    conn.execute("INSERT OR IGNORE INTO evaluaciones SELECT * FROM run_db.evaluaciones")
    conn.commit()
    conn.execute("DETACH DATABASE run_db")
    conn.close()
    os.remove(ruta_run)


# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================

def terminar_procesos(procesos, timeout=30):
    t0    = time.time()
    vivos = list(procesos)
    while vivos and (time.time() - t0) < timeout:
        vivos = [p for p in vivos if p.poll() is None]
        if vivos:
            time.sleep(1)
    for p in vivos:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(2)
    for p in vivos:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


# =============================================================================
# CLEAN STOP (Ctrl+C)
# =============================================================================

_fichas_activos = set()
_fichas_lock    = threading.Lock()
parar           = threading.Event()

def _handler_parada(sig, frame):
    print("\n\n  Ctrl+C — sending FIN to active runs...\n")
    parar.set()
    with _fichas_lock:
        for df in _fichas_activos:
            try:
                with open(os.path.join(df, 'FIN'), 'w') as f:
                    f.write('Stop requested by user\n')
            except Exception:
                pass


# =============================================================================
# RUN EXECUTION
# =============================================================================

def ejecutar_run(run_num):
    """
    Executes a complete PSO + NM (if active) calibration run.
    Returns (run_num, params_dict, best_error, n_evaluations).
    """
    run_dir     = os.path.join(DIR_RESULTADOS, f"run_{run_num:02d}")
    dir_fichas  = os.path.join(run_dir, "FICHAS")
    dir_errores = os.path.join(run_dir, "ERRORES")
    dir_guesses = os.path.join(run_dir, "GUESSES")
    ruta_db     = os.path.join(run_dir, "evaluaciones.db")

    os.makedirs(dir_fichas,  exist_ok=True)
    os.makedirs(dir_errores, exist_ok=True)
    os.makedirs(dir_guesses, exist_ok=True)
    crear_db(ruta_db)

    with _fichas_lock:
        _fichas_activos.add(dir_fichas)

    p_nm = None

    try:
        # --- Simulator ---
        p_sim = subprocess.Popen(
            [PYTHON, "-u", f"{simulador}.py",
             "--fichas",  dir_fichas,
             "--errores", dir_errores,
             "--db",      ruta_db],
            cwd=DIR_SIMULADOR
        )

        time.sleep(1)

        # --- NM in background ---
        if nm_activo:
            p_nm = subprocess.Popen(
                [PYTHON, "-u", "Modulo_NM.py",
                 "--fichas",   dir_fichas,
                 "--errores",  dir_errores,
                 "--guesses",  dir_guesses,
                 "--mejores",  DIR_MEJORES,
                 "--run",      str(run_num),
                 "--t_inicio", str(t_inicio)],
                cwd=DIR_NM
            )

        # --- PSO (blocking) ---
        subprocess.run(
            [PYTHON, "Modulo_MO_rPSO.py",
             "--fichas",   dir_fichas,
             "--errores",  dir_errores,
             "--mejores",  DIR_MEJORES,
             "--guesses",  dir_guesses,
             "--run",      str(run_num),
             "--t_inicio", str(t_inicio)],
            cwd=DIR_PSO
        )

        # --- PSO has finished (already wrote PSO_FIN) ---
        fin_path = os.path.join(dir_fichas, 'FIN')

        if nm_activo and p_nm is not None and p_nm.poll() is None:
            # NM detects PSO_FIN, does the final adjustment, and writes FIN
            try:
                p_nm.wait(timeout=TIMEOUT_ULTIMO_AJUSTE)
            except subprocess.TimeoutExpired:
                print(f"  Run {run_num:02d}  NM took too long in the final adjustment. Forcing stop.")
                p_nm.terminate()
                # NM did not write FIN; we write it
                if not os.path.exists(fin_path):
                    with open(fin_path, 'w') as f:
                        f.write('NM timeout\n')
        else:
            # NM not active or already stopped: write FIN to stop the simulator
            if not os.path.exists(fin_path):
                with open(fin_path, 'w') as f:
                    f.write('PSO finished\n')

        terminar_procesos([p_sim], timeout=30)

    finally:
        with _fichas_lock:
            _fichas_activos.discard(dir_fichas)

    # --- Best result: DB + possible NM final improvement ---
    params, error = mejor_de_db(ruta_db)
    n_eval        = n_evaluaciones_db(ruta_db)

    ruta_final_nm = os.path.join(DIR_MEJORES, f"run_{run_num:02d}_final_nm.yaml")
    if os.path.exists(ruta_final_nm):
        try:
            with open(ruta_final_nm, 'r') as f:
                datos_nm = yaml.safe_load(f)
            error_nm = datos_nm.get('Error_1', float('inf'))
            if error is None or error_nm < error:
                error  = error_nm
                params = {k: datos_nm[k] for k in nombre_parametros}
        except Exception:
            pass

    resumen = {
        'run':                run_num,
        'timestamp':          datetime.now().isoformat(),
        'evaluaciones':       n_eval,
        'mejor_error':        float(error) if error is not None else None,
        'mejores_parametros': {k: float(v) for k, v in params.items()} if params else None,
    }
    with open(os.path.join(DIR_RESULTADOS, f"resumen_run_{run_num:02d}.yaml"), 'w') as f:
        yaml.dump(resumen, f, default_flow_style=False, allow_unicode=True)

    merge_db(ruta_db)
    shutil.rmtree(run_dir)

    return run_num, params, error, n_eval


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description="PSO + NM calibration launcher")
parser.add_argument('--runs',     type=int,        default=5,
                    help='Number of runs (default: 5)')
parser.add_argument('--paralelo', action='store_true',
                    help='Launch all runs simultaneously')
args = parser.parse_args()


# =============================================================================
# SETUP
# =============================================================================

os.makedirs(DIR_MEJORES, exist_ok=True)
t_inicio = time.time()
signal.signal(signal.SIGINT, _handler_parada)

# Continue numbering from existing runs if any
nums_existentes = []
if os.path.isdir(DIR_RESULTADOS):
    for entry in os.listdir(DIR_RESULTADOS):
        if entry.startswith("run_"):
            try:
                nums_existentes.append(int(entry.split("_")[1]))
            except (IndexError, ValueError):
                pass
primer_run   = max(nums_existentes, default=0) + 1
runs_totales = list(range(primer_run, primer_run + args.runs))

modo    = "en paralelo" if args.paralelo else "secuenciales"
modo_nm = "PSO + NM" if nm_activo else "PSO only"

print(f"\n{'═'*60}", flush=True)
print(f"  Calibration {modo_nm} — {args.runs} runs {modo}", flush=True)
print(f"  Simulator: {simulador}", flush=True)
print(f"  Runs: {runs_totales[0]} to {runs_totales[-1]}", flush=True)
print(f"{'═'*60}\n", flush=True)


# =============================================================================
# LAUNCH
# =============================================================================

resultados = []

if args.paralelo:
    with ThreadPoolExecutor(max_workers=args.runs) as executor:
        futures = {executor.submit(ejecutar_run, r): r for r in runs_totales}
        for future in as_completed(futures):
            run_num, params, error, n_eval = future.result()
            resultados.append((run_num, params, error, n_eval))
            if error is not None:
                print(f"  Run {run_num:02d} finished — error: {error:.4f}  ({n_eval} evals)")
            else:
                print(f"  Run {run_num:02d} finished — no valid results")
else:
    for run_num in runs_totales:
        if parar.is_set():
            break
        print(f"  Starting run {run_num:02d}/{runs_totales[-1]}  "
              f"[{datetime.now().strftime('%H:%M:%S')}]", flush=True)
        run_num, params, error, n_eval = ejecutar_run(run_num)
        resultados.append((run_num, params, error, n_eval))
        if error is not None:
            print(f"  Run {run_num:02d} finished — error: {error:.4f}  ({n_eval} evals)\n")
        else:
            print(f"  Run {run_num:02d} finished — no valid results\n")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'═'*60}")
print(f"  SUMMARY OF ALL RUNS")
print(f"{'═'*60}\n")

resultados.sort(key=lambda x: x[0])

mejor_error  = float('inf')
mejor_run    = None
mejor_params = None

print(f"  {'Run':>4}  {'Error':>12}  {'Evals':>6}")
print(f"  {'─'*4}  {'─'*12}  {'─'*6}")

for run_num, params, error, n_eval in resultados:
    marcador = ''
    if error is not None and error < mejor_error:
        mejor_error  = error
        mejor_run    = run_num
        mejor_params = params
        marcador     = ' ◄ BEST'
    error_str = f"{error:.4f}" if error is not None else "N/A"
    print(f"  {run_num:>4}  {error_str:>12}  {n_eval:>6}{marcador}")

if mejor_run is not None:
    print(f"\n  Global best: run {mejor_run}, error = {mejor_error:.4f}")
    print(f"\n  Optimal parameters:")
    for nombre, valor in mejor_params.items():
        print(f"    {nombre:8s} = {valor:.6f}")
    print(f"\n  Results in: {DIR_RESULTADOS}")

print(f"\n  A mayor Gloria de Dios Nuestro Señor\n")

# END -------------------------------------------------------------------------
