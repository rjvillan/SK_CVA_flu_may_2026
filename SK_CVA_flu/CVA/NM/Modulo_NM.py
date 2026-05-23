# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__   = "10 de mayo de 2026"

"""
NM/Modulo_NM.py
---------------
Nelder-Mead module for hybrid PSO+NM calibration.

Takes the best global point from PSO, builds a simplex around it,
and runs Nelder-Mead. Each evaluation communicates with the simulator
via YAML/error files, exactly as PSO does.

After convergence, deposits the result in GUESSES/ for PSO to pick up.
Waits until PSO (or another process) finds a point better than the one
NM reached before launching a new run.
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import yaml
from scipy.optimize import minimize

# Reuse PSO module functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'PSO'))
from aux_externos import generar_ID, escribir_guess
from aux_PSO import leer_fichero_error


# =============================================================================
# FORCED-STOP EXCEPTION
# =============================================================================

class ParadaForzada(Exception):
    pass


# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fichas',   required=True)
parser.add_argument('--errores',  required=True)
parser.add_argument('--guesses',  required=True)
parser.add_argument('--mejores',  required=True)
parser.add_argument('--run',      type=int,   default=0)
parser.add_argument('--t_inicio', type=float, default=None)
args = parser.parse_args()

dir_fichas  = args.fichas
dir_errores = args.errores
dir_guesses = args.guesses
dir_mejores = args.mejores
run_num     = args.run
t_inicio    = args.t_inicio if args.t_inicio is not None else time.time()


# =============================================================================
# CONFIGURATION
# =============================================================================

ruta_cfg_calibrado  = "../0.-Configuracion/cfg_calibrado.yaml"
ruta_cfg_parametros = "../0.-Configuracion/cfg_parametros.yaml"

with open(ruta_cfg_calibrado, 'r') as f:
    cfg = yaml.safe_load(f)

with open(ruta_cfg_parametros, 'r') as f:
    cfg_params = yaml.safe_load(f)

nombre_parametros = list(cfg_params.keys())
cI           = np.array([cfg_params[x]['minimo'] for x in nombre_parametros])
cS           = np.array([cfg_params[x]['maximo'] for x in nombre_parametros])
n_parametros = len(nombre_parametros)


# =============================================================================
# CONSTANTS (hard-coded)
# =============================================================================

XATOL            = 1e-4   # parameter tolerance
FATOL            = 1e-2   # objective tolerance
MAXITER_NM       = 2000   # max iterations per NM run
PERTURBACION     = 0.07   # fraction of range used to perturb the simplex
INTERVALO_ESPERA = 10     # seconds between polls while NM waits for PSO
PERIODO_PRINT    = 200    # NM prints status every this many evaluations

BOUNDS = list(zip(cI, cS))  # scipy handles boundaries internally

FIN_file     = os.path.join(dir_fichas, 'FIN')
PSO_FIN_file = os.path.join(dir_fichas, 'PSO_FIN')


# =============================================================================
# GLOBAL NM EVALUATION COUNTER
# =============================================================================

conta_eval_nm = [0]


# =============================================================================
# OBJECTIVE FUNCTION — communicates with simulator via files
# =============================================================================

def evaluar(params):
    ID    = generar_ID('NM')
    ficha = {
        'Identificador': ID,
        'Semilla':       int(np.random.randint(2147483647)),
    }
    for nombre, valor in zip(nombre_parametros, params):
        ficha[nombre] = float(valor)

    ruta_ficha = os.path.join(dir_fichas, ID + '.yaml')
    with open(ruta_ficha, 'w') as f:
        yaml.dump(ficha, f, default_flow_style=False, allow_unicode=True)

    # Wait for simulator result
    ruta_error = os.path.join(dir_errores, ID + '.error')
    while not os.path.exists(ruta_error):
        if os.path.exists(FIN_file):
            raise ParadaForzada()
        time.sleep(0.05)

    errores = leer_fichero_error(ruta_error)

    conta_eval_nm[0] += 1
    if conta_eval_nm[0] % PERIODO_PRINT == 0:
        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))
        print(f"  NM  Run {run_num:02d}  [{elapsed}]  Eval NM {conta_eval_nm[0]}  —  actual: {errores[0]:.6f}")

    return errores[0]


# =============================================================================
# READ PSO GLOBAL BEST FOR THIS RUN
# =============================================================================

def leer_mejor_pso():
    patron   = os.path.join(dir_mejores, f"run_{run_num:02d}_*_gb.yaml")
    ficheros = glob.glob(patron)
    if not ficheros:
        return None, None

    mejor_error  = float('inf')
    mejor_params = None

    for fichero in ficheros:
        try:
            with open(fichero, 'r') as f:
                datos = yaml.safe_load(f)
            error = datos.get('Error_1', float('inf'))
            if error < mejor_error:
                mejor_error  = error
                mejor_params = np.array([datos[nombre] for nombre in nombre_parametros])
        except Exception:
            continue

    return mejor_params, mejor_error


# =============================================================================
# SIMPLEX CONSTRUCTION
# =============================================================================

def construir_simplex(x0):
    simplex    = np.zeros((n_parametros + 1, n_parametros))
    simplex[0] = x0
    for i in range(n_parametros):
        vertice      = x0.copy()
        rango        = cS[i] - cI[i]
        perturbacion = PERTURBACION * rango * np.random.uniform(0.5, 1.5)
        signo        = 1 if np.random.random() < 0.5 else -1
        vertice[i]   = np.clip(x0[i] + signo * perturbacion, cI[i], cS[i])
        simplex[i + 1] = vertice
    return simplex


# =============================================================================
# SAVE FINAL RESULT TO MEJORES
# =============================================================================

def guardar_en_mejores(params, error):
    from datetime import datetime
    datos = {
        'Run':       run_num,
        'Origen':    'Nelder-Mead final',
        'Timestamp': datetime.now().isoformat(),
    }
    for nombre, valor in zip(nombre_parametros, params):
        datos[nombre] = float(valor)
    datos['Error_1'] = float(error)

    nombre_fichero = f"run_{run_num:02d}_final_nm.yaml"
    ruta = os.path.join(dir_mejores, nombre_fichero)
    with open(ruta, 'w') as f:
        yaml.dump(datos, f, default_flow_style=False, allow_unicode=True)
    return ruta


# =============================================================================
# FINAL ADJUSTMENT (runs when PSO finishes)
# =============================================================================

def ultimo_ajuste():
    params_inicio, error_inicio = leer_mejor_pso()
    if params_inicio is None:
        print(f"  NM  Run {run_num:02d}  Final adjustment: no PSO result available.")
        return

    elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))
    print(f"  NM  Run {run_num:02d}  [{elapsed}]  Final adjustment  —  PSO error: {error_inicio:.6f}")

    simplex = construir_simplex(params_inicio)

    try:
        resultado = minimize(
            evaluar,
            params_inicio,
            method='Nelder-Mead',
            bounds=BOUNDS,
            options={
                'xatol':           XATOL,
                'fatol':           FATOL,
                'maxiter':         MAXITER_NM,
                'initial_simplex': simplex,
            },
        )

        params_nm = resultado.x
        error_nm  = resultado.fun

        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))
        print(f"  NM  Run {run_num:02d}  [{elapsed}]  Final adjustment done  —  NM error: {error_nm:.6f}")

        if error_nm < error_inicio:
            ruta = guardar_en_mejores(params_nm, error_nm)
            print(f"  NM  Run {run_num:02d}  Improvement saved: {os.path.basename(ruta)}")
        else:
            print(f"  NM  Run {run_num:02d}  PSO result not improved.")

    except ParadaForzada:
        print(f"\n  NM  Run {run_num:02d}  Final adjustment interrupted by FIN.")


# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"\n  NM  Run {run_num:02d}  Starting...")

# NM starts as soon as PSO has any result and restarts when PSO beats NM's best
error_referencia = float('inf')

while True:

    # PSO has finished -> final adjustment and stop
    if os.path.exists(PSO_FIN_file):
        ultimo_ajuste()
        with open(FIN_file, 'w') as f:
            f.write('NM finished\n')
        break

    # Forced stop (Ctrl+C from the launcher)
    if os.path.exists(FIN_file):
        print(f"\n  NM  Run {run_num:02d}  FIN file detected. Stopping.")
        break

    params_inicio, error_inicio = leer_mejor_pso()

    # PSO has no results yet
    if params_inicio is None:
        time.sleep(INTERVALO_ESPERA)
        continue

    # PSO has not improved upon NM's best
    if error_inicio >= error_referencia:
        time.sleep(INTERVALO_ESPERA)
        continue

    # New starting point available -> launch NM
    elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))
    print(f"  NM  Run {run_num:02d}  [{elapsed}]  New start  —  PSO error: {error_inicio:.6f}")

    simplex = construir_simplex(params_inicio)

    try:
        resultado = minimize(
            evaluar,
            params_inicio,
            method='Nelder-Mead',
            bounds=BOUNDS,
            options={
                'xatol':           XATOL,
                'fatol':           FATOL,
                'maxiter':         MAXITER_NM,
                'initial_simplex': simplex,
            },
        )

        params_nm = resultado.x
        error_nm  = resultado.fun

        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))
        print(f"  NM  Run {run_num:02d}  [{elapsed}]  Converged   —  NM error: {error_nm:.6f}  (start: {error_inicio:.6f})")

        # Wait until someone beats NM's result
        error_referencia = error_nm

        # Deposit in GUESSES only if NM improved the starting point
        if error_nm < error_inicio:
            escribir_guess(params_nm, nombre_parametros, dir_guesses, 'NM')

    except ParadaForzada:
        print(f"\n  NM  Run {run_num:02d}  Forced stop during evaluation.")
        break


# =============================================================================
# SHUTDOWN
# =============================================================================

print(f"\n{'='*60}")
print(f"  NM  Run {run_num:02d}  Total NM evaluations: {conta_eval_nm[0]}")
print(f"{'='*60}\n")

# END -------------------------------------------------------------------------
