# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__   = "26 de abril de 2026"

"""
gripe_slir.py
-------------
SLIR model simulator with demography for the PSO calibration system.

Integrates the model using scipy.integrate.solve_ivp (RK45, rtol=1e-8, atol=1e-10).

Parameters to calibrate (weekly units as specified in cfg_parametros.yaml):
    beta     : transmission rate (1/week)
    epsilon  : L->I progression rate (1/week)
    gamma    : I->R recovery rate (1/week)
    s0       : initial susceptible fraction
    l0       : initial latent fraction
    i0       : initial infectious fraction
    p        : ILI/infectious ratio (>1 because ILI includes non-influenza causes)
    a        : additive baseline of weekly ILI cases

Note: beta, epsilon and gamma are divided by 7 internally to convert
      from weekly to daily units before integration.

Error function:
    error = RMSE(model_reported, observed_data)
           + proportional penalty if R0 outside [1.0, 3.0]
           + proportional penalty if unique_infected outside [5%, 20%] of N

Usage:
    Independent process that monitors FICHAS/ and processes each particle file.
    python gripe_slir.py

    Also importable as a module for Nelder-Mead:
    from gripe_slir import simular
"""

import argparse
import glob
import os
import sqlite3
import time
import numpy as np
from datetime import datetime

import yaml
from scipy.integrate import solve_ivp

N     = 51_345_000
_mu   = 381_950 / 52 / 7          # births per day (persons/day)
d_dia = 5.55 / 1000 / 52 / 7     # daily mortality rate (1/day)


# =============================================================================
# OBSERVED DATA
# =============================================================================

# Korea 2016-2017. ILI per 1,000 sentinel visits, weeks 36-35
reported = np.array([
     3.8,  4.6,  3.9,  3.9,  4.2,  3.7,  3.7,  4.0,  4.8,  4.2,
     4.5,  5.9,  7.3, 13.3, 34.8, 61.8, 86.2, 63.5, 39.4, 23.9,
    17.0, 12.5,  9.9,  9.0,  7.1,  6.7,  6.1,  7.0,  9.3, 13.2,
    13.7, 16.7, 15.8, 14.5, 13.3,  9.5,  6.8,  7.6,  6.7,  4.9,
     5.1,  5.6,  5.7,  5.3,  5.8,  6.2,  6.3,  4.3,  4.5,  4.4,
     3.8,  5.2,  4.8
]) * N / 1000

DATOS_OBS = reported                          # absolute persons
N_SEMANAS = len(DATOS_OBS)                    # 53 weeks
RMS_OBS   = float(np.sqrt(np.mean(DATOS_OBS ** 2)))   # reference scale


# =============================================================================
# EPIDEMIOLOGICAL CONSTRAINTS
# =============================================================================

R0_MIN  = 1.0
R0_MAX  = 3.0
INF_MIN = 0.05 * N   # minimum unique infected (persons)
INF_MAX = 0.20 * N   # maximum unique infected (persons)


# =============================================================================
# CONFIGURATION
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fichas',  required=True)
parser.add_argument('--errores', required=True)
parser.add_argument('--db',      required=True)
_args = parser.parse_args()

dir_fichas  = _args.fichas
dir_errores = _args.errores
ruta_db     = _args.db

ruta_cfg_parametros = "../0.-Configuracion/cfg_parametros.yaml"
with open(ruta_cfg_parametros, 'r') as f:
    cfg_params = yaml.safe_load(f)
nombre_parametros = list(cfg_params.keys())


# =============================================================================
# CORE FUNCTION: simular(parametros_dict) -> dict
# =============================================================================

def simular(parametros: dict) -> dict:
    """
    Core simulator function. Receives a parameter dictionary and returns
    a dictionary with the error values.

    Interface contract:
        - Input:  {parameter_name: value, 'Semilla': int, ...}
        - Output: {'Errores': [total_rmse]}

    Epidemiological parameters are in weekly units — converted to daily
    internally before RK45 integration.

    Returns a large penalty if the parameters produce an epidemiologically
    incoherent solution.
    """

    try:
        beta    = parametros['beta']
        epsilon = parametros['epsilon']
        gamma   = parametros['gamma']
        s0      = parametros['s0']
        l0      = parametros['l0']
        i0      = parametros['i0']
        p       = parametros['p']
        a       = parametros['a']

        # 1. Initial conditions: enforce s0 + l0 + i0 <= 1
        suma = s0 + l0 + i0
        if suma > 1.0:
            pen_cIni = (suma - 1) * N
            s0 = s0 / suma
            l0 = l0 / suma
            i0 = i0 / suma
        else:
            pen_cIni = 0.0

        # Convert weekly rates to daily for the integrator
        beta_dia    = beta    / 7.0
        epsilon_dia = epsilon / 7.0
        gamma_dia   = gamma   / 7.0

        def ode(t, y):
            s, l, i, r, c = y
            ds = _mu - (beta_dia * i / N + d_dia) * s
            dl = beta_dia * s * i / N - (epsilon_dia + d_dia) * l
            di = epsilon_dia * l - (gamma_dia + d_dia) * i
            dr = gamma_dia * i - d_dia * r
            dc = epsilon_dia * l
            return [ds, dl, di, dr, dc]

        r0_ini = 1.0 - s0 - l0 - i0
        y0     = [s0 * N, l0 * N, i0 * N, r0_ini * N, 0.0]
        t_eval = np.arange(7, N_SEMANAS * 7 + 1, 7, dtype=float)

        sol = solve_ivp(ode, [0.0, N_SEMANAS * 7], y0,
                        method='RK45', t_eval=t_eval,
                        rtol=1e-8, atol=1e-10, dense_output=False)

        if not sol.success or sol.y.shape[1] != N_SEMANAS:
            return {'Errores': [1e10]}

        infectados_unicos = np.diff(np.concatenate([[0.0], sol.y[4]]))
        reportados_sem    = a + p * infectados_unicos

        # 2. RMSE between model-reported and observed cases
        rmse = np.sqrt(np.mean((reportados_sem - DATOS_OBS) ** 2))

        # 3. Penalty for R0 outside [R0_MIN, R0_MAX]
        R0 = beta_dia / (gamma_dia + d_dia)

        if R0 < R0_MIN:
            pen_R0 = (R0_MIN - R0) * RMS_OBS
        elif R0 > R0_MAX:
            pen_R0 = (R0 - R0_MAX) * RMS_OBS
        else:
            pen_R0 = 0.0

        # 4. Penalty for unique infected outside [INF_MIN, INF_MAX] (persons)
        inf_total = infectados_unicos.sum()

        if inf_total < INF_MIN:
            pen_inf = INF_MIN - inf_total
        elif inf_total > INF_MAX:
            pen_inf = inf_total - INF_MAX
        else:
            pen_inf = 0.0

        error_total = rmse + pen_cIni + pen_R0 + pen_inf

        return {
            'Errores':           [float(error_total)],
            'rmse':              float(rmse),
            'R0':                float(R0),
            'pen_cIni':          float(pen_cIni),
            'pen_R0':            float(pen_R0),
            'inf_total':         float(inf_total),
            'pen_infectados':    float(pen_inf),
            'reportados_modelo': reportados_sem.tolist()
        }

    except Exception:
        return {'Errores': [1e10]}


# =============================================================================
# DATABASE STORAGE
# =============================================================================

def guardar_en_db(ficha: dict, resultado: dict):
    """
    Saves one evaluation to evaluaciones.db.

    Parameters:
        ficha    : dictionary read from the YAML particle file
        resultado: dictionary returned by simular()
    """
    try:
        conn   = sqlite3.connect(ruta_db)
        cursor = conn.cursor()

        fila = {
            'id':        ficha['Identificador'],
            'semilla':   ficha.get('Semilla', None),
            'timestamp': datetime.now().isoformat()
        }

        for nombre in nombre_parametros:
            fila[nombre] = ficha.get(nombre, None)

        errores = resultado['Errores']
        for i, error in enumerate(errores):
            fila[f'error_{i+1}'] = error

        columnas = ', '.join(fila.keys())
        valores  = ', '.join(['?' for _ in fila])
        cursor.execute(
            f"INSERT OR REPLACE INTO evaluaciones ({columnas}) VALUES ({valores})",
            list(fila.values())
        )
        conn.commit()
        conn.close()

    except Exception:
        pass


# =============================================================================
# PARTICLE FILE PROCESSING
# =============================================================================

def procesar_ficha(ruta_ficha: str):
    """
    Processes a complete YAML particle file:
        1. Read the file
        2. Call simular()
        3. Write the .error file to dir_errores/
        4. Save to evaluaciones.db
        5. Delete the processed file

    Parameters:
        ruta_ficha: full path to the .yaml particle file
    """
    try:
        with open(ruta_ficha, 'r') as f:
            ficha = yaml.safe_load(f)

        identificador = ficha['Identificador']

        resultado = simular(ficha)

        error_yaml = {'Errores': resultado['Errores']}
        ruta_error = os.path.join(dir_errores, identificador + '.error')
        with open(ruta_error, 'w') as f:
            yaml.dump(error_yaml, f, default_flow_style=False)

        guardar_en_db(ficha, resultado)

        os.remove(ruta_ficha)

    except Exception:
        pass


# =============================================================================
# MAIN LOOP
# =============================================================================

if __name__ == "__main__":

    try:
        while True:

            if os.path.exists(os.path.join(dir_fichas, 'FIN')):
                break

            fichas = sorted(glob.glob(os.path.join(dir_fichas, '*.yaml')))

            for ruta_ficha in fichas:
                procesar_ficha(ruta_ficha)

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass

# END -------------------------------------------------------------------------
