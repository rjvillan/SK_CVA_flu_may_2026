# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__   = "26 de abril de 2026"

"""
1.-Mejor_ajuste_determinista.py
--------------------------------
Reads the best fit from evaluaciones_global.db, runs the SLIR model,
and exports an Excel workbook with three sheets:
    - Reported  : observed vs model-reported cases (per week, 53 weeks)
    - Parameters: best-fit parameters + error + R0 + % unique infected
    - Outputs   : S, L, I, R at the end of each week
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import yaml

# -- Paths --------------------------------------------------------------------
BASE        = os.path.dirname(os.path.abspath(__file__))
RUTA_DB     = os.path.join(BASE, '..', 'RESULTADOS', 'evaluaciones_global.db')
RUTA_PARAMS = os.path.join(BASE, '..', '0.-Configuracion', 'cfg_parametros.yaml')
OUTPUT_XLSX = os.path.join(BASE, 'Mejor_ajuste.xlsx')

N     = 51_345_000
_mu   = 381_950 / 52 / 7          # births per day (persons/day)
d_dia = 5.55 / 1000 / 52 / 7     # daily mortality rate (1/day)

# -- Configuration ------------------------------------------------------------
with open(RUTA_PARAMS) as f:
    cfg_params = yaml.safe_load(f)
nombre_parametros = list(cfg_params.keys())

# -- Observed data (Korea 2016-2017, ILI per 1,000, weeks 36-35) -------------
reported = np.array([
     3.8,  4.6,  3.9,  3.9,  4.2,  3.7,  3.7,  4.0,  4.8,  4.2,
     4.5,  5.9,  7.3, 13.3, 34.8, 61.8, 86.2, 63.5, 39.4, 23.9,
    17.0, 12.5,  9.9,  9.0,  7.1,  6.7,  6.1,  7.0,  9.3, 13.2,
    13.7, 16.7, 15.8, 14.5, 13.3,  9.5,  6.8,  7.6,  6.7,  4.9,
     5.1,  5.6,  5.7,  5.3,  5.8,  6.2,  6.3,  4.3,  4.5,  4.4,
     3.8,  5.2,  4.8
]) * N / 1000
DATOS_OBS = reported
N_SEMANAS = len(DATOS_OBS)

# -- Read best fit from database ----------------------------------------------
conn  = sqlite3.connect(RUTA_DB)
cols  = ', '.join(nombre_parametros + ['error_1'])
fila  = conn.execute(
    f"SELECT {cols} FROM evaluaciones ORDER BY error_1 ASC LIMIT 1"
).fetchone()
conn.close()

if fila is None:
    print("Database is empty.")
    sys.exit(0)

params_dict = dict(zip(nombre_parametros, fila[:-1]))
error       = fila[-1]

# -- Run the model ------------------------------------------------------------
beta_dia    = params_dict['beta']    / 7.0
epsilon_dia = params_dict['epsilon'] / 7.0
gamma_dia   = params_dict['gamma']   / 7.0
s0 = params_dict['s0']
l0 = params_dict['l0']
i0 = params_dict['i0']
p  = params_dict['p']
a  = params_dict['a']

r0_ini = 1.0 - s0 - l0 - i0

def ode(t, y):
    s, l, i, r, c = y
    return [
        _mu - (beta_dia * i / N + d_dia) * s,
        beta_dia * s * i / N - (epsilon_dia + d_dia) * l,
        epsilon_dia * l - (gamma_dia + d_dia) * i,
        gamma_dia * i - d_dia * r,
        epsilon_dia * l,
    ]

y0     = [s0 * N, l0 * N, i0 * N, r0_ini * N, 0.0]
t_eval = np.arange(7, N_SEMANAS * 7 + 1, 7, dtype=float)

sol = solve_ivp(ode, [0.0, N_SEMANAS * 7], y0,
                method='RK45', t_eval=t_eval,
                rtol=1e-8, atol=1e-10, dense_output=False)

S_sem = sol.y[0]
L_sem = sol.y[1]
I_sem = sol.y[2]
R_sem = sol.y[3]
infectados_unicos = np.diff(np.concatenate([[0.0], sol.y[4]]))
reportados_sem    = a + p * infectados_unicos

# -- Derived indicators -------------------------------------------------------
R0      = beta_dia / (gamma_dia + d_dia)
pct_inf = infectados_unicos.sum() / N * 100

# -- Build DataFrames ---------------------------------------------------------
semanas = list(range(1, N_SEMANAS + 1))

df_reportados = pd.DataFrame({
    'Week':           semanas,
    'Observed data':  DATOS_OBS,
    'Model reported': reportados_sem,
})

df_parametros = pd.DataFrame(
    [{'Parameter': k, 'Value': v} for k, v in params_dict.items()] +
    [{'Parameter': 'error_1',           'Value': error},
     {'Parameter': 'R0',                'Value': R0},
     {'Parameter': '% unique infected', 'Value': pct_inf}]
)

df_outputs = pd.DataFrame({
    'Week':        semanas,
    'Susceptible': S_sem,
    'Latent':      L_sem,
    'Infectious':  I_sem,
    'Recovered':   R_sem,
})

# -- Export Excel -------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    df_reportados.to_excel(writer, sheet_name='Reported',   index=False)
    df_parametros.to_excel(writer, sheet_name='Parameters', index=False)
    df_outputs.to_excel(writer,    sheet_name='Outputs',    index=False)

# END -------------------------------------------------------------------------
