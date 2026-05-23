# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__   = "26 de abril de 2026"

"""
1.-Mejor_ajuste_determinista.py
--------------------------------
Reads the best fit from evaluaciones_global.db, runs the SLIR model,
and exports an Excel workbook with three sheets:
    - Reported : observed vs model-reported cases (per week, 33 weeks)
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

N     = 4_950_738
_mu   = 41_157 / 52 / 7   # births per day (persons/day)
d_dia = _mu / N            # daily mortality rate (1/day)

# -- Configuration ------------------------------------------------------------
with open(RUTA_PARAMS) as f:
    cfg_params = yaml.safe_load(f)
nombre_parametros = list(cfg_params.keys())

# -- Observed data (weekly reported cases, absolute persons) ------------------
# Source: IS-150, Generalitat Valenciana, 2016-2017 influenza season
reported_por_100k = np.array([
     2.17,  2.72, 12.36,  3.94, 12.64, 12.50,  9.92, 23.51, 22.96, 10.33,
    20.92, 23.10, 53.53, 63.86,122.69,135.19,166.71,134.10,124.18, 87.77,
    63.45, 39.54, 22.83,  9.78,  7.20,  5.98,  3.94,  2.45,  4.21,  2.04,
     1.36,  2.58,  0.41
])
DATOS_OBS = reported_por_100k * N / 100_000
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

mu_dia = N * d_dia
r0_ini = 1.0 - s0 - l0 - i0

def ode(t, y):
    s, l, i, r, c = y
    return [
        mu_dia - (beta_dia * i / N + d_dia) * s,
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
    'Week':             semanas,
    'Observed data':    DATOS_OBS,
    'Model reported':   reportados_sem,
})

df_parametros = pd.DataFrame(
    [{'Parameter': k, 'Value': v} for k, v in params_dict.items()] +
    [{'Parameter': 'error_1',           'Value': error},
     {'Parameter': 'R0',                'Value': R0},
     {'Parameter': '% unique infected', 'Value': pct_inf}]
)

df_outputs = pd.DataFrame({
    'Week':         semanas,
    'Susceptible':  S_sem,
    'Latent':       L_sem,
    'Infectious':   I_sem,
    'Recovered':    R_sem,
})

# -- Export Excel -------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    df_reportados.to_excel(writer, sheet_name='Reported',   index=False)
    df_parametros.to_excel(writer, sheet_name='Parameters', index=False)
    df_outputs.to_excel(writer,    sheet_name='Outputs',    index=False)

# END -------------------------------------------------------------------------
