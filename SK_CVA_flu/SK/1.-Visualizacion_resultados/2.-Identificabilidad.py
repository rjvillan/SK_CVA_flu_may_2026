# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__   = "11 de mayo de 2026"

"""
2.-Identificabilidad.py
------------------------
Reads the top X% best evaluations from evaluaciones_global.db
and exports an Excel file for parameter identifiability analysis.

Also generates a figure with 8 subplots (one per parameter) showing
evaluations filtered by an error threshold, ordered by error (x-axis).
An identifiable parameter shows a narrow band at the best values that
widens gradually as the error increases.
"""

import math
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# -- Paths --------------------------------------------------------------------
BASE        = os.path.dirname(os.path.abspath(__file__))
RUTA_DB     = os.path.join(BASE, '..', 'RESULTADOS', 'evaluaciones_global.db')
RUTA_PARAMS = os.path.join(BASE, '..', '0.-Configuracion', 'cfg_parametros.yaml')
OUTPUT_XLSX = os.path.join(BASE, 'Identificabilidad.xlsx')
OUTPUT_PNG  = os.path.join(BASE, 'identificabilidad.png')
OUTPUT_CORR = os.path.join(BASE, 'correlacion.png')

# -- Editable settings --------------------------------------------------------
PORCENTAJE_MEJORES = 0.02    # top 2% of records (for Excel export)
DELTA              = 0.05    # confidence region: error <= error_min * (1 + DELTA)

# -- Configuration ------------------------------------------------------------
with open(RUTA_PARAMS) as f:
    cfg_params = yaml.safe_load(f)
nombre_parametros = list(cfg_params.keys())
COL_ERROR = 'error_1'

# -- Read data from database --------------------------------------------------
conn      = sqlite3.connect(RUTA_DB)
total     = conn.execute("SELECT COUNT(*) FROM evaluaciones").fetchone()[0]
k         = max(1, math.ceil(total * PORCENTAJE_MEJORES))
error_min = conn.execute(f"SELECT MIN({COL_ERROR}) FROM evaluaciones").fetchone()[0]

UMBRAL_ERROR = error_min * (1 + DELTA)

cols = ', '.join(['id'] + nombre_parametros + [COL_ERROR])

df_top = pd.read_sql_query(
    f"SELECT {cols} FROM evaluaciones ORDER BY {COL_ERROR} ASC LIMIT {k}",
    conn
)

df_fig = pd.read_sql_query(
    f"SELECT {', '.join(nombre_parametros + [COL_ERROR])} "
    f"FROM evaluaciones WHERE {COL_ERROR} <= {UMBRAL_ERROR} "
    f"ORDER BY {COL_ERROR} ASC",
    conn
)
conn.close()

# -- Export Excel -------------------------------------------------------------
df_top.to_excel(OUTPUT_XLSX, index=False, sheet_name='Filtered', engine='openpyxl')

print(f"Total evaluations:  {total}")
print(f"Top {PORCENTAJE_MEJORES*100:.1f}%:         {k} records")
print(f"Best error:         {error_min:.6f}")
print(f"Threshold ({DELTA*100:.0f}% above optimum): {UMBRAL_ERROR:.4f}")
print(f"Evaluations within threshold: {len(df_fig)}")

# -- Identifiability figure ---------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()

x = df_fig[COL_ERROR].values

for i, nombre in enumerate(nombre_parametros):
    ax = axes[i]
    y  = df_fig[nombre].values

    ax.scatter(x, y, s=4, alpha=0.3, color='steelblue', rasterized=True)
    ax.set_title(nombre, fontsize=11, fontweight='bold')
    ax.set_xlabel('error_1', fontsize=9)
    ax.set_ylabel(nombre, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Identifiability — evaluations with error <= {UMBRAL_ERROR:.4f} '
    f'(n = {len(df_fig)})',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
plt.close()

print(f"Figure saved: {OUTPUT_PNG}")

# -- Spearman correlation figure ----------------------------------------------
corr = df_fig[nombre_parametros].corr(method='spearman')

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')

ticks = range(len(nombre_parametros))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(nombre_parametros, fontsize=10)
ax.set_yticklabels(nombre_parametros, fontsize=10)

for i in range(len(nombre_parametros)):
    for j in range(len(nombre_parametros)):
        v = corr.values[i, j]
        color = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=8, color=color)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title(
    f'Spearman correlation — error <= {UMBRAL_ERROR:.4f} (n = {len(df_fig)})',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(OUTPUT_CORR, dpi=150, bbox_inches='tight')
plt.close()

print(f"Correlation figure saved: {OUTPUT_CORR}")

# END -------------------------------------------------------------------------
