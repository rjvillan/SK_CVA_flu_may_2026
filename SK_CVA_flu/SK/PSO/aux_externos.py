# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__ = "26 de abril de 2026"

"""
aux_externos.py
---------------
Auxiliary functions external to the PSO core.
"""

import os
import numpy as np
import yaml
from datetime import datetime


# =============================================================================
# ID GENERATION
# =============================================================================

def generar_ID(prefijo):
    """
    Generates a unique ID with the given prefix.

    For PSO: prefix = id_instancia (fixed per session,
             distinguishes parallel instances).
    For NM/SUR: prefix = 'NM' or 'SUR'.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return f"{prefijo}_{timestamp}"


# =============================================================================
# GUESS WRITING
# =============================================================================

def escribir_guess(parametros_array, nombre_parametros, dir_guesses, prefijo):
    """
    Writes an external guess as a YAML file in dir_guesses/.

    Parameters:
        parametros_array  : numpy array with the optimal parameters found
        nombre_parametros : list of parameter names
        dir_guesses       : directory where the file is written
        prefijo           : 'NM' or 'SUR', identifies the origin

    Returns:
        path of the written file
    """
    origen = 'Nelder-Mead' if prefijo == 'NM' else 'Surrogate'

    ficha = {}
    ficha['Identificador'] = generar_ID(prefijo)
    ficha['Semilla']       = int(np.random.randint(2147483647))
    ficha['Origen']        = origen

    for nombre, valor in zip(nombre_parametros, parametros_array):
        ficha[nombre] = float(valor)

    nombre_fichero = ficha['Identificador'] + '.yaml'
    ruta = os.path.join(dir_guesses, nombre_fichero)

    with open(ruta, 'w') as f:
        yaml.dump(ficha, f, default_flow_style=False, allow_unicode=True)

    return ruta


# END -------------------------------------------------------------------------
