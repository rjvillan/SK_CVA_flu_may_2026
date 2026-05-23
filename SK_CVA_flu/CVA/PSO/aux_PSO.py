# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__ = "26 de abril de 2026"

"""
aux_PSO.py
----------
Auxiliary functions for the hybrid PSO calibration system.
Independent of the model being calibrated and reusable across projects.

Contents:
    - YAML particle generation and reading
    - Random numbers (al)
    - Search-space boundary handling (ajustar_bordes)
    - Polynomial mutation (mutacion)
    - Pareto dominance and front (dominado, FPareto)
    - Best-set management (mejores_iguales, actualiza_mejor, gbest_2_yaml)
    - Nested-list flattening (flatten)
"""

import numpy as np
import os
import time
import yaml

# =============================================================================
# PARTICLE GENERATION AND READING
# =============================================================================

def Genera_Ficha(p, dir_fichas, nombre_parametros):
    """
    Writes a YAML particle file to dir_fichas/.

    The file contains:
        - Identificador: unique evaluation ID
        - Semilla: always present (the simulator decides whether to use it)
        - One field per parameter with its name and value

    Parameters:
        p                 : Particle object with ID, semilla and parametros
        dir_fichas        : directory where the file is written
        nombre_parametros : list of parameter names
    """
    ficha = {}
    ficha['Identificador'] = p.ID
    ficha['Semilla']       = int(p.semilla)

    for nombre, valor in zip(nombre_parametros, p.parametros):
        ficha[nombre] = float(valor)

    ruta = os.path.join(dir_fichas, p.ID + '.yaml')
    with open(ruta, 'w') as f:
        yaml.dump(ficha, f, default_flow_style=False, allow_unicode=True)


def leer_fichero_error(ruta_error):
    """
    Reads an error YAML file and returns the list of errors (fitness).

    Error file format:
        Errores:
          - 0.4523
          - 0.3812

    Retries in a loop in case the simulator is still writing the file
    when PSO tries to read it.

    Parameters:
        ruta_error: full path to the .error file

    Returns:
        list of floats with errors (one per objective)
    """
    while True:
        try:
            with open(ruta_error, 'r') as f:
                contenido = yaml.safe_load(f)

            errores = contenido['Errores']

            if not isinstance(errores, list):
                errores = [errores]

            return errores

        except Exception:
            # Retry if the simulator has not finished writing yet
            time.sleep(0.1)


# =============================================================================
# RANDOM NUMBERS
# =============================================================================

def al(a=0, b=1, c=1):
    """
    Returns c uniform random numbers in [a, b].

    Typical uses:
        al()           -> one number in [0, 1]
        al(0.25, 0.75) -> one number in [0.25, 0.75]
        al(cI, cS, n)  -> vector of n numbers, one per component
    """
    return np.array(a) + np.random.random(c) * (np.array(b) - np.array(a))


# =============================================================================
# SEARCH-SPACE BOUNDARY HANDLING
# =============================================================================

def ajustar_bordes(v, cI, cS, umbral=0.01):
    """
    Adjusts parameters that are outside or near the boundary of the
    search space. For each component that needs adjustment, randomly
    chooses between two strategies:

        - Random reset: replaces by a uniform value in [cI, cS]
        - Reflection: bounces symmetrically back inside the range

    The per-component random choice introduces diversity and avoids
    having to fix a single boundary strategy.

    Parameters:
        v      : parameter vector (array)
        cI     : lower bounds (array)
        cS     : upper bounds (array)
        umbral : fraction of range considered "near the boundary" (default 1%)

    Returns:
        adjusted vector as a list
    """
    v  = np.array(v,  dtype=float)
    cI = np.array(cI, dtype=float)
    cS = np.array(cS, dtype=float)

    margen = (cS - cI) * umbral

    for i in range(len(v)):

        fuera       = v[i] < cI[i] or v[i] > cS[i]
        cerca_borde = (abs(v[i] - cI[i]) < margen[i]) or (abs(v[i] - cS[i]) < margen[i])

        if fuera or cerca_borde:

            if np.random.random() < 0.5:
                # Strategy 1: random reset
                v[i] = np.random.uniform(cI[i], cS[i])
            else:
                # Strategy 2: reflection with bounces
                max_rebotes = 10
                rebotes = 0
                while (v[i] < cI[i] or v[i] > cS[i]) and rebotes < max_rebotes:
                    if v[i] < cI[i]:
                        v[i] = 2 * cI[i] - v[i]
                    if v[i] > cS[i]:
                        v[i] = 2 * cS[i] - v[i]
                    rebotes += 1
                # If still out of bounds after bounces, random reset
                if v[i] < cI[i] or v[i] > cS[i]:
                    v[i] = np.random.uniform(cI[i], cS[i])

    return v.tolist()


# =============================================================================
# POLYNOMIAL MUTATION
# =============================================================================

def mutacion(vec, cI, cS, tasa_mut=0.1, dist_mut=20):
    """
    Applies polynomial mutation to the parameter vector.

    Each component has probability tasa_mut of being mutated.
    The mutation magnitude follows a polynomial distribution
    controlled by dist_mut (higher values -> smaller mutations).

    Parameters:
        vec      : parameter vector to mutate (array)
        cI       : lower bounds (array)
        cS       : upper bounds (array)
        tasa_mut : probability of mutating each component (default 0.1)
        dist_mut : mutation distribution index (default 20)

    Returns:
        mutated vector as an array
    """
    hijo = np.array(vec, dtype=float)
    n    = len(hijo)

    for i in range(n):
        if np.random.random() < tasa_mut:
            r = np.random.random()
            # Delta following polynomial distribution
            if r < 0.5:
                delta = -1 + (2 * r) ** (1.0 / (dist_mut + 1))
            else:
                delta =  1 - (2 * (1 - r)) ** (1.0 / (dist_mut + 1))

            hijo[i] = hijo[i] + delta * (cS[i] - cI[i])

    # Ensure we stay in bounds after mutation
    hijo = ajustar_bordes(hijo, cI, cS)

    return hijo


# =============================================================================
# PARETO DOMINANCE AND FRONT
# =============================================================================

def dominado(f1, f2):
    """
    Returns True if f1 Pareto-dominates f2.

    f1 dominates f2 if:
        - f1 is less than or equal to f2 in all components, AND
        - f1 is strictly less than f2 in at least one component.

    (Minimisation: smaller values are better.)
    """
    mejor_o_igual_en_todo = np.all(np.less_equal(f1, f2))
    estrictamente_mejor_en_algo = np.any(np.less(f1, f2))
    return mejor_o_igual_en_todo and estrictamente_mejor_en_algo


def FPareto(lista_fitness):
    """
    Computes the non-dominated Pareto front of a list of fitness vectors.

    Efficient version: for each vector, checks whether any other vector
    dominates it. As soon as one is found, the vector is discarded without
    comparing the rest.

    Parameters:
        lista_fitness: list of fitness vectors (list of lists or arrays)

    Returns:
        list of indices of non-dominated vectors
    """
    n      = len(lista_fitness)
    frente = []

    for i in range(n):
        esta_dominado = False

        for j in range(n):
            if i == j:
                continue
            if dominado(lista_fitness[j], lista_fitness[i]):
                esta_dominado = True
                break   # No need to keep looking

        if not esta_dominado:
            frente.append(i)

    return frente


# =============================================================================
# BEST-SET MANAGEMENT
# =============================================================================


def mejores_iguales(lista1, lista2):
    """
    Checks whether two fitness lists are equal as sets.

    Two lists are equal if they have the same size and every element
    of one appears in the other.

    Parameters:
        lista1, lista2: lists of fitness vectors

    Returns:
        True if the lists represent the same set of fitness values
    """
    if len(lista1) != len(lista2):
        return False

    for x in lista1:
        if not any(x == y for y in lista2):
            return False

    for y in lista2:
        if not any(y == x for x in lista1):
            return False

    return True


def actualiza_mejor(mejor_actual, datos_nueva_simulacion):
    """
    Updates the best set (local or global) with the latest simulation data.

    Adds the new simulation to the current set, recomputes the Pareto front,
    and keeps only non-dominated solutions.

    Parameters:
        mejor_actual           : Particle object with lists of ID,
                                 parametros and fitness of current bests
        datos_nueva_simulacion : list [ID, parametros, fitness]
                                 of the just-evaluated simulation

    Returns:
        updated mejor_actual
        hay_mejora: True if the Pareto front changed
    """
    nuevo_ID, nuevos_parametros, nuevo_fitness = datos_nueva_simulacion

    frente_antes = [x for x in mejor_actual.fitness]

    mejor_actual.ID.append(nuevo_ID)
    mejor_actual.parametros.append(nuevos_parametros)
    mejor_actual.fitness.append(nuevo_fitness)

    indices_frente = FPareto(mejor_actual.fitness)

    mejor_actual.ID         = [mejor_actual.ID[i]         for i in indices_frente]
    mejor_actual.parametros = [mejor_actual.parametros[i] for i in indices_frente]
    mejor_actual.fitness    = [mejor_actual.fitness[i]    for i in indices_frente]

    frente_despues = [x for x in mejor_actual.fitness]
    hay_mejora = not mejores_iguales(frente_antes, frente_despues)

    return mejor_actual, hay_mejora


def gbest_2_yaml(gbest, nombre_parametros, dir_mejores, conta_evaluaciones, run_num=0):
    """
    Saves current global bests to YAML files in dir_mejores/.

    File name: run_XX_NNNNNN_gb.yaml (single-objective)
               run_XX_NNNNNN_gb_kk.yaml (multi-objective, one file per Pareto solution)

    Parameters:
        gbest              : Particle object with global bests
        nombre_parametros  : list of parameter names
        dir_mejores        : directory where files are saved
        conta_evaluaciones : number of completed evaluations
        run_num            : current run number

    Returns:
        list of paths of generated files
    """
    rutas_generadas = []
    multi = len(gbest.ID) > 1

    for k, gb_id in enumerate(gbest.ID):

        mejor = {}
        mejor['Run']           = run_num
        mejor['Evaluacion']    = conta_evaluaciones
        mejor['Identificador'] = gb_id

        for nombre, valor in zip(nombre_parametros, gbest.parametros[k]):
            mejor[nombre] = float(valor)

        for i, error in enumerate(gbest.fitness[k]):
            mejor[f'Error_{i+1}'] = float(error)

        if multi:
            nombre_fichero = f"run_{run_num:02d}_{conta_evaluaciones:06d}_gb_{k:02d}.yaml"
        else:
            nombre_fichero = f"run_{run_num:02d}_{conta_evaluaciones:06d}_gb.yaml"

        ruta = os.path.join(dir_mejores, nombre_fichero)

        with open(ruta, 'w') as f:
            yaml.dump(mejor, f, default_flow_style=False, allow_unicode=True)

        rutas_generadas.append(ruta)

    return rutas_generadas


# END --------------------------------------------------------------------------
