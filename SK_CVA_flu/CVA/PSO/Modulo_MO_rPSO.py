# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__ = "26 de abril de 2026"

"""
Modulo_MO_rPSO.py
-----------------
Multi-objective PSO with a repository of global bests.

The simulator is completely independent of this module.
They communicate exclusively via YAML files in dir_fichas and dir_errores.
Directories are received as command-line arguments.
"""

import os
import glob
import argparse
import time
import numpy as np
import yaml
from scipy.stats.qmc import Halton

from aux_PSO import (
    Genera_Ficha, leer_fichero_error,
    al, ajustar_bordes, mutacion,
    actualiza_mejor, gbest_2_yaml,
)
from aux_externos import generar_ID


# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fichas',   required=True)
parser.add_argument('--errores',  required=True)
parser.add_argument('--mejores',  required=True)
parser.add_argument('--guesses',  default='')
parser.add_argument('--run',      type=int,   default=0)
parser.add_argument('--t_inicio', type=float, default=None)
args = parser.parse_args()

dir_fichas  = args.fichas
dir_errores = args.errores
dir_mejores = args.mejores
dir_guesses = args.guesses
run_num     = args.run


# =============================================================================
# CONFIGURATION
# =============================================================================

ruta_cfg_calibrado  = "../0.-Configuracion/cfg_calibrado.yaml"
ruta_cfg_parametros = "../0.-Configuracion/cfg_parametros.yaml"

with open(ruta_cfg_calibrado, 'r') as f:
    cfg = yaml.safe_load(f)

with open(ruta_cfg_parametros, 'r') as f:
    cfg_params = yaml.safe_load(f)

pop_size         = cfg['particulas_pso']
evaluaciones_max = cfg['evaluaciones_max']
numero_objetivos = cfg['numero_objetivos']

nombre_parametros = list(cfg_params.keys())
cI           = np.array([cfg_params[x]['minimo'] for x in nombre_parametros])
cS           = np.array([cfg_params[x]['maximo'] for x in nombre_parametros])
n_parametros = len(nombre_parametros)

id_instancia = f"R{run_num:02d}"


# =============================================================================
# PARTICLE CLASS
# =============================================================================

class Particle:
    def __init__(self):
        self.ID          = None
        self.semilla     = None
        self.parametros  = None
        self.velocidad   = None
        self.mejor_local = self._mejor_vacio()

    def _mejor_vacio(self):
        m            = Particle.__new__(Particle)
        m.ID         = []
        m.parametros = []
        m.fitness    = [[np.inf for _ in range(numero_objetivos)]]
        return m


# =============================================================================
# PARTICLE INITIALISATION
# =============================================================================

# Initial positions and restarts via Halton (low-discrepancy -> better space coverage)
_halton         = Halton(d=n_parametros, scramble=True)
_inicial        = cI + _halton.random(n=pop_size) * (cS - cI)
_halton_restart = Halton(d=n_parametros, scramble=True)

particles = []

for i in range(pop_size):
    p = Particle()

    p.ID         = generar_ID(id_instancia)
    p.semilla    = int(np.random.randint(2147483647))
    p.parametros = _inicial[i]
    p.velocidad  = 0.1 * al(cI, cS, n_parametros)

    p.mejor_local.ID         = [p.ID]
    p.mejor_local.parametros = [p.parametros.copy()]
    p.mejor_local.fitness    = [[np.inf for _ in range(numero_objetivos)]]

    Genera_Ficha(p, dir_fichas, nombre_parametros)
    particles.append(p)


# =============================================================================
# INITIAL GLOBAL BEST
# =============================================================================

mejor_global            = Particle.__new__(Particle)
mejor_global.ID         = [particles[0].ID]
mejor_global.parametros = [particles[0].parametros.copy()]
mejor_global.fitness    = [[np.inf for _ in range(numero_objetivos)]]


# =============================================================================
# COUNTERS
# =============================================================================

conta_fichas       = pop_size   # pop_size files already generated
conta_evaluaciones = 0


# =============================================================================
# MAIN LOOP
# =============================================================================

t_inicio = args.t_inicio if args.t_inicio is not None else time.time()
seguir = True

while seguir:

    for p in particles:

        ruta_error = os.path.join(dir_errores, p.ID + '.error')

        if not os.path.exists(ruta_error):
            continue

        # -----------------------------------------------------------------
        # Process result
        # -----------------------------------------------------------------
        fitness = leer_fichero_error(ruta_error)
        conta_evaluaciones += 1
        os.remove(ruta_error)

        if conta_evaluaciones >= evaluaciones_max:
            seguir = False

        # -----------------------------------------------------------------
        # Update local and global bests
        # -----------------------------------------------------------------
        datos_simulacion = [p.ID, p.parametros, fitness]
        p.mejor_local, hay_mejora_local = actualiza_mejor(p.mejor_local, datos_simulacion)

        # A local improvement may imply a global one; the reverse is not true.
        hay_mejora_global = False
        if hay_mejora_local:
            mejor_global, hay_mejora_global = actualiza_mejor(mejor_global, datos_simulacion)

        # -----------------------------------------------------------------
        # Progress messages
        # -----------------------------------------------------------------
        err_str = "  ".join(f"{v:.6f}" for v in mejor_global.fitness[0])

        if hay_mejora_global:
            elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))
            print(f"  Run {run_num:02d}  [{elapsed}]  Eval {conta_evaluaciones}/{evaluaciones_max}  —  nuevo mejor: {err_str}")
            gbest_2_yaml(mejor_global, nombre_parametros,
                         dir_mejores, conta_evaluaciones, run_num)

        elif conta_evaluaciones % 2500 == 0:
            print(f"  Run {run_num:02d}  Eval {conta_evaluaciones}/{evaluaciones_max}  —  mejor: {err_str}")

        # -----------------------------------------------------------------
        # PSO update
        # -----------------------------------------------------------------

        # Option 1: an external guess is available (from NM or surrogate)
        viene_de_guess = False
        if dir_guesses:
            guesses = glob.glob(os.path.join(dir_guesses, '*.yaml'))
            if guesses:
                with open(guesses[0], 'r') as f:
                    guess = yaml.safe_load(f)
                p.parametros   = np.array([guess[x] for x in nombre_parametros])
                p.semilla      = int(guess.get('Semilla', np.random.randint(2147483647)))
                p.ID           = guess['Identificador']
                viene_de_guess = True
                os.rename(guesses[0], guesses[0] + '.usado')

        # Option 2: random restart (10% probability)
        if not viene_de_guess and al() < 0.1:
            p.parametros = cI + _halton_restart.random(n=1)[0] * (cS - cI)
            p.semilla    = int(np.random.randint(2147483647))

        # Option 3: standard PSO update
        elif not viene_de_guess:
            # Random inertia
            p.velocidad = p.velocidad * al(0.25, 0.75)

            # Attraction towards all historical local bests
            for x in p.mejor_local.parametros:
                p.velocidad = p.velocidad + al(0, 1.5) * (np.array(x) - p.parametros)

            # Attraction towards all global bests on the Pareto front
            for x in mejor_global.parametros:
                p.velocidad = p.velocidad + al(0, 1.5) * (np.array(x) - p.parametros)

            p.parametros = p.parametros + p.velocidad

            # Polynomial mutation (10% probability)
            if al() < 0.1:
                p.parametros = mutacion(p.parametros, cI, cS)

            # Boundary adjustment
            p.parametros = np.array(ajustar_bordes(p.parametros, cI, cS))
            p.semilla    = int(np.random.randint(2147483647))

        # -----------------------------------------------------------------
        # Generate new particle file if max not yet reached
        # -----------------------------------------------------------------
        if conta_fichas < evaluaciones_max:
            if not viene_de_guess:
                p.ID = generar_ID(id_instancia)
            Genera_Ficha(p, dir_fichas, nombre_parametros)
            conta_fichas += 1

    # -------------------------------------------------------------------------
    # Manual forced stop
    # -------------------------------------------------------------------------
    if os.path.exists(os.path.join(dir_fichas, 'FIN')):
        print("\n  FIN file detected. Stopping.")
        seguir = False


# =============================================================================
# SHUTDOWN
# =============================================================================

print(f"\n{'='*60}")
print(f"  Calibration finished.  Evaluations: {conta_evaluaciones}")
print(f"{'='*60}")

gbest_2_yaml(mejor_global, nombre_parametros, dir_mejores, conta_evaluaciones, run_num)

with open(os.path.join(dir_fichas, 'PSO_FIN'), 'w') as f:
    f.write(f"Calibration finished. Evaluations: {conta_evaluaciones}\n")

# END -------------------------------------------------------------------------
