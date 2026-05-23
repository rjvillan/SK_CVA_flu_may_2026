# -*- encoding: utf-8 -*-

__author__ = "Rafael J. Villanueva"
__oracion__ = "A mayor Gloria de Dios Nuestro Señor"
__fecha__   = "10 de mayo de 2026"

"""
1.-Configurar_parametros.py
---------------------------
Interactive script to define the parameters to calibrate.

Generates:
    0.-Configuracion/cfg_parametros.yaml
"""

import os
import yaml

DIR_CONFIG = "0.-Configuracion"


# =============================================================================
# USER INTERACTION FUNCTIONS
# =============================================================================

def preguntar(texto, valor_defecto=None, tipo=str):
    if valor_defecto is not None:
        prompt = f"  {texto} [{valor_defecto}]: "
    else:
        prompt = f"  {texto}: "

    while True:
        respuesta = input(prompt).strip()
        if respuesta == "" and valor_defecto is not None:
            return valor_defecto
        if respuesta == "":
            print("    -> Este campo es obligatorio.")
            continue
        try:
            return tipo(respuesta)
        except ValueError:
            print(f"    -> Valor no válido. Se esperaba un {tipo.__name__}.")


def preguntar_si_no(texto, valor_defecto=True):
    while True:
        respuesta = input(f"  {texto} [{'S' if valor_defecto else 'N'}]: ").strip().upper()
        if respuesta == "":
            return valor_defecto
        if respuesta in ("S", "SI", "SÍ", "Y", "YES"):
            return True
        if respuesta in ("N", "NO"):
            return False
        print("    -> Responde S o N.")


def seccion(titulo):
    print()
    print("=" * 60)
    print(f"  {titulo}")
    print("=" * 60)


def subseccion(titulo):
    print()
    print(f"  --- {titulo} ---")


# =============================================================================
# LOAD EXISTING CONFIGURATION
# =============================================================================

def cargar_parametros_existentes():
    ruta = os.path.join(DIR_CONFIG, "cfg_parametros.yaml")
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"\n  Parámetros existentes encontrados: {ruta}")
        print("  Los valores actuales se usarán como valores por defecto.")
        return cfg
    return {}


# =============================================================================
# PARAMETER CONFIGURATION
# =============================================================================

def configurar_parametros(cfg_parametros):
    seccion("PARÁMETROS A CALIBRAR")
    print()
    print("  Define los parámetros que PSO y NM van a optimizar.")
    print("  Para cada parámetro: nombre, mínimo, máximo y comentario.")

    if cfg_parametros:
        print()
        print("  Parámetros actuales:")
        for nombre, datos in cfg_parametros.items():
            print(f"    {nombre}: [{datos['minimo']}, {datos['maximo']}]  -- {datos.get('comentario', '')}")

        mantener = preguntar_si_no("\n  ¿Mantener los parámetros actuales?", valor_defecto=True)
        if not mantener:
            cfg_parametros = {}

    nuevos_params = dict(cfg_parametros)

    while True:
        print()
        accion = preguntar(
            "¿Qué quieres hacer? (a=añadir, e=editar, b=borrar, fin=terminar)",
            valor_defecto="fin"
        ).lower()

        if accion == "fin":
            if len(nuevos_params) == 0:
                print("    -> Debes definir al menos un parámetro.")
                continue
            break

        elif accion == "a":
            subseccion("Añadir parámetro")
            nombre = preguntar("Nombre del parámetro (sin espacios)").replace(" ", "_")
            if nombre in nuevos_params:
                print(f"    -> El parámetro '{nombre}' ya existe. Usa 'e' para editarlo.")
                continue
            minimo     = preguntar(f"  Valor mínimo de {nombre}", tipo=float)
            maximo     = preguntar(f"  Valor máximo de {nombre}", tipo=float)
            if maximo <= minimo:
                print("    -> El máximo debe ser mayor que el mínimo.")
                continue
            comentario = preguntar("  Comentario (descripción breve)", valor_defecto="")
            nuevos_params[nombre] = {"minimo": minimo, "maximo": maximo, "comentario": comentario}
            print(f"    -> Parámetro '{nombre}' añadido.")

        elif accion == "e":
            subseccion("Editar parámetro")
            if not nuevos_params:
                print("    -> No hay parámetros definidos aún.")
                continue
            nombre = preguntar("Nombre del parámetro a editar")
            if nombre not in nuevos_params:
                print(f"    -> No existe el parámetro '{nombre}'.")
                continue
            actual     = nuevos_params[nombre]
            minimo     = preguntar("  Valor mínimo", valor_defecto=actual["minimo"], tipo=float)
            maximo     = preguntar("  Valor máximo", valor_defecto=actual["maximo"], tipo=float)
            if maximo <= minimo:
                print("    -> El máximo debe ser mayor que el mínimo.")
                continue
            comentario = preguntar("  Comentario", valor_defecto=actual.get("comentario", ""))
            nuevos_params[nombre] = {"minimo": minimo, "maximo": maximo, "comentario": comentario}
            print(f"    -> Parámetro '{nombre}' actualizado.")

        elif accion == "b":
            subseccion("Borrar parámetro")
            if not nuevos_params:
                print("    -> No hay parámetros definidos.")
                continue
            nombre = preguntar("Nombre del parámetro a borrar")
            if nombre not in nuevos_params:
                print(f"    -> No existe el parámetro '{nombre}'.")
                continue
            if preguntar_si_no(f"  ¿Seguro que quieres borrar '{nombre}'?", valor_defecto=False):
                del nuevos_params[nombre]
                print(f"    -> Parámetro '{nombre}' borrado.")

        else:
            print("    -> Opción no reconocida. Escribe a, e, b o fin.")

    return nuevos_params


# =============================================================================
# SUMMARY AND FILE WRITING
# =============================================================================

def mostrar_resumen(cfg_parametros):
    seccion("RESUMEN")
    print()
    print("  Parámetros a calibrar:")
    for nombre, datos in cfg_parametros.items():
        print(f"    {nombre}: [{datos['minimo']}, {datos['maximo']}]  -- {datos.get('comentario', '')}")


def escribir_cfg_parametros(cfg_parametros):
    lineas = (
        f"# ---------------------------------------------------------------\n"
        f"# PARAMETERS TO CALIBRATE\n"
        f"# ---------------------------------------------------------------\n"
        f"\n"
    )
    for nombre, datos in cfg_parametros.items():
        lineas += f"{nombre}:\n"
        lineas += f"  minimo:     {datos['minimo']}\n"
        lineas += f"  maximo:     {datos['maximo']}\n"
        lineas += f"  comentario: {datos.get('comentario', '')}\n"
        lineas += f"\n"

    ruta = os.path.join(DIR_CONFIG, "cfg_parametros.yaml")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(lineas)
    print(f"  Guardado: {ruta}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("  CONFIGURADOR DE PARÁMETROS")
    print("=" * 60)
    print()
    print("  Pulsa Enter para aceptar el valor por defecto entre corchetes.")

    os.makedirs(DIR_CONFIG, exist_ok=True)

    cfg_parametros = cargar_parametros_existentes()
    cfg_parametros = configurar_parametros(cfg_parametros)

    mostrar_resumen(cfg_parametros)

    print()
    if not preguntar_si_no("¿Guardar la configuración?", valor_defecto=True):
        print("\n  Configuración descartada.")
        return

    seccion("GUARDANDO")
    escribir_cfg_parametros(cfg_parametros)
    print()


if __name__ == "__main__":
    main()
