#!/usr/bin/env python3
"""
Comparación de las cuatro estrategias con las mismas preguntas.

No reimplementa nada: importa cada lección y llama a la misma API.

    construir_fragmentos(texto) -> lo que se indexa
    contexto_para_prompt(texto, pregunta, k) -> lo que iría al modelo

Así se ve, lado a lado, cuánto texto se indexó y cuánto texto entraría
en el prompt. Las dos preguntas están elegidas porque piden un dato
concreto del manual:

- los 14 días de la colección general
- la multa diaria y la suspensión del carnet

k = 2 en todas. En parent-document, 2 son padres, no hijos.

    python ejemplos/05_comparacion.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from rag_curso.texto import cargar_manual, encabezado, preparar_salida, vista


LECCIONES = (
    ("01_naive_rag_chunking_fijo.py", "leccion_fijo"),
    ("02_chunking_recursivo.py", "leccion_recursivo"),
    ("03_chunking_semantico.py", "leccion_semantico"),
    ("04_parent_document_retrieval.py", "leccion_parent"),
)

ESCENARIOS = (
    {
        "titulo": "Un dato corto: duración del préstamo",
        "pregunta": "¿Cuántos días dura el préstamo de la colección general?",
        "pistas": ("14 días corridos",),
    },
    {
        "titulo": "Dos hechos de la misma sección: multa y suspensión",
        "pregunta": (
            "¿De cuánto es la multa por cada día de retraso "
            "y cuándo queda suspendido el carnet?"
        ),
        "pistas": ("cincuenta centavos de lunar", "21 días de retraso"),
    },
)

K = 2


def cargar_leccion(archivo: str, alias: str):
    """Importa un ejemplo aunque el nombre del archivo empiece con un número."""

    ruta = Path(__file__).with_name(archivo)
    spec = importlib.util.spec_from_file_location(alias, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def evaluar(modulo, manual: str, pregunta: str, pistas: tuple[str, ...]) -> dict:
    """Mide indexación y contexto con la pregunta, sin imprimir el prompt entero."""

    fragmentos = modulo.construir_fragmentos(manual)
    contextos = modulo.contexto_para_prompt(manual, pregunta, k=K)
    unido = "\n".join(contexto["texto"] for contexto in contextos)
    return {
        "nombre": modulo.NOMBRE,
        "indexados": len(fragmentos),
        "bloques": len(contextos),
        "caracteres_indexados": sum(len(fragmento) for fragmento in fragmentos),
        "caracteres_prompt": sum(len(contexto["texto"]) for contexto in contextos),
        "score": contextos[0]["score"] if contextos else 0.0,
        "pistas": {pista: pista in unido for pista in pistas},
        "vista": vista(contextos[0]["texto"], 200) if contextos else "(sin contexto)",
        "hijo": vista(contextos[0]["coincidencia"], 160) if contextos else "",
        "cambia_unidad": bool(contextos)
        and contextos[0]["coincidencia"].strip() != contextos[0]["texto"].strip(),
    }


def _linea_tabla(columnas: list[str], anchos: list[int]) -> str:
    return "  ".join(columna.ljust(ancho) for columna, ancho in zip(columnas, anchos))


def imprimir_escenario(manual: str, modulos: list, escenario: dict) -> None:
    """Imprime una tabla corta y, debajo, la vista del primer bloque."""

    print("\n" + escenario["titulo"])
    print("Pregunta: " + escenario["pregunta"])
    filas = [
        evaluar(modulo, manual, escenario["pregunta"], escenario["pistas"])
        for modulo in modulos
    ]
    anchos = [32, 10, 8, 16, 14]
    print(
        _linea_tabla(
            ["Estrategia", "Indexados", "Bloques", "Cars. al prompt", "Pistas"],
            anchos,
        )
    )
    print(_linea_tabla(["-" * ancho for ancho in anchos], anchos))
    for fila in filas:
        pistas_ok = sum(1 for esta in fila["pistas"].values() if esta)
        print(
            _linea_tabla(
                [
                    fila["nombre"],
                    str(fila["indexados"]),
                    str(fila["bloques"]),
                    str(fila["caracteres_prompt"]),
                    f"{pistas_ok}/{len(fila['pistas'])}",
                ],
                anchos,
            )
        )

    print("\nDetalle")
    for fila in filas:
        print(f"\n· {fila['nombre']}  (similitud del primer acierto: {fila['score']:.3f})")
        for pista, esta in fila["pistas"].items():
            estado = "entra al prompt" if esta else "NO entra al prompt"
            print(f"    {estado}: {pista}")
        if fila["cambia_unidad"]:
            print(f"    Hijo que coincidió: {fila['hijo']}")
            print(f"    Padre enviado:      {fila['vista']}")
        else:
            print(f"    Primer bloque: {fila['vista']}")


def main() -> None:
    preparar_salida()
    encabezado("Comparación · las cuatro estrategias, las mismas preguntas")
    print(
        "La búsqueda es TF-IDF en los cuatro casos. "
        f"Cada estrategia aporta como máximo {K} bloques al prompt."
    )
    manual = cargar_manual()
    modulos = [cargar_leccion(archivo, alias) for archivo, alias in LECCIONES]
    for escenario in ESCENARIOS:
        imprimir_escenario(manual, modulos, escenario)
    print(
        "\nCómo leerlo: «Indexados» es lo que se guarda para buscar. "
        "«Bloques» y «Cars. al prompt» son lo que el modelo leería. "
        "En las tres primeras estrategias esos textos son los fragmentos. "
        "En parent-document se indexan hijos y al prompt van las secciones."
    )
    print("El prompt completo de cada una está en su propio ejemplo.")


if __name__ == "__main__":
    main()
