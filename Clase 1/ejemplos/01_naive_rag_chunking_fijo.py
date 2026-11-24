#!/usr/bin/env python3
"""
Lección 1 — Naive RAG con chunking fijo
========================================

Naive RAG es el pipeline mínimo, el que conviene mostrar antes de
mejorar nada:

    1. Se parte el documento en ventanas del mismo tamaño.
    2. Cada ventana se convierte en un vector y se guarda.
    3. La pregunta se convierte en un vector con el mismo criterio.
    4. Se traen las ventanas más parecidas.
    5. Esas ventanas se pegan en el prompt (a esto se le dice stuffing).
    6. Un modelo de lenguaje lee el prompt y redacta la respuesta.

Esta lección hace los pasos 1 a 5. El paso 6 no llama a ninguna API:
imprime el prompt. En clase se puede copiar ese prompt a cualquier
modelo y comparar respuestas.

El chunking fijo no mira títulos, puntos ni párrafos. Corta cada
``TAMANO`` caracteres y repite ``SOLAPAMIENTO`` caracteres en el
fragmento siguiente. El solapamiento existe por una sola razón: una
frase que caiga justo en el borde tiene otra oportunidad de quedar
entera del otro lado.

La unidad de corte acá es el carácter, no el token. En producción se
suele contar tokens (tiktoken, el tokenizador del modelo, etc.) porque
el límite del prompt es en tokens. La idea del algoritmo no cambia:
ventana fija más solapamiento. Usamos caracteres para no sumar
dependencias.

Cómo correrlo
-------------
    python ejemplos/01_naive_rag_chunking_fijo.py
    python ejemplos/01_naive_rag_chunking_fijo.py "¿Abren los lunes en el puerto?"

Qué mirar en la salida
----------------------
- Un borde que parte una palabra por la mitad.
- Si la oración de los 14 días quedó entera dentro de algún fragmento.
- La segunda pregunta, redactada con otras palabras: el buscador léxico
  no la vincula con "renovación". Eso no se arregla cambiando el tamaño
  de la ventana. Se arregla con embeddings semánticos, que son otra cosa.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from rag_curso.embeddings import buscar, terminos_compartidos
from rag_curso.texto import (
    cargar_manual,
    encabezado,
    imprimir_contextos,
    imprimir_prompt,
    listar_fragmentos,
    pregunta_desde_argv,
    preparar_salida,
    vista,
)


NOMBRE = "Naive RAG / chunking fijo"

# Ventana chica a propósito: con un manual de este largo se ven varios
# cortes y es fácil encontrar una palabra partida. Subir este número
# esconde el problema, no lo resuelve. 510 no es un número mágico: con
# este corpus parte la oración de los 14 días. Otro tamaño puede dejarla
# entera, y el algoritmo no se entera ni en un caso ni en el otro.
TAMANO = 510
SOLAPAMIENTO = 80
TOP_K = 2

PREGUNTA = "¿Cuál es el horario de la Sede Puerto?"
ORACION_CLAVE = (
    "El préstamo de la colección general dura 14 días corridos, "
    "contados desde el día de salida, que también se computa."
)

# Misma intención que la sección de renovaciones, sin usar sus palabras.
# Sirve para separar dos ideas que se mezclan seguido:
#   - chunking: dónde corto el documento
#   - embeddings: cómo comparo una pregunta con un fragmento
PREGUNTA_CON_OTRAS_PALABRAS = (
    "¿Hay manera de prolongar la tenencia si no llegué a leerlo entero?"
)


def ventanas_fijas(
    texto: str, tamano: int = TAMANO, solapamiento: int = SOLAPAMIENTO
) -> list[tuple[int, int]]:
    """Calcula los intervalos ``[inicio, fin)`` de cada ventana.

    El paso entre ventanas es ``tamano - solapamiento``. Si el paso fuera
    igual al tamaño, los fragmentos no compartirían texto. Si el
    solapamiento fuera mayor o igual que el tamaño, el paso sería cero
    o negativo y el bucle no avanzaría.
    """

    if tamano <= 0:
        raise ValueError("El tamaño tiene que ser un entero positivo.")
    if solapamiento < 0 or solapamiento >= tamano:
        raise ValueError("El solapamiento tiene que ser >= 0 y menor que el tamaño.")

    intervalos: list[tuple[int, int]] = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fin = min(inicio + tamano, n)
        intervalos.append((inicio, fin))
        if fin == n:
            break
        inicio = fin - solapamiento
    return intervalos


def chunking_fijo(
    texto: str, tamano: int = TAMANO, solapamiento: int = SOLAPAMIENTO
) -> list[str]:
    """Parte ``texto`` en ventanas de largo fijo, con solapamiento.

    No recorta al espacio más cercano. Ese "mejor esfuerzo" ya sería
    otro algoritmo. Acá el corte cae donde cae: en medio de una palabra,
    de una oración o de un título. Es el comportamiento que se quiere
    mostrar.
    """

    texto = texto.strip()
    if not texto:
        return []
    return [texto[inicio:fin] for inicio, fin in ventanas_fijas(texto, tamano, solapamiento)]


def construir_fragmentos(texto: str) -> list[str]:
    """API que usa la comparación entre estrategias."""

    return chunking_fijo(texto, TAMANO, SOLAPAMIENTO)


def contexto_para_prompt(texto: str, pregunta: str, k: int = TOP_K) -> list[dict]:
    """Indexa las ventanas fijas y devuelve las ``k`` que irían al prompt.

    Fase offline: cortar y vectorizar.
    Fase online: vectorizar la pregunta y ordenar por similitud.
    Las dos fases están separadas en un sistema real (el índice se
    construye antes y se reutiliza). Acá se rehacen en cada llamada
    porque el corpus entra en memoria y se lee mejor el camino completo.
    """

    fragmentos = construir_fragmentos(texto)
    ranking = buscar(fragmentos, pregunta, k)
    contextos = []
    for posicion, score in ranking:
        fragmento = fragmentos[posicion]
        contextos.append(
            {
                "texto": fragmento,
                "score": score,
                "cita": (
                    f"fuente: manual_lunaria.md | fragmento {posicion} | "
                    "estrategia: chunking fijo"
                ),
                "coincidencia": fragmento,
                "nota": "La unidad indexada y la unidad enviada al modelo son la misma ventana.",
                "terminos": terminos_compartidos(pregunta, fragmento),
            }
        )
    return contextos


def _contrato_del_solapamiento(texto: str, intervalos: list[tuple[int, int]]) -> None:
    """El final de una ventana tiene que reaparecer al inicio de la siguiente.

    Si alguien cambia la cuenta ``fin - solapamiento`` y rompe el paso,
    esta comprobación frena la lección en vez de mostrar un solapamiento
    que ya no existe.
    """

    for (inicio, fin), (siguiente_inicio, siguiente_fin) in zip(intervalos, intervalos[1:]):
        assert siguiente_inicio == fin - SOLAPAMIENTO, (
            "El paso entre ventanas dejó de ser tamaño - solapamiento."
        )
        cola = texto[fin - SOLAPAMIENTO : fin]
        cabeza = texto[siguiente_inicio : siguiente_inicio + SOLAPAMIENTO]
        assert cola == cabeza
        assert siguiente_fin > siguiente_inicio


def _mostrar_corte_de_palabra(texto: str, intervalos: list[tuple[int, int]]) -> None:
    """Imprime el primer borde que cae en medio de una palabra."""

    for inicio, fin in intervalos[:-1]:
        if fin <= 0 or fin >= len(texto):
            continue
        if texto[fin - 1].isalnum() and texto[fin].isalnum():
            antes = texto[max(0, fin - 28) : fin].replace("\n", "↵")
            despues = texto[fin : fin + 28].replace("\n", "↵")
            print("\nCorte en medio de una palabra (│ es el borde; ↵ es un salto de línea):")
            print(f"  …{antes}│{despues}…")
            print(
                "Ese medio token no ayuda a buscar ni se lee bien si el "
                "modelo recibe solo uno de los dos lados."
            )
            return
    print("\nCon este tamaño, ninguna ventana corta una palabra. Probá un TAMANO más chico.")


def _mostrar_oracion_clave(texto: str, intervalos: list[tuple[int, int]]) -> None:
    """Dice si la oración de los 14 días sobrevivió entera a los cortes."""

    inicio = texto.find(ORACION_CLAVE)
    if inicio < 0:
        print(f"\nNo encontré la oración clave:\n  {ORACION_CLAVE}")
        return
    fin = inicio + len(ORACION_CLAVE)
    for indice, (ventana_inicio, ventana_fin) in enumerate(intervalos):
        if ventana_inicio <= inicio and fin <= ventana_fin:
            print(
                f"\nLa oración de los 14 días quedó ENTERA en el fragmento {indice}."
            )
            print(
                "No siempre pasa. Cambiá TAMANO (por ejemplo 220 o 350) y "
                "volvé a correr: el mismo dato puede partirse sin que el "
                "algoritmo se entere."
            )
            return

    print("\nLa oración de los 14 días quedó PARTIDA. Ningún fragmento la contiene entera:")
    print(f"  {ORACION_CLAVE}")
    for indice, (ventana_inicio, ventana_fin) in enumerate(intervalos):
        if ventana_fin <= inicio or ventana_inicio >= fin:
            continue
        recorte = texto[max(ventana_inicio, inicio) : min(ventana_fin, fin)]
        print(f"  fragmento {indice} se queda con: {recorte!r}")


def _demostrar_brecha_lexica(fragmentos: list[str]) -> None:
    """Misma base, pregunta parafraseada: el ranking se va a otro lado."""

    print("\nMisma indexación, otras palabras")
    print(f"Pregunta: {PREGUNTA_CON_OTRAS_PALABRAS}")
    print(
        "El manual habla de «renovaciones». Esta pregunta no usa esa palabra. "
        "TF-IDF solo puede puntuar términos compartidos."
    )
    ranking = buscar(fragmentos, PREGUNTA_CON_OTRAS_PALABRAS, k=1)
    if not ranking:
        print("Sin resultados.")
        return
    posicion, score = ranking[0]
    elegido = fragmentos[posicion]
    print(f"Top 1 · fragmento {posicion} · similitud {score:.3f}")
    compartidos = terminos_compartidos(PREGUNTA_CON_OTRAS_PALABRAS, elegido)
    print(
        "Único puente léxico: "
        + (", ".join(compartidos) if compartidos else "ningún término")
    )
    print(vista(elegido, 240))
    if "renovacion" in elegido.lower() or "renovación" in elegido.lower():
        print("En esta corrida el fragmento ganador sí habla de renovaciones.")
    else:
        print(
            "El fragmento ganador no es la sección de renovaciones. "
            "Achicar o agrandar la ventana no enseña sinónimos."
        )


def main() -> None:
    preparar_salida()
    encabezado("Lección 1 · Naive RAG / chunking fijo")
    print(
        f"Ventana: {TAMANO} caracteres   Solapamiento: {SOLAPAMIENTO}   "
        f"Top-k: {TOP_K}"
    )
    print(
        "Fase offline: cortar el manual en ventanas iguales, sin leer su estructura.\n"
        "Fase online: buscar las ventanas más parecidas a la pregunta y armar el prompt."
    )

    manual = cargar_manual()
    print(f"\nCorpus: datos/manual_lunaria.md ({len(manual)} caracteres)")

    intervalos = ventanas_fijas(manual.strip(), TAMANO, SOLAPAMIENTO)
    fragmentos = [manual.strip()[inicio:fin] for inicio, fin in intervalos]
    _contrato_del_solapamiento(manual.strip(), intervalos)
    print("Contrato del solapamiento: ok (cada ventana repite el borde de la anterior).")

    listar_fragmentos(fragmentos)
    _mostrar_corte_de_palabra(manual.strip(), intervalos)
    _mostrar_oracion_clave(manual.strip(), intervalos)

    pregunta = pregunta_desde_argv(PREGUNTA)
    print(f"\nFase online\nPregunta: {pregunta}")
    contextos = contexto_para_prompt(manual, pregunta, TOP_K)
    if contextos and contextos[0]["score"] < 0.05:
        print(
            "Aviso: la similitud más alta es muy baja. "
            "La pregunta casi no comparte términos con el manual."
        )
    imprimir_contextos(contextos)
    if pregunta == PREGUNTA and contextos:
        horario = "10:00 a 18:00"
        en_el_primero = horario in contextos[0]["texto"]
        en_alguno = any(horario in contexto["texto"] for contexto in contextos)
        if not en_el_primero and en_alguno:
            print(
                "\nEl primer fragmento corta la oración del horario antes de "
                f"«{horario}». El dato entra al prompt solo porque TOP_K "
                "también trae la ventana siguiente. Con TOP_K = 1, no entraría."
            )
        elif not en_alguno:
            print(
                f"\n«{horario}» no quedó en ningún fragmento recuperado. "
                "El modelo no podría responder el horario con este contexto."
            )
    imprimir_prompt(pregunta, contextos)

    if pregunta == PREGUNTA:
        _demostrar_brecha_lexica(fragmentos)

    print(
        "\nPara seguir: la lección 2 corta por párrafos y oraciones "
        "antes de resignarse a una ventana fija."
    )


if __name__ == "__main__":
    main()
