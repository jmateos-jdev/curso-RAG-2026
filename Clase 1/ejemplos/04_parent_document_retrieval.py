#!/usr/bin/env python3
"""
Lección 4 — Parent-document retrieval
=====================================

Hasta acá, el pedazo que se indexa y el pedazo que se pega en el prompt
son el mismo. Eso obliga a un compromiso incómodo:

- Si el pedazo es chico, la búsqueda es precisa y el modelo se queda
  sin el párrafo de al lado, que a veces tiene la condición, la excepción
  o la consecuencia.
- Si el pedazo es grande, el modelo tiene contexto y la búsqueda se
  diluye: un vector que mezcla multa, reclamo y suspensión se parece
  poco a una pregunta concreta.

Parent-document retrieval (también lo vas a ver como small-to-big)
separa las dos unidades:

    documento
      └── padre   (lo que lee el modelo: una sección entera)
            └── hijos   (lo que se indexa: unas pocas oraciones)

Se busca entre los hijos. Cuando un hijo gana, al prompt no va el hijo:
va su padre. Si dos hijos del mismo padre entran en el ranking, el padre
se manda una sola vez.

En LangChain esto es ``ParentDocumentRetriever``. En LlamaIndex, el
primo más cercano es el ``AutoMergingRetriever``, que además puede
subir del padre al abuelo si varios hijos del mismo padre coincidieron.
Acá hay un solo nivel, que es el que hace visible la idea. Un segundo
nivel sería guardar el capítulo como abuelo y repetir la misma regla.

El tamaño del padre es una decisión de diseño, no un dato del markdown.
Este manual usa ``##`` como padre. Si el FAQ entero es un solo ``##``,
una pregunta sobre fotos de manuscritos arrastra el resto de las
preguntas frecuentes. Partir ese FAQ en varios ``##`` achicaría el
contexto. Conviene probarlo en clase.

Cómo correrlo
-------------
    python ejemplos/04_parent_document_retrieval.py

Qué mirar en la salida
----------------------
La pregunta junta dos hechos de la misma sección que están en párrafos
distintos: el monto de la multa y la suspensión a los 21 días. El hijo
que coincide suele tener solo uno. El padre que se envía tiene los dos.
Esa diferencia es toda la lección.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from rag_curso.embeddings import buscar, terminos_compartidos
from rag_curso.texto import (
    cargar_manual,
    dividir_oraciones,
    encabezado,
    imprimir_contextos,
    imprimir_prompt,
    listar_fragmentos,
    pregunta_desde_argv,
    preparar_salida,
)


NOMBRE = "Parent-document retrieval"

# El hijo es chico para que el vector hable de una sola idea.
# El solapamiento se cuenta en oraciones, no en caracteres: se repite
# la última oración del hijo anterior para no perder un hecho que
# quede entre dos paquetes.
TAMANO_HIJO = 320
SOLAPAMIENTO_ORACIONES = 1

# Cuántos hijos se miran antes de deduplicar, y cuántos padres distintos
# se llegan a inyectar. k, para esta lección, es cantidad de padres:
# es lo que ocupa lugar en el prompt.
K_HIJOS_REVISADOS = 6
K_PADRES = 2
TOP_K = K_PADRES

PREGUNTA = (
    "¿De cuánto es la multa por cada día de retraso y cuándo queda suspendido el carnet?"
)
MARCA_MULTA = "cincuenta centavos de lunar"
MARCA_SUSPENSION = "21 días de retraso"


def dividir_en_padres(texto: str) -> list[dict]:
    """Cada sección ``##`` es un padre. Lo anterior al primer ``##`` también.

    ``^## `` no coincide con ``### ``: un subtítulo de tres numerales
    se queda dentro del padre. Así se ve que la jerarquía la decide
    quien indexa, no el archivo.
    """

    partes = re.split(r"(?=^## )", texto.strip(), flags=re.MULTILINE)
    padres = []
    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue
        primera_linea = parte.splitlines()[0]
        if primera_linea.startswith("#"):
            titulo = primera_linea.lstrip("#").strip()
        else:
            titulo = "Introducción"
        padres.append({"titulo": titulo, "texto": parte})
    return padres


def empaquetar_hijos(
    texto_padre: str,
    tamano: int = TAMANO_HIJO,
    solapamiento_oraciones: int = SOLAPAMIENTO_ORACIONES,
) -> list[str]:
    """Mete oraciones del padre en hijos que rondan ``tamano`` caracteres.

    La primera oración entra aunque sea más larga que el tope: un hijo
    vacío no se puede indexar, y partir la oración sería volver al
    chunking fijo. El avance mínimo es una oración, así el solapamiento
    no deja el bucle clavado en la misma posición.
    """

    oraciones = dividir_oraciones(texto_padre)
    if not oraciones:
        return []

    hijos: list[str] = []
    inicio = 0
    while inicio < len(oraciones):
        actual: list[str] = []
        largo = 0
        cursor = inicio
        while cursor < len(oraciones):
            extra = len(oraciones[cursor]) + (1 if actual else 0)
            if actual and largo + extra > tamano:
                break
            actual.append(oraciones[cursor])
            largo += extra
            cursor += 1
        if not actual:
            break
        hijos.append("\n".join(actual))
        if cursor >= len(oraciones):
            break
        siguiente = cursor - solapamiento_oraciones
        if siguiente <= inicio:
            siguiente = inicio + 1
        inicio = siguiente
    return hijos


def construir_hijos(texto: str) -> list[dict]:
    """Índice chico: cada registro sabe qué padre tiene que devolver."""

    registros = []
    for id_padre, padre in enumerate(dividir_en_padres(texto)):
        for hijo in empaquetar_hijos(padre["texto"]):
            registros.append(
                {
                    "texto": hijo,
                    "padre_id": id_padre,
                    "padre_titulo": padre["titulo"],
                    "padre_texto": padre["texto"],
                }
            )
    return registros


def construir_fragmentos(texto: str) -> list[str]:
    """Devuelve los hijos, que son las unidades indexadas.

    No son los textos que van al prompt. Esa diferencia es el punto
    de la lección y la comparación la muestra con los dos números.
    """

    return [registro["texto"] for registro in construir_hijos(texto)]


def contexto_para_prompt(texto: str, pregunta: str, k: int = TOP_K) -> list[dict]:
    """Busca hijos y traduce cada acierto al padre, sin repetir secciones.

    ``k`` es la cantidad máxima de padres en el prompt. Se revisan más
    hijos que ``k`` porque los primeros puestos pueden ser tres hijos
    de la misma sección: después de deduplicar quedaría un solo bloque.
    """

    hijos = construir_hijos(texto)
    if not hijos:
        return []
    ranking = buscar(
        [hijo["texto"] for hijo in hijos],
        pregunta,
        k=max(k * 3, K_HIJOS_REVISADOS),
    )
    vistos: set[int] = set()
    contextos = []
    for posicion, score in ranking:
        hijo = hijos[posicion]
        if hijo["padre_id"] in vistos:
            continue
        vistos.add(hijo["padre_id"])
        contextos.append(
            {
                "texto": hijo["padre_texto"],
                "score": score,
                "cita": (
                    "fuente: manual_lunaria.md | "
                    f"sección: {hijo['padre_titulo']} | "
                    f"recuperado vía hijo {posicion}"
                ),
                "coincidencia": hijo["texto"],
                "nota": (
                    f"Coincidió un hijo de «{hijo['padre_titulo']}». "
                    "Al modelo se le manda la sección completa, una sola vez."
                ),
                "terminos": terminos_compartidos(pregunta, hijo["texto"]),
            }
        )
        if len(contextos) >= k:
            break
    return contextos


def _informe_de_la_leccion(contextos: list[dict]) -> None:
    """Contrasta el hijo ganador con el padre que sí viaja al modelo."""

    if not contextos:
        return
    hijo = contextos[0]["coincidencia"]
    padre = contextos[0]["texto"]
    print("\nContraste hijo / padre en el primer bloque")
    print(f"  El hijo menciona la multa ({MARCA_MULTA!r}): {MARCA_MULTA in hijo}")
    print(
        f"  El hijo menciona la suspensión ({MARCA_SUSPENSION!r}): "
        f"{MARCA_SUSPENSION in hijo}"
    )
    print(f"  El padre menciona la multa: {MARCA_MULTA in padre}")
    print(f"  El padre menciona la suspensión: {MARCA_SUSPENSION in padre}")
    print(
        "Si el hijo tiene solo uno de los dos hechos y el padre tiene ambos, "
        "el modelo recibe la consecuencia que la búsqueda chica no había pedido."
    )
    print(
        f"Tamaño del hijo: {len(hijo)} caracteres. "
        f"Tamaño del padre enviado: {len(padre)} caracteres."
    )


def main() -> None:
    preparar_salida()
    encabezado("Lección 4 · Parent-document retrieval")
    print(
        f"Hijo: ~{TAMANO_HIJO} caracteres   "
        f"Solapamiento: {SOLAPAMIENTO_ORACIONES} oración   "
        f"Padres en el prompt: {K_PADRES}"
    )

    manual = cargar_manual()
    padres = dividir_en_padres(manual)
    hijos = construir_hijos(manual)
    print(f"\nCorpus: datos/manual_lunaria.md ({len(manual)} caracteres)")
    print(f"Padres (secciones): {len(padres)}")
    for padre in padres:
        print(f"  - {padre['titulo']} ({len(padre['texto'])} caracteres)")
    listar_fragmentos(
        [hijo["texto"] for hijo in hijos],
        titulo="Hijos indexados (esto es lo que se busca; no lo que lee el modelo)",
    )

    pregunta = pregunta_desde_argv(PREGUNTA)
    print(f"\nPregunta: {pregunta}")
    contextos = contexto_para_prompt(manual, pregunta, K_PADRES)
    imprimir_contextos(contextos)
    if len(contextos) < K_PADRES:
        texto_cantidad = "una sola sección" if len(contextos) == 1 else f"{len(contextos)} secciones"
        print(
            f"\nHabía lugar para {K_PADRES} padres en el prompt, pero los hijos "
            f"mejor ubicados caen en {texto_cantidad}. El mismo padre no se envía dos veces."
        )
    if pregunta == PREGUNTA:
        _informe_de_la_leccion(contextos)
    imprimir_prompt(pregunta, contextos)
    print(
        "\nCierre del recorrido: el padre ayuda cuando es la sección correcta "
        "y estorba cuando es demasiado grande. El tamaño se elige, no se hereda."
    )


if __name__ == "__main__":
    main()
