#!/usr/bin/env python3
"""
Lección 2 — Chunking recursivo
==============================

El chunking fijo de la lección 1 es simple y predecible, y también es
ciego: corta donde le toque. El recursivo intenta lo contrario. Prueba
separadores de mayor a menor, y solo baja de nivel cuando el bloque
sigue siendo demasiado grande.

Orden que usa esta lección:

    1. Secciones markdown (líneas que empiezan con "## ")
    2. Párrafos (línea en blanco)
    3. Líneas
    4. Oraciones
    5. Palabras
    6. Caracteres, ya como último recurso

Si un bloque entra en ``TAMANO``, se queda como está. Si no entra, se
abre con el separador más grueso que aparezca adentro y se vuelve a
intentar con los pedazos. Los pedazos chicos se vuelven a juntar hasta
llenar la ventana, y el solapamiento se hace con pedazos enteros (una
oración o un párrafo), no con una cantidad fija de caracteres.

Es el criterio de ``RecursiveCharacterTextSplitter`` en LangChain.
Allá los separadores por defecto son ``["\\n\\n", "\\n", " ", ""]``.
Acá agregamos la sección markdown y la oración porque el corpus de la
clase tiene títulos y vale la pena ver que el orden se puede adaptar
al formato. El algoritmo es el mismo.

Cómo correrlo
-------------
    python ejemplos/02_chunking_recursivo.py
    python ejemplos/02_chunking_recursivo.py "¿Cuántos libros puedo llevarme?"

Qué mirar en la salida
----------------------
- Los fragmentos empiezan en un título, un párrafo o una oración.
- Ningún fragmento supera TAMANO.
- Entre dos fragmentos vecinos se repite alguna oración o párrafo
  completo: ese es el solapamiento de esta estrategia.
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
    dividir_oraciones,
    encabezado,
    imprimir_contextos,
    imprimir_prompt,
    listar_fragmentos,
    pregunta_desde_argv,
    preparar_salida,
    sufijo_compartido,
    vista,
)


NOMBRE = "Chunking recursivo"

# Más holgado que la ventana fija de la lección 1 para que entre un
# párrafo entero. El solapamiento tiene que poder guardar ese párrafo:
# en este manual rondan los 300–400 caracteres. Si el solapamiento es
# más chico que la pieza, se descarta entera y el solapamiento no se ve.
# Bajar TAMANO a 300 obliga al algoritmo a descender hasta oraciones
# y palabras: esa es la parte recursiva.
TAMANO = 1000
SOLAPAMIENTO = 450
TOP_K = 2

# De mayor unidad de sentido a menor. El string vacío significa
# "ya no hay estructura: cortá por carácter".
SEPARADORES = ("SECCION", "\n\n", "\n", "ORACION", " ", "")
NOMBRES_SEPARADORES = {
    "SECCION": "sección ##",
    "\n\n": "párrafo",
    "\n": "línea",
    "ORACION": "oración",
    " ": "palabra",
    "": "carácter",
}

PREGUNTA = (
    "¿Cuántas renovaciones permite la colección general y qué lo impide?"
)


def _aparece(texto: str, separador: str) -> bool:
    """Dice si tiene sentido partir ``texto`` con ese separador."""

    if separador == "SECCION":
        return texto.startswith("## ") or "\n## " in texto
    if separador == "ORACION":
        return len(dividir_oraciones(texto)) > 1
    if separador == "":
        return True
    return separador in texto


def _elegir_separador(texto: str, separadores: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Elige el separador más grueso que realmente parte el texto.

    Devuelve también la cola de la lista. La llamada recursiva solo
    puede usar separadores más finos: si volviera a probar el mismo,
    un bloque que no se divide entraría en un bucle.
    """

    for indice, separador in enumerate(separadores):
        if separador == "":
            return "", ()
        if _aparece(texto, separador):
            return separador, tuple(separadores[indice + 1 :])
    return "", ()


def _dividir(texto: str, separador: str) -> list[str]:
    """Aplica un separador y devuelve las partes sin vacíos.

    La sección se detecta con una mirada hacia adelante (``?=``) para no
    comerse los ``##`` del título. Las oraciones salen del mismo
    cortador que usa la lección 3, así "oración" significa lo mismo en
    las dos clases.
    """

    import re

    if separador == "SECCION":
        partes = re.split(r"(?=^## )", texto.strip(), flags=re.MULTILINE)
    elif separador == "ORACION":
        partes = dividir_oraciones(texto)
    else:
        partes = texto.split(separador)
    return [parte.strip() for parte in partes if parte.strip()]


def _separador_de_union(separador: str) -> str:
    """Con qué texto se vuelven a pegar las partes al armar un fragmento.

    Al dividir oraciones o palabras el separador original se pierde
    (el punto ya quedó pegado a la oración; el espacio, no). Por eso
    esas dos se reúnen con un espacio. Las secciones y los párrafos se
    reúnen con una línea en blanco, que es lo que las separaba.
    """

    if separador in {"SECCION", "\n\n"}:
        return "\n\n"
    if separador in {"ORACION", " "}:
        return " "
    return separador


def _largo(piezas: list[str], separador: str) -> int:
    """Largo que tendrían las piezas ya unidas, separadores incluidos."""

    if not piezas:
        return 0
    return sum(len(pieza) for pieza in piezas) + len(separador) * (len(piezas) - 1)


def _fusionar(
    partes: list[str], separador: str, tamano: int, solapamiento: int
) -> list[str]:
    """Junta partes chicas en fragmentos que no superen ``tamano``.

    Cuando la parte siguiente no entra, se cierra el fragmento actual y
    se conservan al final las partes que quepan en ``solapamiento``.
    Esas partes se repiten al inicio del fragmento nuevo. Si una sola
    parte ya es más chica que el solapamiento pero, sumada a lo que
    sigue, se pasa del tamaño, también se descarta: el límite duro es
    el tamaño, no el deseo de solapar.
    """

    fragmentos: list[str] = []
    actual: list[str] = []
    for parte in partes:
        if len(parte) > tamano:
            raise ValueError(
                "Llegó a la fusión una parte más larga que el tamaño máximo. "
                "Hay que partirla con un separador más fino antes."
            )
        no_entra = actual and _largo(actual + [parte], separador) > tamano
        if no_entra:
            fragmentos.append(separador.join(actual))
            while actual and (
                _largo(actual, separador) > solapamiento
                or _largo(actual + [parte], separador) > tamano
            ):
                actual.pop(0)
        actual.append(parte)
    if actual:
        fragmentos.append(separador.join(actual))
    return fragmentos


def chunking_recursivo(
    texto: str,
    tamano: int = TAMANO,
    solapamiento: int = SOLAPAMIENTO,
    separadores: tuple[str, ...] = SEPARADORES,
) -> list[str]:
    """Parte ``texto`` respetando la estructura, de lo grande a lo chico.

    Casos que cortan la recursión:
    - el texto ya entra en ``tamano``: es un fragmento
    - no queda separador: se corta por carácter, como la lección 1
    - el separador elegido no llega a dividir (un solo pedazo): se
      prueba el nivel siguiente en vez de llamarse a sí misma con la
      misma lista
    """

    if tamano <= 0:
        raise ValueError("El tamaño tiene que ser un entero positivo.")
    if solapamiento < 0 or solapamiento >= tamano:
        raise ValueError("El solapamiento tiene que ser >= 0 y menor que el tamaño.")

    texto = texto.strip()
    if not texto:
        return []
    if len(texto) <= tamano:
        return [texto]

    separador, restantes = _elegir_separador(texto, separadores)
    if separador == "":
        # Último recurso. Un token larguísimo, o un texto sin espacios,
        # igual tiene que poder indexarse.
        return [texto[inicio : inicio + tamano] for inicio in range(0, len(texto), tamano)]

    partes = _dividir(texto, separador)
    if len(partes) <= 1:
        if not restantes:
            return [texto[inicio : inicio + tamano] for inicio in range(0, len(texto), tamano)]
        return chunking_recursivo(texto, tamano, solapamiento, restantes)

    union = _separador_de_union(separador)
    fragmentos: list[str] = []
    buffer: list[str] = []

    def volcar_buffer() -> None:
        nonlocal buffer
        if buffer:
            fragmentos.extend(_fusionar(buffer, union, tamano, solapamiento))
            buffer = []

    for parte in partes:
        if len(parte) > tamano:
            # Esta parte no entra ni sola. Se cierra lo que veníamos
            # juntando y se abre la parte con separadores más finos.
            volcar_buffer()
            siguientes = restantes if restantes else ("",)
            fragmentos.extend(
                chunking_recursivo(parte, tamano, solapamiento, siguientes)
            )
        else:
            buffer.append(parte)
    volcar_buffer()
    return [fragmento for fragmento in fragmentos if fragmento.strip()]


def construir_fragmentos(texto: str) -> list[str]:
    """API que usa la comparación entre estrategias."""

    return chunking_recursivo(texto, TAMANO, SOLAPAMIENTO, SEPARADORES)


def contexto_para_prompt(texto: str, pregunta: str, k: int = TOP_K) -> list[dict]:
    """Igual que en la lección 1: el cambio está solo en cómo se corta."""

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
                    "estrategia: chunking recursivo"
                ),
                "coincidencia": fragmento,
                "nota": (
                    "La unidad indexada y la enviada al modelo coinciden. "
                    "El corte intentó respetar secciones, párrafos y oraciones."
                ),
                "terminos": terminos_compartidos(pregunta, fragmento),
            }
        )
    return contextos


def _mostrar_solapamientos(fragmentos: list[str]) -> None:
    """Imprime el texto que de verdad se repite entre vecinos."""

    print("\nSolapamiento efectivo entre fragmentos vecinos")
    hubo = False
    for indice, (anterior, siguiente) in enumerate(zip(fragmentos, fragmentos[1:])):
        repetido = sufijo_compartido(anterior, siguiente)
        if len(repetido) < 40:
            continue
        hubo = True
        print(f"  entre {indice} y {indice + 1} ({len(repetido)} caracteres):")
        print(f"    {vista(repetido, 160)}")
    if not hubo:
        print(
            "  No hubo un tramo largo repetido. Suele pasar si SOLAPAMIENTO "
            "es menor que el párrafo o la oración que se está usando como pieza."
        )


def main() -> None:
    preparar_salida()
    encabezado("Lección 2 · Chunking recursivo")
    print(
        f"Tamaño máximo: {TAMANO}   Solapamiento: {SOLAPAMIENTO}   Top-k: {TOP_K}"
    )
    nombres = " → ".join(NOMBRES_SEPARADORES[separador] for separador in SEPARADORES)
    print("Separadores, de mayor a menor: " + nombres)

    manual = cargar_manual()
    fragmentos = construir_fragmentos(manual)
    demasiado_grandes = [len(fragmento) for fragmento in fragmentos if len(fragmento) > TAMANO]
    if demasiado_grandes:
        raise RuntimeError(
            f"Hay fragmentos por encima de {TAMANO}: {demasiado_grandes}. "
            "El recursivo no está cumpliendo su límite."
        )
    print(f"\nCorpus: datos/manual_lunaria.md ({len(manual)} caracteres)")
    print(f"Ningún fragmento supera {TAMANO} caracteres.")
    listar_fragmentos(fragmentos)
    _mostrar_solapamientos(fragmentos)

    pregunta = pregunta_desde_argv(PREGUNTA)
    print(f"\nPregunta: {pregunta}")
    contextos = contexto_para_prompt(manual, pregunta, TOP_K)
    imprimir_contextos(contextos)
    imprimir_prompt(pregunta, contextos)
    print(
        "\nPara seguir: el recursivo corta por la forma (títulos, puntos). "
        "La lección 3 corta cuando cambia el tema, aunque no haya un título."
    )


if __name__ == "__main__":
    main()
