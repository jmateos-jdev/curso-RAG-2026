#!/usr/bin/env python3
"""
Lección 3 — Chunking semántico
==============================

Las dos lecciones anteriores deciden el corte con la forma del texto:
una cuenta de caracteres, o una lista de separadores. El chunking
semántico mira el contenido. La idea (Greg Kamradt, y después
LangChain y LlamaIndex) es:

    1. Partir el documento en oraciones.
    2. Representar cada oración con un vector.
    3. Medir la distancia entre oraciones vecinas.
    4. Cortar donde esa distancia pega un salto: ahí cambió el tema.
    5. Dejar juntas las oraciones que siguen hablando de lo mismo.

No es lo mismo que la búsqueda semántica.

- Búsqueda semántica: comparo la pregunta con los fragmentos.
- Chunking semántico: comparo cada oración con la oración de al lado
  para decidir el corte, antes de que exista una pregunta.

En este curso la búsqueda es siempre TF-IDF, también en esta lección.
Si no, no se podría saber si un mejor resultado vino del corte o del
buscador. El vector que decide los cortes es intercambiable:

- Por defecto, TF-IDF sobre las oraciones. No hace falta instalar nada.
  Mide parecido léxico, así que "semántico" acá hay que leerlo como
  "por salto de tema según las palabras". Con un manual que cambia de
  vocabulario en cada sección, el salto se ve igual.
- Si exportás USAR_MODELO_SEMANTICO=1 y tenés sentence-transformers,
  las distancias salen de un modelo de oraciones multilingüe. El resto
  del algoritmo no se toca.

El umbral por defecto es un percentil de las distancias: se cortan los
saltos más grandes y se ignoran las variaciones chicas. También está
el método por desvío estándar: se activa cambiando ``METODO``.

Con TF-IDF, comparar la oración sola contra la siguiente casi siempre
da distancia 1, porque dos frases vecinas no comparten términos. El
percentil deja de distinguir. Por eso ``BUFFER`` vale 1: cada medición
incluye a la vecina, igual que el valor por defecto de LangChain. Las
oraciones que después se agrupan siguen siendo las originales.

Cómo correrlo
-------------
    python ejemplos/03_chunking_semantico.py
    USAR_MODELO_SEMANTICO=1 python ejemplos/03_chunking_semantico.py

Qué mirar en la salida
----------------------
- Los saltos más grandes. Algunos son un cambio de tema (de la multa
  diaria a la suspensión). Otros solo dejan de repetir palabras: TF-IDF
  no entiende que "allí" sigue hablando del mismo lugar.
- Que los fragmentos ya no tienen todos el mismo largo.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from rag_curso.embeddings import (
    CodificadorDeOraciones,
    IndiceTfidf,
    buscar,
    similitud,
    terminos_compartidos,
)
from rag_curso.texto import (
    cargar_manual,
    dividir_oraciones,
    encabezado,
    imprimir_contextos,
    imprimir_prompt,
    listar_fragmentos,
    pregunta_desde_argv,
    preparar_salida,
    vista,
)


NOMBRE = "Chunking semántico"

# Percentil 80: se consideran corte las distancias del 20 % más alto.
# Subirlo (90) deja fragmentos más largos. Bajarlo (60) parte de más.
PERCENTIL_DE_CORTE = 80
METODO = "percentil"  # la otra opción implementada es "desvio"

# 0 compara la oración sola con la siguiente. Con TF-IDF eso deja casi
# todas las distancias en 1 y el percentil no puede elegir.
# 1 imita el buffer de LangChain: la medición incluye a la vecina.
# Las oraciones que se agrupan siguen siendo las originales.
BUFFER = 1

# Una oración de una o dos palabras ("No.") no tiene términos y parece
# un cambio total de tema. Se pega a la anterior antes de medir.
MINIMO_ORACION = 25

# Después del corte, un fragmento diminuto se junta con el siguiente.
# Evita indexar una sola frase suelta sin disimular los cortes grandes.
MINIMO_FRAGMENTO = 80

# Red de seguridad. El criterio principal es el salto de tema; este
# tope solo actúa si un tema se extiende más de lo que queremos
# meter en un fragmento.
TAMANO_MAXIMO = 1300
TOP_K = 2

PREGUNTA = "¿Qué condiciones exige la Sala Ónice para ver un manuscrito?"


def pegar_titulos(oraciones: list[str]) -> list[str]:
    """Une cada título con la oración que le sigue.

    Un título corto ("## Multas y suspensión del carnet") no dice el
    reglamento, pero anuncia el tema. Si se mide solo, la distancia con
    los dos lados es alta y genera un fragmento que es apenas el título.
    Pegarlo a la primera oración del apartado lo deja en el fragmento
    correcto.
    """

    pegadas: list[str] = []
    indice = 0
    while indice < len(oraciones):
        actual = oraciones[indice]
        if actual.startswith("#") and indice + 1 < len(oraciones):
            pegadas.append(actual + "\n" + oraciones[indice + 1])
            indice += 2
            continue
        pegadas.append(actual)
        indice += 1
    return pegadas


def absorber_oraciones_cortas(oraciones: list[str], minimo: int = MINIMO_ORACION) -> list[str]:
    """Pega a la oración anterior las que casi no tienen texto.

    "No." es una respuesta, no un tema. Sola, el índice no tiene
    términos para representarla y la distancia con cualquier vecina
    sale máxima. Pegarla a la pregunta deja el par junto, que es
    como se lee.
    """

    if not oraciones:
        return []
    absorbidas = [oraciones[0]]
    for oracion in oraciones[1:]:
        if len(oracion) < minimo:
            absorbidas[-1] = absorbidas[-1] + " " + oracion
        else:
            absorbidas.append(oracion)
    return absorbidas


def fusionar_fragmentos_cortos(fragmentos: list[str], minimo: int = MINIMO_FRAGMENTO) -> list[str]:
    """Junta un fragmento diminuto con el que sigue.

    Es un piso práctico, no el criterio de corte. Un resto de una
    línea no merece un vector propio; el salto que lo separó queda
    visible en la lista de distancias.
    """

    if not fragmentos:
        return []
    fusionados: list[str] = []
    pendiente = fragmentos[0]
    for fragmento in fragmentos[1:]:
        if len(pendiente) < minimo:
            pendiente = pendiente + "\n" + fragmento
        else:
            fusionados.append(pendiente)
            pendiente = fragmento
    if fusionados and len(pendiente) < minimo:
        fusionados[-1] = fusionados[-1] + "\n" + pendiente
    else:
        fusionados.append(pendiente)
    return fusionados


def textos_para_medir(oraciones: list[str], buffer_size: int) -> list[str]:
    """Arma el texto cuyo vector se compara con el del vecino.

    Con buffer 0, ese texto es la oración. Con buffer 1, es la oración
    más la anterior y la siguiente. El buffer no cambia los grupos: solo
    estabiliza la medición.
    """

    if buffer_size <= 0:
        return list(oraciones)
    medidos = []
    for indice in range(len(oraciones)):
        inicio = max(0, indice - buffer_size)
        fin = min(len(oraciones), indice + buffer_size + 1)
        medidos.append(" ".join(oraciones[inicio:fin]))
    return medidos


def percentil(valores: list[float], p: float) -> float:
    """Percentil lineal, sin numpy. ``p`` va de 0 a 100."""

    if not valores:
        return 0.0
    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return ordenados[0]
    posicion = (len(ordenados) - 1) * (p / 100.0)
    inferior = math.floor(posicion)
    superior = math.ceil(posicion)
    if inferior == superior:
        return ordenados[inferior]
    peso = posicion - inferior
    return ordenados[inferior] * (1.0 - peso) + ordenados[superior] * peso


def calcular_umbral(distancias: list[float], metodo: str, percentil_corte: float) -> float:
    """Define a partir de qué distancia vecina se abre un fragmento nuevo.

    Percentil: corto los saltos relativamente grandes de ESTE documento.
    No hay un 0.35 mágico que sirva para cualquier corpus: un manual
    homogéneo y un libro de temas mezclados tienen escalas distintas.

    Desvío: corto lo que está a más de una desviación estándar por
    encima de la media. Es la otra regla clásica; cambia ``METODO``
    para verla en la misma salida.
    """

    if not distancias:
        return 1.0
    if metodo == "percentil":
        return percentil(distancias, percentil_corte)
    if metodo == "desvio":
        media = sum(distancias) / len(distancias)
        varianza = sum((distancia - media) ** 2 for distancia in distancias) / len(distancias)
        return media + math.sqrt(varianza)
    raise ValueError(f"Método desconocido: {metodo!r}. Usá 'percentil' o 'desvio'.")


def crear_codificador():
    """Elige quién mide el parecido entre oraciones vecinas.

    La búsqueda posterior no usa este objeto: sigue en ``buscar()``,
    que es TF-IDF, para que las lecciones sean comparables.
    """

    if os.environ.get("USAR_MODELO_SEMANTICO") == "1":
        return CodificadorDeOraciones(), "modelo de oraciones (sentence-transformers)"
    return IndiceTfidf(), "TF-IDF sobre las oraciones"


def distancias_entre_vecinas(oraciones: list[str], codificador, buffer_size: int) -> list[float]:
    """Distancia = 1 - similitud, entre cada oración y la siguiente.

    0 significa "dicen casi lo mismo" con el codificador elegido.
    1 significa "no comparten dirección". Valores negativos del coseno
    se recortan a distancia 1: para decidir un corte no necesitamos
    distinguir "poco relacionado" de "opuesto".
    """

    codificador.ajustar(textos_para_medir(oraciones, buffer_size))
    vectores = codificador.encode(textos_para_medir(oraciones, buffer_size))
    distancias = []
    for indice in range(len(vectores) - 1):
        cercania = similitud(vectores[indice], vectores[indice + 1])
        distancias.append(min(1.0, max(0.0, 1.0 - cercania)))
    return distancias


def _empaquetar(oraciones: list[str], tamano_maximo: int) -> list[str]:
    """Parte un grupo de oraciones si, ya unidas, superan el tope.

    Una oración sola se queda junta aunque supere el tope: volver a
    cortarla por carácter desharía el criterio semántico.
    """

    paquetes: list[str] = []
    actual: list[str] = []
    largo = 0
    for oracion in oraciones:
        extra = len(oracion) + (1 if actual else 0)
        if actual and largo + extra > tamano_maximo:
            paquetes.append("\n".join(actual))
            actual = [oracion]
            largo = len(oracion)
            continue
        actual.append(oracion)
        largo += extra
    if actual:
        paquetes.append("\n".join(actual))
    return paquetes


def agrupar_por_saltos(
    oraciones: list[str], distancias: list[float], umbral: float, tamano_maximo: int
) -> list[str]:
    """Abre un fragmento nuevo cuando la distancia vecina supera el umbral."""

    if not oraciones:
        return []
    grupos: list[list[str]] = [[oraciones[0]]]
    for indice, distancia in enumerate(distancias):
        if distancia >= umbral and grupos[-1]:
            grupos.append([oraciones[indice + 1]])
        else:
            grupos[-1].append(oraciones[indice + 1])

    fragmentos: list[str] = []
    for grupo in grupos:
        fragmentos.extend(_empaquetar(grupo, tamano_maximo))
    return [fragmento for fragmento in fragmentos if fragmento.strip()]


def analizar(texto: str) -> dict:
    """Corre el chunking semántico y devuelve también las mediciones.

    La lección imprime los saltos. La comparación entre estrategias solo
    necesita ``fragmentos``, pero tirar las distancias obligaría a
    calcularlas dos veces o a esconderlas.
    """

    oraciones = absorber_oraciones_cortas(pegar_titulos(dividir_oraciones(texto)))
    codificador, nombre_codificador = crear_codificador()
    if len(oraciones) < 2:
        return {
            "oraciones": oraciones,
            "distancias": [],
            "umbral": 1.0,
            "fragmentos": [texto.strip()] if texto.strip() else [],
            "codificador": nombre_codificador,
        }
    distancias = distancias_entre_vecinas(oraciones, codificador, BUFFER)
    umbral = calcular_umbral(distancias, METODO, PERCENTIL_DE_CORTE)
    fragmentos = fusionar_fragmentos_cortos(
        agrupar_por_saltos(oraciones, distancias, umbral, TAMANO_MAXIMO)
    )
    return {
        "oraciones": oraciones,
        "distancias": distancias,
        "umbral": umbral,
        "fragmentos": fragmentos,
        "codificador": nombre_codificador,
    }


def construir_fragmentos(texto: str) -> list[str]:
    """API que usa la comparación entre estrategias."""

    return analizar(texto)["fragmentos"]


def contexto_para_prompt(texto: str, pregunta: str, k: int = TOP_K) -> list[dict]:
    """Busca sobre los fragmentos semánticos con el TF-IDF común del curso."""

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
                    "estrategia: chunking semántico"
                ),
                "coincidencia": fragmento,
                "nota": (
                    "El corte salió de saltos entre oraciones vecinas. "
                    "La búsqueda de la pregunta sigue siendo TF-IDF, como en el resto del curso."
                ),
                "terminos": terminos_compartidos(pregunta, fragmento),
            }
        )
    return contextos


def _mostrar_saltos(oraciones: list[str], distancias: list[float], umbral: float) -> None:
    """Muestra los saltos más grandes y marca cuáles se convierten en corte."""

    print(f"\nUmbral de corte: {umbral:.3f}  (método: {METODO}, buffer: {BUFFER})")
    print(
        "La distancia mide vocabulario compartido entre oraciones vecinas. "
        "Un salto alto puede ser un tema nuevo o solo palabras distintas "
        "para la misma idea."
    )
    orden = sorted(range(len(distancias)), key=lambda i: distancias[i], reverse=True)
    print("Saltos más grandes entre oraciones vecinas:")
    for indice in orden[:8]:
        marca = "CORTE" if distancias[indice] >= umbral else "sigue"
        print(f"  {distancias[indice]:.3f}  {marca}")
        print(f"     A: {vista(oraciones[indice], 110)}")
        print(f"     B: {vista(oraciones[indice + 1], 110)}")


def main() -> None:
    preparar_salida()
    encabezado("Lección 3 · Chunking semántico")
    print(
        f"Método: {METODO}   Percentil: {PERCENTIL_DE_CORTE}   "
        f"Buffer: {BUFFER}   Tope: {TAMANO_MAXIMO} caracteres   Top-k: {TOP_K}"
    )

    manual = cargar_manual()
    resultado = analizar(manual)
    print(f"Codificador de las distancias: {resultado['codificador']}")
    print(f"Oraciones medidas: {len(resultado['oraciones'])}")
    print(f"Corpus: datos/manual_lunaria.md ({len(manual)} caracteres)")
    _mostrar_saltos(resultado["oraciones"], resultado["distancias"], resultado["umbral"])
    listar_fragmentos(resultado["fragmentos"])

    pregunta = pregunta_desde_argv(PREGUNTA)
    print(f"\nPregunta: {pregunta}")
    contextos = contexto_para_prompt(manual, pregunta, TOP_K)
    imprimir_contextos(contextos)
    imprimir_prompt(pregunta, contextos)
    print(
        "\nPara seguir: estos fragmentos ya son la unidad que se busca y "
        "la que se le manda al modelo. La lección 4 separa las dos cosas."
    )


if __name__ == "__main__":
    main()
