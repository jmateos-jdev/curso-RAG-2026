"""Búsqueda por similitud para las lecciones.

Todas las estrategias de este curso recuperan fragmentos con el mismo
índice TF-IDF. Es una decisión de clase: si la respuesta cambia entre
una lección y otra, la causa es el corte del documento, no un buscador
distinto.

TF-IDF no entiende sinónimos. "Extender el plazo" no se acerca a
"renovación" salvo que compartan palabras. Un embedding de oraciones
(OpenAI, Cohere, sentence-transformers) sí podría. Abajo queda el
gancho para enchufar uno, pero el recorrido por defecto no lo necesita
y no descarga modelos.

Los vectores que devuelve este módulo están normalizados: su similitud
coseno es el producto punto. El número NO es una probabilidad.
"""

from __future__ import annotations

import math
import re
from collections import Counter


# Palabras que aparecen en casi cualquier oración en español.
# Sacarlas evita que "de", "la" o "que" decidan el ranking.
# No es parte del chunking: es una limpieza del buscador de la clase.
STOPWORDS = frozenset(
    """
    a al algo algunas algunos ante antes aquel aquella aquellas aquellos
    aqui aquí así aunque bien cada como con contra cual cuales cuando
    de del desde donde el ella ellas ellos en entre era erais eran
    es esa esas ese eso esos esta estas este esto estos fue fueron ha
    han hasta hay la las le les lo los mas más me mi mis mucho muy
    no nos o os otra otras otro otros para pero poco por porque que
    quien quienes qué cómo cuándo dónde cuál cuáles cuánto cuántos
    cuánta cuántas se sea según ser si sido sin sobre sr sra su sus te
    tiene tienen tu tus un una unas unos y ya yo
    """.split()
)

_TOKEN = re.compile(r"[0-9]+|[a-záéíóúüñ]+", re.IGNORECASE)


def tokenizar(texto: str) -> list[str]:
    """Pasa un texto a la lista de términos que el índice sí considera.

    Separamos números y palabras para que "14" y "días" sean dos pistas
    distintas. No sacamos acentos ni aplicamos stemming: "renovación" y
    "renovar" siguen siendo términos diferentes. Esa brecha es real en
    los buscadores léxicos y conviene mostrarla, no esconderla.
    """

    terminos = []
    for crudo in _TOKEN.findall(texto.lower()):
        if crudo in STOPWORDS:
            continue
        if len(crudo) == 1 and not crudo.isdigit():
            continue
        terminos.append(crudo)
    return terminos


def _normalizar(vector: dict[str, float]) -> dict[str, float]:
    """Lleva el vector a longitud 1 para que el producto punto sea el coseno."""

    norma = math.sqrt(sum(valor * valor for valor in vector.values()))
    if norma == 0.0:
        return {}
    return {termino: valor / norma for termino, valor in vector.items()}


def vectorizar(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Construye el vector TF-IDF de un texto ya tokenizado.

    TF de un término = veces que aparece / cantidad de términos del texto.
    Dividir por el largo evita que un fragmento gane solo por ser más largo.
    IDF llega ya calculado sobre el corpus indexado. Un término que no
    estaba en ese corpus se descarta: no tenemos un peso honesto para él.
    """

    if not tokens:
        return {}
    cuentas = Counter(tokens)
    total = len(tokens)
    crudo: dict[str, float] = {}
    for termino, cuenta in cuentas.items():
        peso_idf = idf.get(termino)
        if peso_idf is None:
            continue
        crudo[termino] = (cuenta / total) * peso_idf
    return _normalizar(crudo)


class IndiceTfidf:
    """Índice léxico en memoria, ajustado al corpus que se le pasa.

    El ajuste (fit) mira todos los documentos y decide el IDF.
    Después, cada texto —incluido el de la pregunta— se transforma
    con ESE mismo IDF. Si se mezclan dos ajustes distintos, los
    vectores dejan de ser comparables.
    """

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.cantidad_documentos = 0

    def ajustar(self, textos: list[str]) -> None:
        """Calcula el IDF a partir de los textos que vamos a indexar.

        Fórmula suavizada, la misma idea que usa scikit-learn:
            idf(t) = log((1 + N) / (1 + df(t))) + 1
        N es la cantidad de fragmentos. df(t) es en cuántos fragmentos
        aparece el término. Un término que está en todos los fragmentos
        pesa poco; uno que está en un solo fragmento pesa mucho.
        """

        documentos = [tokenizar(texto) for texto in textos]
        self.cantidad_documentos = len(documentos)
        frecuencia_documental: Counter[str] = Counter()
        for tokens in documentos:
            frecuencia_documental.update(set(tokens))
        n = self.cantidad_documentos
        self.idf = {
            termino: math.log((1.0 + n) / (1.0 + df)) + 1.0
            for termino, df in frecuencia_documental.items()
        }

    def encode_uno(self, texto: str) -> dict[str, float]:
        """Vector de un solo texto. Hay que haber llamado a ``ajustar`` antes."""

        if not self.idf:
            raise RuntimeError(
                "El índice no está ajustado. Llamá a ajustar() con los fragmentos."
            )
        return vectorizar(tokenizar(texto), self.idf)

    def encode(self, textos: list[str]) -> list[dict[str, float]]:
        """Vector de cada texto, todos con el mismo IDF."""

        return [self.encode_uno(texto) for texto in textos]


class CodificadorDeOraciones:
    """Embeddings neuronales opcionales, para la lección de chunking semántico.

    No participa del recorrido por defecto. Para usarlo:

        pip install sentence-transformers
        USAR_MODELO_SEMANTICO=1 python ejemplos/03_chunking_semantico.py

    El modelo multilingüe pesa unos cientos de MB y se descarga la
    primera vez. ``ajustar`` no hace nada: a diferencia de TF-IDF, el
    modelo ya trae los pesos aprendidos y no mira nuestro corpus.
    """

    def __init__(self, modelo: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Falta sentence-transformers. Instálalo o corré la lección "
                "sin USAR_MODELO_SEMANTICO."
            ) from error
        self.modelo = SentenceTransformer(modelo)

    def ajustar(self, textos: list[str]) -> None:
        """Existe para compartir interfaz con IndiceTfidf. El modelo ya viene entrenado."""

        return None

    def encode(self, textos: list[str]) -> list[list[float]]:
        """Devuelve un vector denso y normalizado por texto."""

        vectores = self.modelo.encode(list(textos), normalize_embeddings=True)
        return [vector.tolist() for vector in vectores]

    def encode_uno(self, texto: str) -> list[float]:
        return self.encode([texto])[0]


def similitud(izquierda, derecha) -> float:
    """Producto punto entre dos vectores ya normalizados.

    Acepta el diccionario ralo de TF-IDF o la lista densa de un modelo
    de oraciones. Si alguno de los dos está vacío, la similitud es 0.
    """

    if not izquierda or not derecha:
        return 0.0
    if isinstance(izquierda, dict):
        if len(izquierda) > len(derecha):
            izquierda, derecha = derecha, izquierda
        return sum(
            valor * derecha.get(termino, 0.0) for termino, valor in izquierda.items()
        )
    return sum(a * b for a, b in zip(izquierda, derecha))


def ordenar_por_similitud(vectores: list, consulta, k: int) -> list[tuple[int, float]]:
    """Devuelve los ``k`` índices más parecidos a la consulta.

    Cada item es ``(posicion_en_la_lista, similitud_coseno)``.
    Si ``k`` es mayor que la cantidad de vectores, se devuelven todos.
    """

    if k <= 0:
        raise ValueError("k tiene que ser un entero positivo.")
    puntajes = [
        (indice, similitud(consulta, vector)) for indice, vector in enumerate(vectores)
    ]
    puntajes.sort(key=lambda par: par[1], reverse=True)
    return puntajes[:k]


def buscar(fragmentos: list[str], pregunta: str, k: int) -> list[tuple[int, float]]:
    """Ajusta un TF-IDF sobre ``fragmentos`` y devuelve el top-k de la pregunta.

    Esta es la fase de recuperación, igual en todas las lecciones.
    El índice se arma con los fragmentos ya cortados: si el corte cambia,
    cambian los vectores y puede cambiar el ranking, aunque la pregunta sea la misma.
    """

    if not fragmentos:
        return []
    indice = IndiceTfidf()
    indice.ajustar(fragmentos)
    vectores = indice.encode(fragmentos)
    consulta = indice.encode_uno(pregunta)
    return ordenar_por_similitud(vectores, consulta, k)


def terminos_compartidos(pregunta: str, texto: str) -> list[str]:
    """Términos de la pregunta que también están en el texto recuperado.

    Es una explicación léxica del ranking: muestra por qué TF-IDF eligió
    ese fragmento. Un embedding neuronal puede elegir un fragmento aunque
    esta lista salga vacía; por eso la pista no sustituye al score.
    """

    return sorted(set(tokenizar(pregunta)) & set(tokenizar(texto)))
