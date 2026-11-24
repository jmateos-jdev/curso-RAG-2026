"""Utilidades compartidas del curso de RAG.

Los algoritmos de cada estrategia están en ``ejemplos/``, comentados
como material de clase. Este paquete solo reúne lo que todas las
lecciones necesitan: leer el manual, pasar texto a vectores y armar
el prompt que recibiría el modelo de lenguaje.
"""

from rag_curso.embeddings import (
    IndiceTfidf,
    buscar,
    ordenar_por_similitud,
    similitud,
    terminos_compartidos,
)
from rag_curso.texto import armar_prompt, cargar_manual, vista

__all__ = [
    "IndiceTfidf",
    "armar_prompt",
    "buscar",
    "cargar_manual",
    "ordenar_por_similitud",
    "similitud",
    "terminos_compartidos",
    "vista",
]
