"""Chunking recursivo: títulos en los metadatos y corte por separadores naturales.

Primero MarkdownHeaderTextSplitter separa por #, ## y ### y guarda esa
jerarquía en los metadatos de cada sección. Después
RecursiveCharacterTextSplitter parte lo que no entra en el tamaño, probando
en orden párrafos, líneas, palabras y, recién al final, caracteres.
El tamaño se mide en tokens del tokenizador cl100k_base, no en caracteres.
"""

from pathlib import Path

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

TEXTO = Path(__file__).with_name("texto_reglamento.md").read_text(encoding="utf-8")
encoder = tiktoken.get_encoding("cl100k_base")

encabezados = [("#", "Modulo"), ("##", "Seccion"), ("###", "Subseccion")]
secciones = MarkdownHeaderTextSplitter(headers_to_split_on=encabezados).split_text(TEXTO)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=8,
    separators=["\n\n", "\n", " ", ""],
    length_function=lambda texto: len(encoder.encode(texto)),
)
fragmentos = splitter.split_documents(secciones)

print(f"{len(fragmentos)} fragmentos (hasta 40 tokens, solapamiento de 8)\n")
for i, frag in enumerate(fragmentos, 1):
    tokens = len(encoder.encode(frag.page_content))
    print(f"--- Fragmento {i} ({tokens} tokens) ---")
    print(f"metadatos: {frag.metadata}")
    print(frag.page_content)
    print()
