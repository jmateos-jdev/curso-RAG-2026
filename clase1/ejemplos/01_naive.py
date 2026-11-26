"""Chunking naive: corte fijo por cantidad de caracteres.

No mira párrafos, oraciones ni títulos. Parte el texto cada N caracteres,
aunque el corte caiga en medio de una palabra. En un manual real ese N
ronda los 500 caracteres; aquí es más chico para que la fractura se vea
en este reglamento corto.
"""

from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter

TEXTO = Path(__file__).with_name("texto_reglamento.md").read_text(encoding="utf-8")

# separator="" parte en caracteres y rearma bloques de chunk_size.
# En un manual real el corte naive típico ronda los 500 caracteres.
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=80,
    chunk_overlap=0,
    length_function=len,
    strip_whitespace=False,
)

fragmentos = splitter.split_text(TEXTO)

print(f"{len(fragmentos)} fragmentos (corte fijo de 80 caracteres, sin solapamiento)\n")
for i, frag in enumerate(fragmentos, 1):
    print(f"--- Fragmento {i} ({len(frag)} caracteres) ---")
    print(frag)
    print()
