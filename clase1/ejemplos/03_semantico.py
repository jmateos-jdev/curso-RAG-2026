"""Chunking semántico: corta donde cambia el tema, no al llegar a un tamaño.

SemanticChunker embebe cada oración y mide la distancia coseno con la
siguiente. Un salto grande (poco parecido) marca un corte. El umbral es un
percentil de esas distancias. En documentos largos el temario usa el
percentil 95; este reglamento es corto, así que 70 alcanza para ver los
cambios de tema.

La primera ejecución descarga el modelo de embeddings. No hace falta una API key.

Para umbral estadístico (media + k desviaciones estándar), reemplazar el
splitter por:

    SemanticChunker(
        embeddings,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.5,
    )
"""

from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

TEXTO = Path(__file__).with_name("texto_reglamento.md").read_text(encoding="utf-8")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
)
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)

fragmentos = [frag.strip() for frag in splitter.split_text(TEXTO) if frag.strip()]

print(f"{len(fragmentos)} fragmentos (corte por percentil de distancia entre oraciones)\n")
for i, frag in enumerate(fragmentos, 1):
    print(f"--- Fragmento {i} ({len(frag)} caracteres) ---")
    print(frag)
    print()
