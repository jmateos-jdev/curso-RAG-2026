"""Indexa tres documentos en Qdrant en memoria y busca la consulta.

m y ef_construct quedan guardados en la colección. hnsw_ef es el ancho de
la búsqueda. El cliente local (:memory:) resuelve por fuerza bruta, así que
con tres puntos el resultado es el exacto; contra un Qdrant en servidor,
hnsw_ef sí recorre el grafo.
"""

import warnings

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", message="Local mode performs exact")

docs = [
    "El API Gateway autentica usuarios y centraliza políticas de rate limiting.",
    "El balanceador distribuye tráfico entrante entre instancias de backend.",
    "La amortización de pasivos financieros se registra en asientos contables.",
]
query = "¿Cómo distribuir y balancear solicitudes en la red?"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
doc_vecs = model.encode(docs)
query_vec = model.encode([query])

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="kb_infra",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
)
client.upsert(
    collection_name="kb_infra",
    points=[
        PointStruct(id=i, vector=doc_vecs[i].tolist(), payload={"text": docs[i]})
        for i in range(len(docs))
    ],
    wait=True,
)

res = client.query_points(
    collection_name="kb_infra",
    query=query_vec[0].tolist(),
    limit=2,
    search_params=SearchParams(hnsw_ef=64),
)

print(f"Consulta: {query}")
print("Índice: HNSW  m=16  ef_construct=100  hnsw_ef=64\n")
for p in res.points:
    print(f"Hit ID: {p.id} | Score: {p.score:.4f} | {p.payload['text']}")
