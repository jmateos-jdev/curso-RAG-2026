"""Compara dos HNSW sobre 5.000 vectores aleatorios de dimensión 384.

La configuración rápida arma un grafo chico. La de alta fidelidad arma uno
más denso. Las dos se consultan con el mismo ef de búsqueda, así la
diferencia viene de M y ef_construction. El Recall@10 se mide contra la
búsqueda exacta en NumPy.

Qdrant en :memory: busca por fuerza bruta y no arma el grafo. Este
laboratorio usa Chroma, que indexa con hnswlib y sí honra esos parámetros.
"""

import time

import chromadb
import numpy as np

N = 5_000
D = 384
K = 10
CONSULTAS = 20
# Fijo y bajo a propósito: con ef alto las dos configs aciertan todo y no se ve el trade-off.
HNSW_EF = 16

rng = np.random.default_rng(7)
base = rng.standard_normal((N, D)).astype(np.float32)
base /= np.linalg.norm(base, axis=1, keepdims=True)
queries = rng.standard_normal((CONSULTAS, D)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

configs = [
    ("Rápida", "bench_rapida", 16, 64),
    ("Alta fidelidad", "bench_fiel", 64, 256),
]


def vecinos_exactos(q):
    scores = base @ q
    return set(np.argsort(-scores)[:K].tolist())


def medir(client, nombre, coleccion, m, ef_construct):
    col = client.create_collection(
        name=coleccion,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": m,
            "hnsw:construction_ef": ef_construct,
            "hnsw:search_ef": HNSW_EF,
        },
    )
    ids = [str(i) for i in range(N)]
    vectores = base.tolist()
    t0 = time.perf_counter()
    for inicio in range(0, N, 1000):
        fin = inicio + 1000
        col.add(ids=ids[inicio:fin], embeddings=vectores[inicio:fin])
    insercion = time.perf_counter() - t0

    recalls = []
    t1 = time.perf_counter()
    for q in queries:
        res = col.query(query_embeddings=[q.tolist()], n_results=K)
        hallados = {int(i) for i in res["ids"][0]}
        recalls.append(len(hallados & vecinos_exactos(q)) / K)
    latencia_ms = (time.perf_counter() - t1) / CONSULTAS * 1000
    recall = float(np.mean(recalls))
    print(
        f"{nombre:16}  m={m:<3} ef_construct={ef_construct:<4}  "
        f"inserción {insercion:6.2f} s   "
        f"latencia {latencia_ms:6.2f} ms   "
        f"Recall@{K} {recall:.3f}"
    )


print(f"{N} vectores · dimensión {D} · {CONSULTAS} consultas · ef de búsqueda={HNSW_EF}\n")
client = chromadb.EphemeralClient()
for nombre, coleccion, m, ef_construct in configs:
    medir(client, nombre, coleccion, m, ef_construct)
