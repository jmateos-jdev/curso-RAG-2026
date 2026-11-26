"""Coseno, producto punto y distancia L2 sobre tres frases.

El modelo multilingüe embebe la consulta y los documentos. La similitud
coseno solo mira el ángulo. El producto punto y la distancia L2 también
responden al largo del vector, así que no tienen por qué coincidir con el coseno.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "El API Gateway autentica usuarios y centraliza políticas de rate limiting.",
    "El balanceador distribuye tráfico entrante entre instancias de backend.",
    "La amortización de pasivos financieros se registra en asientos contables.",
]
query = "¿Cómo distribuir y balancear solicitudes en la red?"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
doc_vecs = model.encode(docs, normalize_embeddings=False)
query_vec = model.encode([query], normalize_embeddings=False)
scores = cosine_similarity(query_vec, doc_vecs)[0]
q = query_vec[0]

print(f"Consulta: {query}\n")
for i, texto in enumerate(docs):
    v = doc_vecs[i]
    punto = float(np.dot(q, v))
    l2 = float(np.linalg.norm(q - v))
    print(f"Doc {i + 1} | Coseno: {scores[i]:.4f} | Punto: {punto:.4f} | L2: {l2:.4f}")
    print(f"         {texto}")
    print()
