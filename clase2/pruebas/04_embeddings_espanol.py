"""Compara un modelo entrenado en inglés con uno multilingüe.

Cinco consultas en español, con modismos y con jerga de sistemas, contra el
mismo corpus corto. El ranking es por similitud coseno. Un modelo que no vio
español técnico suele subir pasajes que comparten palabras sueltas y bajar
el que de verdad responde.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS = [
    "El sueldo se cobra el primer día hábil. La plata entra en la cuenta del banco.",
    "Conseguí laburo nuevo en el área de sistemas y empiezo el lunes.",
    "Si no entregamos el informe hoy estamos en el horno con la gerencia.",
    "El API Gateway autentica usuarios y centraliza las políticas de rate limiting.",
    "El balanceador distribuye el tráfico entrante entre las instancias del backend.",
    "La amortización de pasivos financieros se registra en asientos contables.",
    "El préstamo de la biblioteca dura siete días y se renueva una vez.",
    "Hay que declarar el uso de herramientas de IA en la portada del trabajo.",
    "La base vectorial guarda embeddings y busca los más parecidos a la consulta.",
    "Un índice HNSW arma un grafo para no comparar la consulta contra todos los vectores.",
]
CONSULTAS = [
    "¿Dónde veo si ya me depositaron la plata?",
    "¿Cómo consigo laburo en sistemas?",
    "¿Qué pasa si llegamos tarde y estamos en el horno?",
    "¿Cómo repartir las solicitudes entre los servidores?",
    "¿Para qué sirve un índice HNSW en una base de vectores?",
]
MODELOS = [
    ("Inglés", "all-MiniLM-L6-v2"),
    ("Multilingüe", "paraphrase-multilingual-MiniLM-L12-v2"),
]
TOP = 3


def ranking(model, consulta):
    q = model.encode([consulta])
    docs = model.encode(CORPUS)
    scores = cosine_similarity(q, docs)[0]
    orden = scores.argsort()[::-1][:TOP]
    return [(int(i), float(scores[i])) for i in orden]


cargados = [(nombre, SentenceTransformer(modelo)) for nombre, modelo in MODELOS]

for consulta in CONSULTAS:
    print(f"\nConsulta: {consulta}")
    for nombre, model in cargados:
        print(f"  {nombre}")
        for puesto, (i, score) in enumerate(ranking(model, consulta), 1):
            print(f"    {puesto}. {score:.3f}  {CORPUS[i]}")
