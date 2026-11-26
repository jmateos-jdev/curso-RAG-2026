# Clase 2 — Bases vectoriales y embeddings

Cuatro scripts en `pruebas/`. No hay que levantar un servidor: Qdrant corre en memoria (`:memory:`) y Chroma usa un cliente efímero.

## Instalación

Hace falta Python 3.9 o más nuevo. Desde `clase2/pruebas`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows, el entorno se activa con `.venv\Scripts\activate`.

Qué instala cada paquete:

| Paquete | Para qué |
| --- | --- |
| `sentence-transformers` | Descarga y carga los modelos de embeddings. |
| `numpy` | Producto punto, distancia L2 y los 5.000 vectores del benchmark. |
| `scikit-learn` | Similitud coseno en `01` y `04`. |
| `qdrant-client` | La colección en memoria de `02_qdrant_hnsw.py`. |
| `chromadb` | Las dos configuraciones HNSW de `03_benchmark_hnsw.py`. |
| `tiktoken` | Está en el requirements, pero ninguno de los cuatro scripts lo importa. Se instala igual con el archivo. |

La primera corrida de un modelo necesita internet. `01`, `02` y `04` bajan `paraphrase-multilingual-MiniLM-L12-v2`. `04` además baja `all-MiniLM-L6-v2`. Después quedan en la caché de Hugging Face (`~/.cache/huggingface`) y no se vuelven a descargar.

## Cómo correr los ejemplos

Con el entorno activado y el directorio de trabajo en `clase2/pruebas`:

```bash
python 01_similitud.py
python 02_qdrant_hnsw.py
python 03_benchmark_hnsw.py
python 04_embeddings_espanol.py
```

- `01_similitud.py` compara coseno, producto punto y distancia L2 sobre las mismas tres frases.
- `02_qdrant_hnsw.py` indexa tres documentos en Qdrant y busca la consulta. Con tres puntos el cliente local resuelve por fuerza bruta: guarda `m` y `ef_construct`, pero no recorre el grafo HNSW.
- `03_benchmark_hnsw.py` sí arma el grafo. Compara una configuración rápida y una de alta fidelidad sobre 5.000 vectores e imprime tiempo de inserción, latencia y Recall@10. Tarda más que el resto porque indexa esos vectores dos veces.
- `04_embeddings_espanol.py` corre las mismas consultas en español contra un modelo entrenado en inglés y contra el multilingüe.
