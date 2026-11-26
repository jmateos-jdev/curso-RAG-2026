# Clase 1 — Estrategias de segmentación

Cuatro scripts en `ejemplos/`. Leen `texto_reglamento.md` de esa misma carpeta. Todo corre en memoria: no hay servidor ni API key.

## Instalación

Hace falta Python 3.9 o más nuevo. Desde `clase1/ejemplos`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows, el entorno se activa con `.venv\Scripts\activate`.

Qué instala cada paquete:

| Paquete | Para qué |
| --- | --- |
| `langchain-text-splitters` | `CharacterTextSplitter` en `01`. `MarkdownHeaderTextSplitter` y `RecursiveCharacterTextSplitter` en `02`. Los splitters padre e hijo en `04`. |
| `tiktoken` | `02_recursivo.py` mide el tamaño con el encoding `cl100k_base`. La primera vez puede descargarlo. |
| `langchain-experimental` | `SemanticChunker` de `03_semantico.py`. |
| `langchain-huggingface` | Carga el modelo de embeddings en `03` y `04`. |
| `sentence-transformers` | El modelo que usan esos embeddings. |
| `langchain-classic` | `ParentDocumentRetriever` y el almacén en memoria de `04_jerarquico.py`. |
| `langchain-core` | `Document` y `InMemoryVectorStore` de `04`. |

`03` y `04` usan `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` en CPU. La primera corrida necesita internet. Después el modelo queda en la caché de Hugging Face (`~/.cache/huggingface`) y no se vuelve a descargar. `01` y `02` no bajan un modelo de embeddings.

## Cómo correr los ejemplos

Con el entorno activado y el directorio de trabajo en `clase1/ejemplos`:

```bash
python 01_naive.py
python 02_recursivo.py
python 03_semantico.py
python 04_jerarquico.py
```

- `01_naive.py` corta el reglamento cada 80 caracteres, sin mirar palabras ni títulos.
- `02_recursivo.py` separa por títulos de markdown y después parte por párrafos, líneas y palabras. El tamaño se mide en tokens, no en caracteres.
- `03_semantico.py` corta donde cambia el tema: embebe cada oración y abre un fragmento cuando la distancia con la siguiente pega un salto. La primera vez descarga el modelo.
- `04_jerarquico.py` indexa fragmentos chicos y, al buscar, devuelve el fragmento padre. La primera vez también descarga el modelo.
