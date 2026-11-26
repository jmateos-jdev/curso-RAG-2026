"""Chunking jerárquico: se busca con el hijo chico y se responde con el padre.

Los fragmentos hijos son cortos, así el embedding apunta a una idea. Lo que
vuelve al modelo es el padre, más largo, guardado en un almacén clave-valor.
Acá los dos almacenes son memoria. En el laboratorio el vector store de los
hijos se cambia por Chroma; el docstore del padre puede ser Redis o SQL.
"""

from pathlib import Path

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXTO = Path(__file__).with_name("texto_reglamento.md").read_text(encoding="utf-8")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
)

# El padre conserva la sección; el hijo aísla poco texto para indexarlo.
padre = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
hijo = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=0)

vectorstore = InMemoryVectorStore(embedding=embeddings)
docstore = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=hijo,
    parent_splitter=padre,
)
retriever.add_documents(
    [Document(page_content=TEXTO, metadata={"source": "texto_reglamento.md"})]
)

consulta = "¿Hasta cuándo puedo entregar un trabajo con retraso?"
hijo_hallado = vectorstore.similarity_search(consulta, k=1)[0]
padres = retriever.invoke(consulta)

print(f"Consulta: {consulta}\n")
print("--- Hijo indexado (lo que matchea la búsqueda) ---")
print(hijo_hallado.page_content.strip())
print()
print("--- Padre devuelto al modelo ---")
print(padres[0].page_content.strip())
print()
print(
    f"El hijo tiene {len(hijo_hallado.page_content)} caracteres; "
    f"el padre, {len(padres[0].page_content)}."
)
