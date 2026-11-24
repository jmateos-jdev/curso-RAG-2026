# Curso de RAG: cuatro formas de cortar el documento

Ejemplos para ver, con el mismo manual y el mismo buscador, qué cambia
cuando cambia el chunking.

El corpus es ficticio: el manual de la Biblioteca Universitaria Lunaria.
Las cifras que importan en clase aparecen una sola vez.

| Dato | Dónde está |
| --- | --- |
| Sede Puerto, de martes a sábado, de 10:00 a 18:00 | Sedes y horarios |
| Préstamo de la colección general: 14 días corridos | Préstamos |
| Dos renovaciones, salvo que haya una reserva | Renovaciones |
| Manuscritos: cita de 48 horas, guantes, no salen de la Sala Ónice | Manuscritos |
| Multa de cincuenta centavos de lunar por día | Multas |
| A los 21 días de retraso el carnet queda suspendido | Multas, otro párrafo |

## Cómo correrlo

Hace falta Python 3.9 o más nuevo. No hay que instalar paquetes.

```bash
python ejemplos/01_naive_rag_chunking_fijo.py
python ejemplos/02_chunking_recursivo.py
python ejemplos/03_chunking_semantico.py
python ejemplos/04_parent_document_retrieval.py
python ejemplos/05_comparacion.py
```

Cualquier lección acepta otra pregunta al final del comando:

```bash
python ejemplos/02_chunking_recursivo.py "¿Cuántos libros puedo llevarme?"
```

Los comentarios del código son parte del material. Conviene leer el
archivo, no solo la salida.

## Qué hace cada lección

1. **Naive RAG / chunking fijo.** Ventanas del mismo largo, con
   solapamiento. No mira puntos ni títulos. En este corpus la ventana
   parte una palabra y también parte la oración de los 14 días. El
   horario de la Sede Puerto queda cortado en el primer resultado: entra
   completo solo porque se piden dos fragmentos.
2. **Chunking recursivo.** Intenta cortar por sección, párrafo, línea,
   oración y palabra, en ese orden, y solo baja de nivel si el bloque no
   entra. El solapamiento repite un párrafo entero, no una cantidad fija
   de caracteres.
3. **Chunking semántico.** Mide la distancia entre oraciones vecinas y
   corta en los saltos grandes. No es lo mismo que la búsqueda semántica:
   acá el vector decide el corte, antes de que exista una pregunta.
4. **Parent-document retrieval.** Indexa hijos chicos y, cuando uno
   coincide, manda al modelo la sección completa. La pregunta de la multa
   y la suspensión está armada para que el hijo tenga uno de los dos
   hechos y el padre tenga los dos.

`ejemplos/05_comparacion.py` hace las mismas dos preguntas a las cuatro
estrategias y muestra cuánto texto se indexa y cuánto texto iría al prompt.

## Decisiones de diseño

**No hay llamada a un modelo.** Cada lección imprime el prompt armado
por stuffing: los fragmentos pegados dentro de la instrucción. Se puede
copiar ese prompt a cualquier modelo. El curso corre sin API key.

**La búsqueda es siempre TF-IDF.** Si un resultado cambia entre
lecciones, la causa es el corte. TF-IDF no entiende sinónimos ni
negaciones: la lección 1 lo muestra con una pregunta que pide una
renovación sin usar esa palabra y termina en otra sección, enganchada
por la palabra "entero".

**El corte se mide en caracteres, no en tokens.** En producción el límite
del prompt es en tokens. La forma del algoritmo no cambia; usar
caracteres evita sumar un tokenizador.

**El chunking semántico también usa TF-IDF para las distancias**, con una
ventana de una oración a cada lado. Sin esa ventana, dos frases vecinas
casi no comparten términos, todas las distancias valen 1 y el percentil
no puede elegir dónde cortar. Para medir los saltos con un modelo de
oraciones:

```bash
pip install sentence-transformers
USAR_MODELO_SEMANTICO=1 python ejemplos/03_chunking_semantico.py
```

La búsqueda de la pregunta sigue siendo TF-IDF igual.

## Cómo se llama esto en las librerías

| En este curso | LangChain | LlamaIndex |
| --- | --- | --- |
| Chunking fijo | `CharacterTextSplitter`, `TokenTextSplitter` | `TokenTextSplitter` |
| Chunking recursivo | `RecursiveCharacterTextSplitter` | `SentenceSplitter` |
| Chunking semántico | `SemanticChunker` | `SemanticSplitterNodeParser` |
| Parent-document | `ParentDocumentRetriever` | `AutoMergingRetriever` |

El recursivo de LangChain arranca en párrafos, líneas, espacios y
caracteres. Acá se agregan la sección markdown y la oración porque el
manual tiene títulos. El recorrido es el mismo: separador grueso primero,
fino solo si hace falta.

## Qué probar en clase

- En la lección 1, poner `TOP_K = 1`. El horario de la Sede Puerto deja
  de entrar completo.
- En la lección 2, bajar `TAMANO` a 300. El recursivo deja de quedarse
  en los párrafos y parte oraciones.
- En la lección 3, cambiar `METODO` a `"desvio"` o `PERCENTIL_DE_CORTE`
  a 90. Se ven menos cortes y fragmentos más largos.
- En la lección 4, partir el FAQ en varios `##`. Cada pregunta viaja
  sola, en vez de arrastrar el resto de las preguntas frecuentes.
- Preguntar algo que no esté en el manual. El prompt le pide al modelo
  que lo diga; sin esa frase, el modelo completa con conocimiento general.

## Estructura

```
datos/manual_lunaria.md          corpus de la clase
rag_curso/texto.py               lectura, oraciones y armado del prompt
rag_curso/embeddings.py          TF-IDF y ranking, compartidos
ejemplos/01_...py                el algoritmo de cada estrategia
ejemplos/05_comparacion.py       las cuatro, lado a lado
```
