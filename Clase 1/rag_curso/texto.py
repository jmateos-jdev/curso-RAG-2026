"""Lectura del corpus y formato de la salida de las lecciones.

Nada de esto parte el documento: el corte vive en cada ejemplo.
Acá están las piezas que, si se copiaran en las cuatro lecciones,
taparían el algoritmo que queremos mirar.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Raíz del curso: .../RAG/  (este archivo está en .../RAG/rag_curso/)
RAIZ = Path(__file__).resolve().parent.parent
RUTA_MANUAL = RAIZ / "datos" / "manual_lunaria.md"


def preparar_salida() -> None:
    """Fuerza UTF-8 en la consola.

    En Windows, la consola a veces arranca en una codificación vieja y
    los acentos del manual se imprimen rotos. En macOS y Linux no cambia
    nada visible. Se llama al empezar cada lección.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def cargar_manual() -> str:
    """Devuelve el texto completo del manual de ejemplo.

    Un sistema real leería muchos archivos (PDF, HTML, tickets).
    Para la clase alcanza un solo documento largo, porque el tema
    de estas lecciones es cómo se corta, no cómo se parsea cada formato.
    """

    if not RUTA_MANUAL.exists():
        raise FileNotFoundError(
            f"No encuentro el corpus de la clase en {RUTA_MANUAL}."
        )
    return RUTA_MANUAL.read_text(encoding="utf-8")


def pregunta_desde_argv(defecto: str) -> str:
    """Usa la pregunta escrita en la terminal, o la de la lección.

    Ejemplo:
        python ejemplos/01_naive_rag_chunking_fijo.py "¿Abren los lunes en el puerto?"
    """

    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return defecto


def vista(texto: str, largo: int = 180) -> str:
    """Resume un fragmento en una sola línea para poder listarlo."""

    limpio = " ".join(texto.split())
    if len(limpio) <= largo:
        return limpio
    return limpio[: largo - 1] + "…"


def encabezado(titulo: str) -> None:
    """Separa en la terminal el arranque de una lección."""

    linea = "=" * 72
    print(f"\n{linea}\n{titulo}\n{linea}")


def listar_fragmentos(fragmentos: list[str], titulo: str = "Fragmentos indexados") -> None:
    """Imprime cuántos cortes salieron y una vista de cada uno."""

    print(f"\n{titulo}")
    total = len(fragmentos)
    print(f"Cantidad: {total}")
    if total == 0:
        return
    largos = [len(fragmento) for fragmento in fragmentos]
    promedio = sum(largos) // total
    print(
        "Largo en caracteres  "
        f"mínimo {min(largos)}  ·  medio {promedio}  ·  máximo {max(largos)}"
    )
    for indice, fragmento in enumerate(fragmentos):
        print(f"  [{indice:02d}] {len(fragmento):4d} car.  {vista(fragmento, 78)}")


def sufijo_compartido(anterior: str, siguiente: str, tope: int = 500) -> str:
    """Devuelve el texto repetido entre el final de un fragmento y el inicio del otro.

    Sirve para mostrar el solapamiento en la terminal. Recorre de mayor
    a menor y se detiene en la primera coincidencia, que es la más larga.
    """

    limite = min(len(anterior), len(siguiente), tope)
    for largo in range(limite, 0, -1):
        cola = anterior[-largo:]
        if siguiente.startswith(cola):
            return cola
    return ""


def dividir_oraciones(texto: str) -> list[str]:
    """Corta el texto en oraciones usando la puntuación final.

    Es un cortador pedagógico, no un segmentador lingüístico.
    Parte después de `.`, `!` o `?` cuando sigue un espacio o un salto
    de línea. No intenta entender abreviaturas: el manual de la clase
    evita formas como "Sr." a propósito.

    Los títulos de sección (líneas que empiezan con #) se dejan como
    oraciones propias para que el salto de tema pueda apoyarse en ellos.
    """

    import re

    oraciones: list[str] = []
    for bloque in re.split(r"\n+", texto.strip()):
        bloque = bloque.strip()
        if not bloque:
            continue
        if bloque.startswith("#"):
            oraciones.append(bloque)
            continue
        partes = re.split(r"(?<=[.!?])\s+", bloque)
        oraciones.extend(parte.strip() for parte in partes if parte.strip())
    return oraciones


def armar_prompt(pregunta: str, contextos: list[dict]) -> str:
    """Arma el prompt de un RAG por stuffing.

    Stuffing quiere decir: pegar los fragmentos recuperados dentro de
    una sola instrucción y pedirle al modelo que responda con eso.
    No llamamos a ningún modelo. El resultado de estas lecciones es
    justamente este texto: es lo que el modelo vería, además de la
    pregunta. Así el curso corre sin API key y se puede leer el contexto
    que cada estrategia habría entregado.

    Cada item de ``contextos`` necesita las claves ``cita`` y ``texto``.
    """

    if not contextos:
        bloque_contexto = "(no se recuperó ningún fragmento)"
    else:
        partes = []
        for numero, contexto in enumerate(contextos, start=1):
            cita = contexto.get("cita", f"Fragmento {numero}")
            partes.append(f"[{numero}] {cita}\n{contexto['texto'].strip()}")
        bloque_contexto = "\n\n".join(partes)

    return (
        "Sos el asistente de la Biblioteca Universitaria Lunaria.\n"
        "Respondé solo con la información del contexto. "
        "Si el dato no está, decí que no consta en el manual. "
        "No completes con conocimientos generales.\n\n"
        f"Contexto:\n{bloque_contexto}\n\n"
        f"Pregunta: {pregunta}\n\n"
        "Respuesta:"
    )


def imprimir_contextos(contextos: list[dict]) -> None:
    """Muestra qué se inyectaría en el prompt y por qué salió elegido."""

    if not contextos:
        print("\nNo hubo coincidencias.")
        return

    print("\nContexto que recibiría el modelo")
    for numero, contexto in enumerate(contextos, start=1):
        score = contexto.get("score", 0.0)
        print(f"\n--- bloque {numero}  ·  similitud {score:.3f} ---")
        print(contexto.get("cita", ""))
        nota = contexto.get("nota")
        if nota:
            print(nota)
        compartidos = contexto.get("terminos") or []
        if compartidos:
            print("Términos de la pregunta presentes en la coincidencia:")
            print("  " + ", ".join(compartidos))
        coincidencia = contexto.get("coincidencia")
        texto = contexto["texto"]
        if coincidencia is not None and coincidencia.strip() != texto.strip():
            print("Unidad que coincidió con la búsqueda (chica):")
            print(vista(coincidencia, 280))
            print("Unidad que se envía al modelo (grande):")
        print(texto.strip())
        print(f"[{len(texto)} caracteres]")


def imprimir_prompt(pregunta: str, contextos: list[dict]) -> None:
    """Imprime el prompt final, que es el cierre de un RAG sin el modelo."""

    print("\nPrompt armado (stuffing). No se envía a ningún modelo.")
    print("-" * 72)
    print(armar_prompt(pregunta, contextos))
    print("-" * 72)
