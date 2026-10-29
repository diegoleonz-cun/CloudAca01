import glob
import os
import re
import unicodedata
import uuid
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).resolve().parent
CORPUS_DIR = APP_DIR / "corpus"
CHROMA_DIR = APP_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "knowledge_base"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
SEMANTIC_RESULTS = 5
LITERAL_CONTEXT_CHARS = 250

CORPUS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner="Cargando modelo multilingue de embeddings...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    return get_chroma_client().get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def delete_collection_if_exists():
    client = get_chroma_client()
    collection_names = [
        getattr(item, "name", item) for item in client.list_collections()
    ]
    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)


def read_document(file_path):
    extension = Path(file_path).suffix.lower()

    if extension == ".csv":
        try:
            dataframe = pd.read_csv(file_path, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(file_path, dtype=str, encoding="latin-1")

        return dataframe.fillna("").astype(str).to_csv(index=False)

    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    return ""


def corpus_files():
    files = []
    for extension in ("*.txt", "*.log", "*.csv"):
        files.extend(glob.glob(str(CORPUS_DIR / extension)))
    return files


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    overlap = min(max(overlap, 0), chunk_size - 1)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()
    ]
    chunks = []
    current = ""

    def append_chunk(value):
        value = value.strip()
        if value:
            chunks.append(value)

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                append_chunk(current)
                current = ""

            remaining = paragraph
            while len(remaining) > chunk_size:
                split_at = remaining.rfind(" ", 0, chunk_size + 1)
                if split_at <= 0:
                    split_at = chunk_size
                append_chunk(remaining[:split_at])
                next_start = max(0, split_at - overlap)
                while next_start > 0 and not remaining[next_start - 1].isspace():
                    next_start -= 1
                remaining = remaining[next_start:].lstrip()
            current = remaining
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            append_chunk(current)
            tail_start = max(0, len(current) - overlap)
            while tail_start > 0 and not current[tail_start - 1].isspace():
                tail_start -= 1
            tail = current[tail_start:].strip()
            current = f"{tail}\n\n{paragraph}" if tail else paragraph

    append_chunk(current)
    return chunks


def rebuild_index(progress_bar=None, status_text=None):
    delete_collection_if_exists()
    collection = get_chroma_client().create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    files = corpus_files()
    total_chunks = 0
    embedder = load_embedder()

    for file_index, file_path in enumerate(files):
        source = os.path.basename(file_path)
        content = read_document(file_path)

        if not content.strip():
            continue

        chunks = split_text(content)
        embeddings = embedder.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[
                {"source": source, "chunk_index": index}
                for index in range(len(chunks))
            ],
        )
        total_chunks += len(chunks)

        if progress_bar is not None and files:
            progress_bar.progress((file_index + 1) / len(files))
        if status_text is not None:
            status_text.write(f"Indexando: {source}")

    return f"Corpus indexado correctamente. Archivos: {len(files)}. Fragmentos: {total_chunks}."


def normalize_for_search(text):
    normalized = []
    source_positions = []

    for index, character in enumerate(text):
        decomposed = unicodedata.normalize("NFD", character.casefold())
        for item in decomposed:
            if not unicodedata.combining(item):
                normalized.append(item)
                source_positions.append(index)

    return "".join(normalized), source_positions


def literal_matches(query):
    normalized_query, _ = normalize_for_search(query.strip())
    if not normalized_query:
        return []

    matches = []
    for file_path in corpus_files():
        content = read_document(file_path)
        if not content.strip():
            continue

        normalized_content, positions = normalize_for_search(content)
        search_from = 0

        while True:
            match_start = normalized_content.find(normalized_query, search_from)
            if match_start < 0:
                break

            start = positions[match_start]
            end = positions[match_start + len(normalized_query) - 1] + 1
            context_start = max(0, start - LITERAL_CONTEXT_CHARS)
            context_end = min(len(content), end + LITERAL_CONTEXT_CHARS)
            matches.append(
                {
                    "source": os.path.basename(file_path),
                    "position": start,
                    "content": content[context_start:context_end].strip(),
                }
            )
            search_from = match_start + 1

    return matches


def best_match(query):
    collection = get_collection()
    result_count = collection.count()
    if result_count == 0:
        return None

    query_embedding = load_embedder().encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(SEMANTIC_RESULTS * 3, result_count),
    )


def chunks_overlap(first_document, second_document):
    first, _ = normalize_for_search(first_document)
    second, _ = normalize_for_search(second_document)
    first = re.sub(r"\s+", " ", first).strip()
    second = re.sub(r"\s+", " ", second).strip()
    limit = min(CHUNK_OVERLAP, len(first), len(second))
    if limit == 0:
        return False

    minimum = min(80, limit)
    return any(
        first[-size:] == second[:size]
        for size in range(limit, minimum - 1, -1)
    )


def build_response(query):
    exact_matches = literal_matches(query)
    results = best_match(query)
    semantic_documents = []

    if results:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        normalized_query, _ = normalize_for_search(query)
        selected_chunks = []

        for document, metadata, distance in zip(documents, metadatas, distances):
            if len(semantic_documents) >= SEMANTIC_RESULTS:
                break

            normalized_document, _ = normalize_for_search(document)
            if normalized_query and normalized_query in normalized_document:
                continue

            duplicate = False
            for previous_metadata, previous_document in selected_chunks:
                if metadata["source"] != previous_metadata["source"]:
                    continue
                if normalized_document == normalize_for_search(previous_document)[0]:
                    duplicate = True
                    break
                if abs(metadata["chunk_index"] - previous_metadata["chunk_index"]) == 1:
                    earlier, later = (
                        (previous_document, document)
                        if previous_metadata["chunk_index"] < metadata["chunk_index"]
                        else (document, previous_document)
                    )
                    if chunks_overlap(earlier, later):
                        duplicate = True
                        break

            if duplicate:
                continue

            selected_chunks.append((metadata, document))
            semantic_documents.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

    return {
        "exact_matches": exact_matches,
        "semantic_documents": semantic_documents,
        "has_index": get_collection().count() > 0,
    }


def render_section_title(title):
    st.markdown(
        f'<h3 style="color: #e85d04; font-weight: 700;">{title}</h3>',
        unsafe_allow_html=True,
    )


def render_results(query, response):
    st.write("Tu consulta:", query)

    exact_matches = response["exact_matches"]
    if exact_matches:
        render_section_title(
            f"COINCIDENCIAS LITERALES ENCONTRADAS: {len(exact_matches)}"
        )
        for index, match in enumerate(exact_matches, start=1):
            detail = (
                f"Coincidencia literal #{index} (posicion {match['position']})"
                f" - Archivo: {match['source']}"
            )
            st.table(
                pd.DataFrame(
                    [
                        {"Campo": "Detalle", "Informacion": detail},
                        {"Campo": "Contenido", "Informacion": match["content"]},
                    ]
                )
            )
    else:
        st.write("No se encontraron coincidencias literales.")

    semantic_documents = response["semantic_documents"]
    if semantic_documents:
        render_section_title("RESULTADOS SEMANTICAMENTE RELACIONADOS")
        for index, result in enumerate(semantic_documents, start=1):
            detail = (
                f"Resultado semantico #{index}"
                f" - Distancia Coseno: {result['distance']:.6f}"
                f" - Metadatos: {result['metadata']}"
            )
            st.table(
                pd.DataFrame(
                    [
                        {"Campo": "Detalle", "Informacion": detail},
                        {"Campo": "Contenido", "Informacion": result["document"]},
                    ]
                )
            )
    else:
        st.write("No hay resultados semanticos adicionales.")

    if not response["has_index"]:
        st.info("El corpus aun no tiene documentos indexados.")


def save_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        filename = os.path.basename(uploaded_file.name.replace("\\", "/"))
        destination = CORPUS_DIR / filename
        destination.write_bytes(uploaded_file.getvalue())


def main():
    st.set_page_config(page_title="Busqueda Hibrida", page_icon="🔎", layout="wide")
    st.title("🔎 Busqueda Hibrida")
    st.caption(
        "Consulta documentos TXT, LOG y CSV mediante coincidencias literales "
        "y busqueda semantica."
    )

    corpus_tab, chat_tab = st.tabs(["Corpus", "Chat"])

    with corpus_tab:
        uploaded_files = st.file_uploader(
            "Subir archivos TXT / LOG / CSV",
            type=["txt", "log", "csv"],
            accept_multiple_files=True,
        )

        if st.button("Indexar corpus", type="primary"):
            try:
                if uploaded_files:
                    save_uploaded_files(uploaded_files)

                progress_bar = st.progress(0)
                status_text = st.empty()
                with st.spinner("Indexando documentos..."):
                    status = rebuild_index(progress_bar, status_text)
                st.success(status)
            except Exception as error:
                st.error(f"No se pudo indexar el corpus: {error}")

        try:
            indexed_chunks = get_collection().count()
            st.caption(f"Fragmentos indexados: {indexed_chunks}")
        except Exception as error:
            st.error(f"No se pudo consultar el indice: {error}")

    with chat_tab:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                elif message.get("error"):
                    st.error(message["error"])
                else:
                    render_results(message["query"], message["response"])

        prompt = st.chat_input("Escribe tu consulta...")
        if prompt:
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Buscando coincidencias..."):
                        response = build_response(prompt)
                    render_results(prompt, response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "query": prompt,
                            "response": response,
                        }
                    )
                except Exception as error:
                    error_message = f"No se pudo completar la busqueda: {error}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "error": error_message}
                    )


if __name__ == "__main__":
    main()
