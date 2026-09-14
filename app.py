#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon=" 📚",
    layout="centered"
)

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF and ask questions about it.")


# ============================================================
# GEMINI API KEY
# ============================================================

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your Gemini API key in the sidebar.")
    st.stop()


client = genai.Client(
    api_key=api_key
)

MODEL = "gemini-3.5-flash"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedder = load_embedding_model()


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(text, chunk_size=800):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# PROCESS PDF
# PDF → TEXT → CHUNKS → EMBEDDINGS → FAISS
# ============================================================

def process_pdf(uploaded_file):

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    # Use first 3 pages
    for page in reader.pages[:3]:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Create chunks
    chunks = create_chunks(
        text,
        chunk_size=800
    )

    if len(chunks) == 0:
        return None, None

    # Create embeddings
    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    # Add embeddings to FAISS
    index.add(embeddings)

    return chunks, index


# ============================================================
# RETRIEVAL
# QUESTION → EMBEDDING → FAISS → TOP 3 CHUNKS
# ============================================================

def retrieve(
    query,
    chunks,
    index,
    k=3
):

    # Convert question into embedding
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Don't request more chunks than available
    k = min(k, len(chunks))

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for i in indices[0]:

        results.append(
            chunks[i]
        )

    return results


# ============================================================
# RAG
# RETRIEVAL + AUGMENTATION + GENERATION
# ============================================================

def basic_rag(
    question,
    chunks,
    index
):

    # -------------------------
    # RETRIEVAL
    # -------------------------

    relevant_chunks = retrieve(
        question,
        chunks,
        index,
        k=3
    )

    # -------------------------
    # AUGMENTATION
    # -------------------------

    context = "\n\n---\n\n".join(
        relevant_chunks
    )

    prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY
the information provided in the document
context below.

If the answer is not available in the
context, say:

"I don't know from this document."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

    # -------------------------
    # GENERATION
    # -------------------------

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text, relevant_chunks


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF BUTTON
# ============================================================

if uploaded_file:

    if st.button("Process PDF"):

        with st.spinner(
            "Reading PDF and creating embeddings..."
        ):

            chunks, index = process_pdf(
                uploaded_file
            )

        if chunks is None:

            st.error(
                "Could not extract text from the PDF."
            )

        else:

            # Save RAG objects in session state
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

            st.success(
                f"PDF processed successfully! "
                f"{len(chunks)} chunks created."
            )


# ============================================================
# QUESTION SECTION
# ============================================================

if "chunks" in st.session_state:

    st.divider()

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document..."
            ):

                answer, sources = basic_rag(
                    question,
                    st.session_state["chunks"],
                    st.session_state["index"]
                )

            # -------------------------
            # DISPLAY ANSWER
            # -------------------------

            st.subheader("Answer")

            st.write(answer)

            # -------------------------
            # DISPLAY RETRIEVED CHUNKS
            # -------------------------

            with st.expander(
                "View retrieved chunks"
            ):

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    st.markdown(
                        f"**Retrieved Chunk {i}**"
                    )

                    st.write(source)

                    st.divider()

