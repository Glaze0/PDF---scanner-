# Simple RAG AI app

##Creating a Streamlit based RAG AI app using Python

-- Using PDF as document, through pypdf library of Python.
-- Used Straeamlit library of Python for web.
-- using all-MiniLM-L6-v2 model of SentenceTransformer for embedding of chunks.
-- using FAISS library for indexing of chunks
-- using genai from google with model gemini-3.5-flash for generating the answer.

=======================================================================================

## Deployment Steps

-- Do it on terminal: 
`pip install streamlit pypdf faiss-cpu sentence-transformers google-genai numpy`

-- From your terminal: 
navigate to the path where this file is stored, and copy the path. on terminal go to that path using 'cd'
run: `streamlit run app.py`
