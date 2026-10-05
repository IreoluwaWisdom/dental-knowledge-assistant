import streamlit as st

from rag import load_embedding_resources

st.title("Dental Knowledge Assistant")

with st.spinner("Loading embedding model..."):
    model, chunk_embeddings = load_embedding_resources()

st.success("Embedding model loaded successfully.")
st.write(f"Number of chunk embeddings: {len(chunk_embeddings)}")