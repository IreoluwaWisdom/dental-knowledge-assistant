import streamlit as st

from rag import (
    retrieve,
    build_context,
    call_llm,
    similarity_threshold,
    get_sources
)

st.title("Dental Knowledge Assistant")
st.success("RAG import successful.")
st.write("If you can see this, importing rag.py did not crash the service.")