import streamlit as st
from rag import (retrieve, build_context, call_llm, similarity_threshold, get_sources)

st.title("How may I help you?")
question = st.text_area("Type your question here")
button_clicked = st.button("Ask")

if button_clicked: 
    top_results = retrieve(question)
    best_score = top_results.values[0].item()

    sources = get_sources(top_results)
    if best_score < similarity_threshold:
        st.subheader(f"QUESTION: {question}\n")
        st.write("RETRIEVAL STATUS:")
        st.write("No sufficiently relevant context found\n")

    else:
        context = build_context(top_results)
        answer = call_llm(question, context)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for source in sources:
            st.write(source)