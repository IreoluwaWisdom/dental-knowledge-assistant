import streamlit as st

from rag import (
    get_chunk_embeddings,
    retrieve,
    build_context,
    get_sources,
    call_llm,
    similarity_threshold
)


st.title("Dental Knowledge Assistant")


@st.cache_resource
def load_embeddings():
    return get_chunk_embeddings()


chunk_embeddings = load_embeddings()


question = st.text_area("Type your dental question here")

button_clicked = st.button("Ask")


if button_clicked:

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching the knowledge base..."):

            top_results = retrieve(
                question,
                chunk_embeddings
            )

            filtered_results = [
                result
                for result in top_results
                if result["score"] >= similarity_threshold
            ]

            if not filtered_results:
                st.warning(
                    "No sufficiently relevant information was found "
                    "in the hospital knowledge base."
                )

            else:
                context = build_context(filtered_results)

                answer = call_llm(
                    question,
                    context
                )

                sources = get_sources(filtered_results)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources")

                for source in sorted(sources):
                    st.write(f"- {source}")