import streamlit as st
from rag_pipeline.qa_generator import answer_question

st.title("🌍 ClimaFact: Ask About Climate Policy")
st.markdown("Ask a question about climate policy, environmental risk, or sustainability!")

query = st.text_input("Your question:", placeholder = "e.g. What is NYC's plan for sea-level rise?")

if query:
    with st.spinner("Searching and generating response..."):
        response = answer_question(query)
        st.markdown("### Answer")
        #Print the answer
        st.write(response["answer"])
        st.markdown("### Sources")

        #Print the sources
        for source in response["sources"]:
            st.markdown(f"- {source}")