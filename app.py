# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from modules.rag import get_pdf_text, get_text_chunks, save_vector_store, user_query_response

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="📚 Chat with PDF", layout="wide")
st.header("Chat with PDF using Gemini 💬")

user_question = st.text_input("Ask a question based on your uploaded PDFs:")

if user_question:
    with st.spinner("Thinking..."):
        reply, retrieved_chunks = user_query_response(user_question)

        # 🧠 Display the answer
        st.subheader("💬 Answer")
        st.write(reply)

        # 📚 Show the retrieved chunks
        with st.expander("🔍 Retrieved Chunks Used in Answer"):
            for i, doc in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i + 1}:**")
                st.info(doc.page_content)


with st.sidebar:
    st.subheader("📁 Upload PDF")
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
    if st.button("📄 Submit & Process"):
        with st.spinner("⏳ Processing PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(raw_text)
            save_vector_store(chunks)
            st.success("✅ PDF processed and indexed!")
