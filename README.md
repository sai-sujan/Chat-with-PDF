# 📚 Chat with PDF using Gemini + RAG

An intelligent app that lets you **chat with PDF documents** using **Google Gemini** and **LangChain's RAG (Retrieval-Augmented Generation)** approach.  
Upload one or more PDFs, ask natural language questions, and get accurate, context-aware answers.

---

## ✨ Features

- 📁 Upload multiple PDFs and process them into vector embeddings
- 🔍 Perform semantic search using **FAISS**
- 🧠 Answer questions using **Gemini's `ChatGoogleGenerativeAI`**
- 📄 View the **retrieved chunks** that contributed to each answer
- 💬 RAG pipeline with custom prompt + Gemini reasoning

---

## 🖼️ App Preview

<div align="center">
  <img width="1710" alt="Screenshot 2025-06-01 at 3 58 46 PM" src="https://github.com/user-attachments/assets/4862a0e9-d51a-40ab-992a-5970a01551d6" />

  
</div>

---

## 🧠 How It Works

1. **PDF Upload:** User uploads one or more PDF files
2. **Text Extraction:** Pages are read using `PyPDF2` and merged into a single text stream
3. **Chunking:** Text is split into overlapping chunks using `RecursiveCharacterTextSplitter`
4. **Embedding & Vector Store:** Chunks are embedded using `GoogleGenerativeAIEmbeddings` and saved to a FAISS index
5. **User Query:** A question is asked, and top matching chunks are retrieved from FAISS
6. **Answering:** A Gemini model responds using a custom prompt and retrieved context

---

## 🛠 Tech Stack

- 🧠 `LangChain` (`PromptTemplate`, `FAISS`, `load_qa_chain`)
- 🤖 `GoogleGenerativeAIEmbeddings`, `ChatGoogleGenerativeAI` from `langchain_google_genai`
- 📚 `PyPDF2` for PDF text extraction
- 🧱 `FAISS` for fast semantic search
- 🌐 `Streamlit` for frontend
- 🔐 `.env` for API key handling

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/sai-sujan/Chat-with-PDF.git
cd chat-with-pdf

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Google API key
echo "GOOGLE_API_KEY=your-gemini-api-key" > .env

# 4. Run the app
streamlit run app.py
