import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import tempfile

st.set_page_config(page_title="Closed-loop RAG", layout="centered")
st.title("📄 Closed-Loop RAG (Docs Q&A)")
st.markdown("Upload documents, ask questions. The AI will ONLY use your content for answers.")

def extract_text(uploaded_file):
    text = ""
    if uploaded_file.name.endswith('pdf'):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.name.endswith('docx'):
        doc = DocxDocument(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""
    metadatas = []
    file_names = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_names.append(file_name)
        text = extract_text(uploaded_file)
        all_text += "\n" + text
        metadatas.append({"file_name": file_name})

    st.success(f"✅ Uploaded {len(uploaded_files)} file(s).")
    st.write(f"Total length: {len(all_text):,} characters.")

    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 20)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([all_text], metadatas=metadatas)
    st.write(f"Indexed into {len(docs)} chunks.")

    embedder = st.radio("Embedding model", ["OpenAI", "all-MiniLM-L6-v2"])
    if embedder == "OpenAI":
        openai_api_key = st.text_input("Your OpenAI API key", type="password")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=None)

    llm_backend = st.radio("LLM backend", ["OpenAI GPT-3.5/4", "No LLM - just show chunks"])
    if llm_backend == "OpenAI GPT-3.5/4":
        openai_api_key = st.text_input("OpenAI Key for LLM", type="password")
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff"
        )
        user_question = st.text_area("Ask your question about the docs:")
        if st.button("Get Answer", disabled=not user_question):
            with st.spinner("Retrieving..."):
                resp = qa(user_question)
                st.subheader("Answer")
                st.write(resp["result"])
                st.subheader("Source Chunks Used")
                for i, doc in enumerate(resp["source_documents"], 1):
                    st.caption(f"Chunk {i}:")
                    st.code(doc.page_content[:700] + ("..." if len(doc.page_content) > 700 else ""), language="markdown")
    else:
        user_question = st.text_area("Ask your question about the docs:")
        if st.button("Get Chunks", disabled=not user_question):
            with st.spinner("Retrieving..."):
                matches = vectordb.similarity_search(user_question, k=5)
                st.subheader("Most Relevant Chunks")
                for i, doc in enumerate(matches, 1):
                    st.caption(f"Chunk {i}:")
                    st.code(doc.page_content[:700] + ("..." if len(doc.page_content) > 700 else ""), language="markdown")
