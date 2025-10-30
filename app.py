import streamlit as st
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

st.set_page_config(page_title="Closed-Loop RAG", layout="centered")

st.title("📄 Closed-Loop Retrieval-Augmented Generation (RAG) App")
st.markdown(
    "Upload documents and get AI answers strictly based on those documents. No external search."
)

# Text extraction function
def extract_text(uploaded_file):
    text = ""
    if uploaded_file.name.endswith("pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.name.endswith("docx"):
        doc = DocxDocument(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:  # TXT fallback
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text


uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_text = ""
    metadatas = []
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        all_text += "\n" + text
        metadatas.append({"source": uploaded_file.name})

    st.success(f"✅ Uploaded {len(uploaded_files)} document(s) with {len(all_text):,} characters.")

    # Chunking params
    chunk_size = st.slider("Chunk size (characters)", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=500, value=200, step=50)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([all_text], metadatas=metadatas)
    st.write(f"Split into {len(docs)} chunks.")

    # Embedding Model selection
    embedder = st.radio("Embedding Model:", options=["OpenAI", "all-MiniLM-L6-v2"])

    if embedder == "OpenAI":
        openai_embedding_key = st.text_input("OpenAI API Key for Embeddings:", type="password")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_embedding_key) if openai_embedding_key else None
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if embeddings is not None:
        # Use unique collection name each run to avoid conflicts
        collection_name = f"collection-{uuid.uuid4()}"
        vectordb = Chroma.from_documents(docs, embedding=embeddings,collection_name=collection_name)


        # LLM Backend choice
        llm_backend = st.radio("LLM Backend:", options=["OpenAI GPT-3.5/4", "Show Relevant Chunks Only"])

        if llm_backend == "OpenAI GPT-3.5/4":
            openai_llm_key = st.text_input("OpenAI API Key for LLM:", type="password")
            if openai_llm_key:
                llm = OpenAI(openai_api_key=openai_llm_key, temperature=0)
                retriever = vectordb.as_retriever(search_kwargs={"k": 5})
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type="stuff",
                )

                question = st.text_area("Ask your question about the uploaded documents:")
                if st.button("Get Answer", disabled=not question):
                    with st.spinner("Retrieving answer..."):
                        result = qa.run(question)
                        st.subheader("Answer")
                        st.write(result)

            else:
                st.warning("Enter OpenAI API key for LLM to generate answer.")

        else:  # Show chunks only
            question = st.text_area("Ask your question to retrieve chunks:")
            if st.button("Get Relevant Chunks", disabled=not question):
                with st.spinner("Searching relevant chunks..."):
                    matches = vectordb.similarity_search(query=question, k=5)
                    st.subheader("Relevant Document Chunks")
                    for i, doc in enumerate(matches, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content[:700] + ("..." if len(doc.page_content) > 700 else ""))
    else:
        st.warning("Please provide the required API key for embedding or select an embedding model.")
