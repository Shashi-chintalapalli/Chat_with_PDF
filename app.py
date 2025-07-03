import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline

# Load .env
load_dotenv()
huggingface_token = os.getenv("Hugging_face")

# Streamlit config
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")

# Custom CSS for better visuals
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        color: #fff;
        background-color: #0e1117;
    }
    .pdf-preview {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 10px;
        max-height: 400px;
        overflow-y: scroll;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("📄 Chat with your PDF")
st.write("Upload a PDF, ask questions, and get smart answers!")

# File upload
pdf = st.file_uploader("📤 Upload your PDF here", type="pdf", key="pdf_uploader")

if pdf:
    # Load and read PDF
    pdf_reader = PdfReader(pdf)
    full_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    # Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(full_text)

    # Embedding & Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embedding=embeddings)

    # Load model
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=huggingface_token)
    hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # User query
    query = st.text_input("📝 Ask something about the PDF:")

    if query:
        docs = vector_store.similarity_search(query)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        st.subheader("💬 Answer:")
        st.write(response)

        # Show preview on button click
        if st.button("📄 Show PDF Preview"):
            st.markdown('<div class="pdf-preview">{}</div>'.format(full_text.replace('\n', '<br>')), unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Made with ❤️ by Shashi Kumar Reddy • 📧 shashi.chintalapalli@gmail.com
    </div>
""", unsafe_allow_html=True)
