import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()
huggingface_token = os.getenv("Hugging_face")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_text = ""

    if request.method == "POST":
        pdf_file = request.files["pdf"]
        user_query = request.form["query"]

        if pdf_file:
            # Read PDF
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    full_text += content

            # Split text
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_text(full_text)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(texts, embedding=embeddings)

            # Load Model
            model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=huggingface_token)
            hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            llm = HuggingFacePipeline(pipeline=hf_pipeline)

            # Run QA
            docs = vector_store.similarity_search(user_query)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=user_query)

    return render_template("index.html", answer=answer, pdf_text=full_text)


if __name__ == "__main__":
    app.run(debug=True)
