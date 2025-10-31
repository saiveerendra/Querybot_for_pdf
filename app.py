import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import requests
import google.generativeai as genai

# -----------------------------------------------------------------------------
# Load API Keys
# -----------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
# st.write(api_key)

# Translation API (optional)
TRANSLATION_URL = "https://deep-translate1.p.rapidapi.com/language/translate/v2"
HEADERS = {
    "content-type": "application/json",
    "X-RapidAPI-Key": "2761bb9559msh64aaf6be6191b4cp191e83jsn51da63e6da75",
    "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
}

# -----------------------------------------------------------------------------
# PDF Text Extraction
# -----------------------------------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------------------------------------------------------
# Split Text into Chunks
# -----------------------------------------------------------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# -----------------------------------------------------------------------------
# Create and Save FAISS Vector Store
# -----------------------------------------------------------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# -----------------------------------------------------------------------------
# Load Conversational QA Chain
# -----------------------------------------------------------------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context.
    If the answer is not in the context, say: "Answer is not available in the context."
    Do not make up information.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# -----------------------------------------------------------------------------
# Translation Helper
# -----------------------------------------------------------------------------
def translate_text(text, source_lang="en", target_lang="te"):
    payload = {"q": text, "source": source_lang, "target": target_lang}
    response = requests.post(TRANSLATION_URL, json=payload, headers=HEADERS)
    if response.status_code == 200:
        return response.json()["data"]["translations"]["translatedText"]
    else:
        return f"Translation failed: {response.text}"

# -----------------------------------------------------------------------------
# Handle User Query
# -----------------------------------------------------------------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    if "output_text" in response:
        translated_text = translate_text(response["output_text"])
        st.write(translated_text)
    else:
        st.error("No response generated.")

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config("Chat with PDF", page_icon="📘")
    st.header("📘 Chat with your PDF (Gemini-Pro powered RAG)")

    user_question = st.text_input("Ask a question from your uploaded PDF files:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("⚙️ Menu")
        st.write("💡 Using **Google Gemini-Pro** for intelligent context-aware answers.")
        pdf_docs = st.file_uploader("Upload PDF files and click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete! You can now ask questions.")

if __name__ == "__main__":
    main()
