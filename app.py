import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests

# ==========================
# Load API Keys
# ==========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
rapid_api_key = os.getenv("RAPID_API_KEY")
genai.configure(api_key=api_key)

# Translation API Config
TRANSLATION_URL = "https://deep-translate1.p.rapidapi.com/language/translate/v2"
HEADERS = {
    "content-type": "application/json",
    "X-RapidAPI-Key": rapid_api_key,
    "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
}

# ==========================
# Helper Functions
# ==========================

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Avoid NoneType errors
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "answer is not available in the context".
    Do not provide a wrong answer.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def translate_text(text, source_lang="en", target_lang="te"):
    payload = {"q": text, "source": source_lang, "target": target_lang}
    response = requests.post(TRANSLATION_URL, json=payload, headers=HEADERS)
    if response.status_code == 200:
        return response.json()["data"]["translations"]["translatedText"]
    else:
        return f"Translation failed: {response.text}"

def user_input(user_question, target_lang):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists("faiss_index"):
        st.error("FAISS index not found. Please upload and process PDFs first.")
        return

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    answer = response.get("output_text") or response.get("result", "")
    if answer:
        translated_text = translate_text(answer, target_lang=target_lang)
        st.write("### 🧠 Answer:")
        st.write(translated_text)
    else:
        st.error("No response generated.")

# ==========================
# Streamlit App
# ==========================

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("📄 Chat with PDF using Gemini + FAISS + Translation")

    user_question = st.text_input("Ask a question about the PDF content:")

    target_lang = st.selectbox("Select translation language:", ["te", "hi", "ta", "ml", "en"], index=0, help="Language to translate the answer into")

    if user_question:
        user_input(user_question, target_lang)

    with st.sidebar:
        st.title("📚 Upload & Process")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDFs processed successfully!")

if __name__ == "__main__":
    main()
