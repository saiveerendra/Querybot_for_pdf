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

# ⚙️ Page configuration must come FIRST
st.set_page_config(page_title="Chat with PDF", layout="wide")

# ==========================
# Load API Keys
# ==========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)

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

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists("faiss_index"):
        st.error("❌ FAISS index not found. Please upload and process PDFs first.")
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
        st.markdown("### 🧠 Answer:")
        st.write(answer)
    else:
        st.warning("No response generated.")

# ==========================
# Streamlit App
# ==========================

def main():
    st.title("📄 Chat with PDF using Gemini + FAISS")

    with st.sidebar:
        st.header("📚 Upload & Process")
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

    user_question = st.text_input("Ask a question about the PDF content:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
