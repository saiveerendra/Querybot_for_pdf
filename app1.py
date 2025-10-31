import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import boto3
import os

# Load environment variables
load_dotenv()

# Initialize AWS Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # change if your region differs
    aws_access_key_id="AKIA4AQ3T6K63DMN7UGG",
    aws_secret_access_key="AKIA4AQ3T6K6TKFLQTMT"
)
# AKIA4AQ3T6K6TKFLQTMT
# AKIA4AQ3T6K63DMN7UGG

# Translation API URL (same as before)
TRANSLATION_URL = "https://deep-translate1.p.rapidapi.com/language/translate/v2"
HEADERS = {
    "content-type": "application/json",
    "X-RapidAPI-Key": "2761bb9559msh64aaf6be6191b4cp191e83jsn51da63e6da75",
    "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
}

# --- PDF Processing ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=500)
    return text_splitter.split_text(text)

# --- Vector Store Creation using Titan Embeddings ---
def get_vector_store(text_chunks):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# --- QA Chain using Titan LLM ---
def get_conversational_chain(vector_store):
    prompt_template = """
    You are a helpful assistant. Use the given context to answer the user's question.
    If the answer is not in the context, say "Answer not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock_client)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- Translation ---
def translate_text(text, source_lang="en", target_lang="te"):
    payload = {"q": text, "source": source_lang, "target": target_lang}
    response = requests.post(TRANSLATION_URL, json=payload, headers=HEADERS)

    if response.status_code == 200:
        return response.json()["data"]["translations"]["translatedText"]
    else:
        return f"Translation failed: {response.text}"

# --- User Input and Query ---
def user_input(user_question):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    qa_chain = get_conversational_chain(new_db)
    response = qa_chain.run(user_question)

    translated_text = translate_text(response)
    st.write(translated_text)

# --- Streamlit UI ---
def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF (Powered by AWS Titan)")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Submit & Process'", accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete ✅")

if __name__ == "__main__":
    main()
