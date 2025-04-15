import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Ensure the environment variable is loaded, or set it directly

# API Key for OpenAI. Ensure this is secured and not exposed in production environments.
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = ""
def get_pdf_text(pdf_docs):
    """ Extracts text from uploaded PDF files. """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:  # Check for None to avoid adding 'None' to text
                text += extracted_text
    return text
@st.cache_data
def get_text_chunks(text):
    """ Splits text into manageable chunks for processing. """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)
@st.cache_data
def get_vectorstore(text_chunks):
    """ Generates a vector store from text chunks using embeddings. """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Explicitly pass the API key here
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
@st.cache_data
def get_conversation_chain(_vectorstore):
    """ Creates a conversational retrieval chain. """
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)  # If needed, also pass the API key here
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory
    )

def handle_user_input(user_question):
    """ Handles user input and updates the UI with responses. """
    if not st.session_state.conversation:
        st.error("Please process your documents first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.markdown(css, unsafe_allow_html=True)

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True, type=['pdf'])
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
