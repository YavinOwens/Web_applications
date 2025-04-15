import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file, if present
load_dotenv()

# Securely load the OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

def get_vectorstore_from_url(url):
    """Fetches content from a URL, splits it into chunks, and creates a FAISS vectorstore."""
    # Load the document from the URL
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Create a vectorstore from the chunks using OpenAI embeddings
    vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

    return vector_store

def get_context_retriever_chain(vector_store):
    """Creates a history-aware retriever chain."""
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain):
    """Creates a conversational retrieval-augmented generation (RAG) chain.""" 
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    """Generates a response to the user's input using the conversational RAG chain."""
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# App configuration
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if not website_url:
    st.info("Please enter a website URL")
else:
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    if "vector_store" not in st.session_state:
        with st.spinner("Processing..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # User input handling
    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display the conversation
    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)
