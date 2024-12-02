import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
import tempfile
from typing import Iterator

# Configuration de la page
st.set_page_config(
    page_title="Assistant PDF Intelligent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design plus professionnel
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e3d59;
        padding-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #1e3d59;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def initialize_models():
    return {
        'llm': Ollama(model="mistral", base_url="http://127.0.0.1:11434"),
        'embeddings': OllamaEmbeddings(model="mistral", base_url="http://127.0.0.1:11434")
    }

models = initialize_models()

# En-tête de l'application
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("📚 Assistant PDF Intelligent")
    st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Votre assistant personnel pour l'analyse et la compréhension de documents
        </div>
    """, unsafe_allow_html=True)

def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    pdf_chunks = text_splitter.split_documents(pages)
    
    vector_store = Chroma.from_documents(
        documents=pdf_chunks,
        embedding=models['embeddings'],
        persist_directory="./pdf_chroma_db"
    )
    vector_store.persist()
    
    os.unlink(pdf_path)
    return vector_store

# Zone de téléchargement améliorée
with st.container():
    st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h3 style='color: #1e3d59; margin-bottom: 1rem;'>📎 Importation de document</h3>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf")
    if uploaded_file is not None and st.session_state.vector_store is None:
        with st.spinner("⏳ Analyse du document en cours..."):
            st.session_state.vector_store = process_pdf(uploaded_file)
        st.success("✅ Document analysé avec succès !")

def get_response(question) -> Iterator[str]:
    try:
        if st.session_state.vector_store is not None:
            retriever = st.session_state.vector_store.as_retriever()
            
            prompt = PromptTemplate.from_template("""Répondez à la question suivante en vous basant sur le contexte fourni. 
            Si la question ne peut pas être répondue à partir du contexte, donnez une réponse générale basée sur vos connaissances.
            
            Contexte: {context}
            Question: {input}
            
            Réponse: """)
            
            combine_docs_chain = create_stuff_documents_chain(
                llm=models['llm'],
                prompt=prompt
            )
            
            retrieval_chain = create_retrieval_chain(
                retriever,
                combine_docs_chain
            )
            
            response = retrieval_chain.invoke({
                "input": question
            })
            
            words = response['answer'].split()
            for word in words:
                yield word + " "
        else:
            response = models['llm'].invoke(question)
            words = response.split()
            for word in words:
                yield word + " "

    except Exception as e:
        yield f"⚠️ Une erreur s'est produite : {str(e)}"

# Zone de chat améliorée
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
st.text_input("💭 Posez votre question", key="user_input", on_change=lambda: handle_user_input())

def handle_user_input():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in get_response(user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.user_input = ""

# Historique des messages stylisé
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='chat-message user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='chat-message assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

# Barre latérale améliorée
with st.sidebar:
    st.markdown("""
    <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
    <h3 style='color: #1e3d59; margin-bottom: 1rem;'>📖 Guide d'utilisation</h3>
    
    <p style='color: #666; margin-bottom: 1rem;'><strong>1. Import du document</strong><br>
    Téléchargez votre fichier PDF en utilisant le bouton d'import.</p>
    
    <p style='color: #666; margin-bottom: 1rem;'><strong>2. Analyse du contenu</strong><br>
    Attendez que le système analyse votre document.</p>
    
    <p style='color: #666; margin-bottom: 1rem;'><strong>3. Posez vos questions</strong><br>
    Interrogez l'assistant sur le contenu du document ou sur d'autres sujets.</p>
    
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 2rem;'>
    <h4 style='color: #1e3d59; margin-bottom: 0.5rem;'>Fonctionnalités</h4>
    <ul style='color: #666;'>
    <li>Analyse de documents PDF</li>
    <li>Questions-réponses intelligentes</li>
    <li>Base de connaissances générale</li>
    <li>Interface conversationnelle</li>
    </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)