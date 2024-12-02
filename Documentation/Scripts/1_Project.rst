Guide de démarrage RAG
=====================

Introduction
-----------
Guide d'implémentation détaillé d'un système RAG (Retrieval Augmented Generation) avec PDF.

Installation
-----------

Prérequis
~~~~~~~~~

.. code-block:: bash

   pip install streamlit langchain chromadb pypdf

Architecture
-----------

Composants
~~~~~~~~~

1. **PDF Loader**: Extraction de texte
2. **Text Splitter**: Segmentation
3. **Vector Store**: Stockage et recherche
4. **LLM**: Génération de réponses

Implémentation
-------------

Extraction PDF
~~~~~~~~~~~~

.. code-block:: python

   from langchain.document_loaders import PyPDFLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   import tempfile

   def process_pdf(pdf_file):
       with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
           tmp_file.write(pdf_file.getvalue())
           pdf_path = tmp_file.name
       
       loader = PyPDFLoader(pdf_path)
       pages = loader.load()
       return pages

.. note::
   Le fichier temporaire permet de gérer les uploads PDF de manière sécurisée.

Segmentation
~~~~~~~~~~~

.. code-block:: python

   def split_text(pages):
       text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=512,
           chunk_overlap=128
       )
       chunks = text_splitter.split_documents(pages)
       return chunks

.. tip::
   - ``chunk_size``: Taille optimale pour le contexte
   - ``chunk_overlap``: Évite la perte d'information

Stockage Vectoriel
~~~~~~~~~~~~~~~~

.. code-block:: python

   from langchain.vectorstores import Chroma
   
   def create_vectorstore(chunks):
       vectorstore = Chroma.from_documents(
           documents=chunks,
           collection_name="pdf_collection"
       )
       return vectorstore

Traitement Questions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from langchain.prompts import PromptTemplate
   
   def get_response(question, vectorstore):
       # Recherche contextuelle
       docs = vectorstore.similarity_search(question, k=3)
       context = "\n".join([d.page_content for d in docs])
       
       # Template de prompt
       prompt = PromptTemplate.from_template("""
       Question: {question}
       Contexte: {context}
       
       Instructions:
       1. Utilisez uniquement le contexte fourni
       2. Si l'information n'est pas disponible, indiquez-le
       3. Répondez de manière concise et précise
       """)
       
       # Génération réponse
       response = llm(prompt.format(
           question=question,
           context=context
       ))
       return response

Interface Utilisateur
-------------------

Structure Streamlit
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import streamlit as st

   st.set_page_config(
       page_title="Assistant PDF RAG",
       layout="wide"
   )

   # Upload PDF
   pdf_file = st.file_uploader(
       "Chargez votre PDF",
       type="pdf"
   )

   # Zone questions
   question = st.text_input(
       "Posez votre question"
   )

   if question:
       response = get_response(question)
       st.write(response)

Bonnes Pratiques
---------------

Optimisation
~~~~~~~~~~
1. Calibrer ``chunk_size`` selon vos besoins
2. Ajuster ``k`` dans ``similarity_search``
3. Optimiser le prompt template

Gestion Erreurs
~~~~~~~~~~~~~

.. code-block:: python

   try:
       response = get_response(question)
   except Exception as e:
       st.error(f"Erreur: {str(e)}")

Ressources
---------
- Documentation LangChain: https://python.langchain.com
- Documentation Streamlit: https://docs.streamlit.io
- Guide Embeddings: https://www.pinecone.io/learn/dense-vector-embeddings

Exercices Pratiques
-----------------

Niveau Débutant
~~~~~~~~~~~~~
1. Ajouter compteur de tokens
2. Afficher temps de traitement

Niveau Avancé
~~~~~~~~~~~
1. Implémenter historique questions
2. Ajouter métadonnées segments
3. Optimiser recherche similitude
