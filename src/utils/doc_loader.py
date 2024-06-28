from openai import OpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st



def extract_text_from_pdf(uploaded_file):
    
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue()) 
        
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    texts_doc = text_splitter.split_documents(pages)    
    db = make_database(texts_doc)
    retriever = RAG_retour_utilisateur_retriever(db)
    st.markdown(" db created ")
    return pages, texts_doc, retriever

def make_database(doc_texte):
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",],
    )
    chunks = text_splitter.split_documents(doc_texte)

    lc_embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    db = FAISS.from_documents(chunks, lc_embed_model)
    return db

    
def RAG_retour_utilisateur_retriever(db, score_thresh =  0.1,  tok_k = 6):

    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_thresh,"k": tok_k})
    return retriever