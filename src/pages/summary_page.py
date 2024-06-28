import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import json

from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader

from utils.summary import make_database, TOC_proposition, classifier_plan, map_reduce_with_toc, classifier_RAG, maj_summary_noRAG, maj_summary_RAG, RAG_retour_utilisateur_retriever
from utils.API_setup import client_openAI_init
import fitz



# Function to reset the state of the application
def reset_state():
    for key in st.session_state.keys():
        del st.session_state[key]

def summary_page():

    if 'pages' not in st.session_state:
        st.warning("Please upload a PDF from the Home page.")
        return
    if 'toc' not in st.session_state:
        st.session_state.toc = TOC_proposition(st.session_state.pages[0].page_content)
    st.title("Edit Table of Contents")
    st.session_state.toc = st.text_area("Table of Contents (one item per line):", value=st.session_state.toc, height=200)
    st.write("Here is an example of ToC, feel free to change it and push the 'Generate Summary' button.")

    
    st.title("Summary")
    
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            st.session_state.summary = map_reduce_with_toc(st.session_state.pages, st.session_state.toc)
            st.success("Summary generation complete!")

    if st.session_state.summary:
        st.write("### Summary")
        st.write(st.session_state.summary)

        st.title("Provide Feedback")
        feedback = st.text_area("Your Feedback:", height=100)

        if st.button("Refine Summary"):
            with st.spinner("Refining summary..."):
                rep = classifier_RAG( feedback, my_model_name = "meta-llama/Meta-Llama-3-8B-Instruct")
                rep_dict = json.loads(rep)
        
                if rep_dict['information_demandée'] == 'False': 
                    st.session_state.summary = maj_summary_noRAG(feedback, st.session_state.summary)
                else : 
                    retrieve_docs = st.session_state.retriever.get_relevant_documents(rep_dict['information_demandée'])
                    st.session_state.summary = maj_summary_RAG(feedback, st.session_state.summary, retrieve_docs)
                
                st.success("Summary refinement complete!")
                st.write("### Refined Summary")
                st.write(st.session_state.summary)


