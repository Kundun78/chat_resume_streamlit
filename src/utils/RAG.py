from openai import OpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from utils.API_setup import set_model_endpoint, client_openAI_init



def RAG_response(retour_utilisateur, retrieved_doc, my_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"): 
    
    model_name, openai_api_base, openai_api_key= set_model_endpoint(my_model_name)
    print(f"Model: {model_name}, Endpoint: {openai_api_base}, api_key: {openai_api_key}")


    llm = ChatOpenAI(
            openai_api_base=openai_api_base,
            api_key=openai_api_key,
            model=my_model_name,
            temperature=0.5,
        )
    
    concatenated_retrieved_doc = " \n ".join([doc.page_content for doc in retrieved_doc])


        
    harmonisation_prompt_template = """
                        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
                        Tu es un agent de l'administration qui rédige des textes dans un français parfait et avec une grande maîtrise de la rédaction.
                        <|eot_id|>\n\n
                        <|start_header_id|>user<|end_header_id|>
                        Je vais te donner une question et un contexte. Tu devras t'appuyer sur le contexte si possible pour répondre à la question.
                        \n Voici un texte :
                        '''{resume}'''
                        \n 
                        
                        Voici la demande que j'ai sur ce texte : 
                        '''{retour_utilisateur}'''. \n 
                        Tu dois donc le modifier pour qu'il corresponde à cette. Pour ceci, voici des éléments que tu peux utiliser :
                        
                        '''{concatenated_retrieved_doc}''' \n 
                        Tu dois donc le modifier pour qu'il corresponde à ce retour. Le texte soumis dois être conforme au texte de base mais respecter la modification demandée.
                        Le texte dois garder la même structure.  REPOND EN FRANCAIS;
    
                            
                        <|eot_id|>
                        \n\n<|start_header_id|>assistant<|end_header_id|>
                        """
    harmonisation_prompt = PromptTemplate(template=harmonisation_prompt_template, input_variables=["resume","retour_utilisateur", "concatenated_retrieved_doc"])

    output_parser = StrOutputParser()

    chain = harmonisation_prompt | llm | output_parser

    output = chain.invoke({"resume": resume, "retour_utilisateur":retour_utilisateur, "concatenated_retrieved_doc":concatenated_retrieved_doc})

    return output