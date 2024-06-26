from openai import OpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser



from API_setup import set_model_endpoint, client_openAI_init


def TOC_proposition( text , my_model_name = "gradientai/Llama-3-8B-Instruct-262k"):


    prompt_systeme = """Tu es un agent de l'administration française qui fait des synthèses de textes. Tu sais en particulier faire des plans de synthèses. Tu parles en français."""


    prompt_user_template = f"""Tu dois me faire un plan d'une synthèse du texte suivant : \n 
                        ```{text}``` \n 
                        Réponds en français.
                        Je veux un plan en 2, 3 ou 4 parties. Donne moi les titres de chaque chapitre.
                        REPONDS EN FRANCAIS
                        """


    chat_messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": prompt_user_template},
    ]


    # Mistral models don't have any system role messages
    if my_model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        chat_messages.pop(0)

    client = client_openAI_init(my_model_name)

    # stream chat.completions
    chat_response = client.chat.completions.create(
        model=my_model_name, # this must be the model name the was deployed to the API server
    #    stream=True,
        max_tokens=3000,
        top_p=0.9,
        temperature=0.2,
        messages=chat_messages
    )
    output = chat_response.choices[0].message.content
    print(output)
    return output


def map_reduce_with_toc(list_doc, plan, my_model_name = "meta-llama/Meta-Llama-3-8B-Instruct",chunk_size=2000,chunk_overlap=100):



    model_name, openai_api_base, openai_api_key= set_model_endpoint(my_model_name)
    print(f"Model: {model_name}, Endpoint: {openai_api_base}, api_key: {openai_api_key}")

    llm = ChatOpenAI(
            openai_api_base=openai_api_base,
            api_key=openai_api_key,
            model=my_model_name,
            temperature=0.5,
        )


    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    text_docs_splitted = text_splitter.split_documents(list_doc)



    map_prompt_template = """
                            <|begin_of_text|><|start_header_id|> system <|end_header_id|>  Tu es un agent de l'administration qui fait des résumés de textes en français. 
                            <|eot_id|>
                            <|start_header_id|>user<|end_header_id|>
                            \n Tu dois résumer le texte suivant: 
                            '''{text}'''
                            \n Ne garde que les éléments importants et pertinents du passage. Concerver les idées générales et les conculsions.
                        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                        """


    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])



    combine_prompt_template = """
                        <|begin_of_text|><|start_header_id|> system <|end_header_id|> Tu es un agent de l'administration qui fait des synthèses de textes. 
                        <|eot_id|>
                        <|start_header_id|>user<|end_header_id|>

                        Voici plusieurs textes qui sont des résumés d'un seul document. Tu dois synthéstiser ces textes pour obtenir une seule synthèse du document initial.
                        Voici les textes : \n 
                        ```{text}``` \n 
                        Sois complet et réponds suis bien le plan qui t'as été donné. 
                        Structurer ta synthèse. Utilise des connecteurs logiques.
                        Ta synthèse dois suivre le plan suivant:
                        """ + plan + """
                        
                        
                        <|eot_id|>
                        <|start_header_id|>assistant<|end_header_id|>
                        """

    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        #return_intermediate_steps=True,
    )

    map_reduce_outputs = map_reduce_chain(text_docs_splitted)

    return map_reduce_outputs

def maj_summary_RAG(retour_utilisateur, resume, retrieved_doc, my_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"): 
    
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
                        Je vais te donner un texte et je souhaite que tu le MODIFIE. Tu ne dois pas le remplacer mais uniquement rajouter des choses en respectant une demande que je te donnerais après.
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

def maj_summary_noRAG(retour_utilisateur, resume, my_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"): 
    
    model_name, openai_api_base, openai_api_key= set_model_endpoint(my_model_name)
    print(f"Model: {model_name}, Endpoint: {openai_api_base}, api_key: {openai_api_key}")


    llm = ChatOpenAI(
            openai_api_base=openai_api_base,
            api_key=openai_api_key,
            model=my_model_name,
            temperature=0.5,
        )
    

        
    harmonisation_prompt_template = """
                        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
                        Tu es un agent de l'administration qui rédige des textes dans un français parfait et avec une grande maîtrise de la rédaction.
                        <|eot_id|>\n\n
                        <|start_header_id|>user<|end_header_id|>
                        \n Voici un texte, C'est un résumé d'un texte je souhaite que tu repasses dessus pour que je puisse le publier en respectant un retour. Voici le texte de base :
                        '''{resume}'''
                        \n 
                        Voici un retour que j'ai sur ce texte : 
                        '''{retour_utilisateur}'''. \n 
                        Tu dois donc le modifier pour qu'il corresponde à ce retour. Le texte soumis dois être conforme au texte de base mais respecter la modification demandée.
                        Le texte dois garder la même structure.
                            
                        <|eot_id|>
                        \n\n<|start_header_id|>assistant<|end_header_id|>
                        """
    harmonisation_prompt = PromptTemplate(template=harmonisation_prompt_template, input_variables=["resume","retour_utilisateur", "concatenated_retrieved_doc"])

    output_parser = StrOutputParser()

    chain = harmonisation_prompt | llm | output_parser

    output = chain.invoke({"resume": resume, "retour_utilisateur":retour_utilisateur})

    return output
    

def classifier_RAG( retour_utilisateur , my_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"):


    prompt_systeme = """Tu es un agent intermédiaire qui doit classifier les besoins d'un utilisateur qui demande des modifications sur un texte généré. """

    prompt_user_template = f"""Un utilisateur fais des retours sur un résumé de texte qui a été généré. 
                            Je vais te donner son retour et tu vas me dire si l'utilisateur à besoin 
                            d'informations supplémentaires dans son résumé. Voici le retour de l'utilisateur : \n 
                            ```{retour_utilisateur}``` \n 

                            Dis si l'utilisateur à besoin de rajouter des informations dans le résumé qui a été généré. 
                                
                            Si oui, dis moi les informations clairement dont l'utilisateur à besoin ET UNIQUEMENT CELA. 
                            Ne me Repond QUE L'INFORMATION DEMANDEE. Répond sous forme d'un json avec comme clé : 'information_demandée'. 
                            Si le retour de l'utilisateur ne demande pas d'information supplémentaire mais fait un retour sur le style, la taille, des informations à enlever, des informations à réduire ou tout autre demande, REPOND 'False'
                            Je ne veux qu'un json en sortie, exemple : 
                            Si besoin d'information : 
                            
                            {{"information_demandée": "Ici mettre les informations demandées"}}```
                            
                            Si pas besoin d'information : 
                            
                            ```{{"information_demandée": "False"}}```

                            
                            """

    chat_messages = [
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_user_template},
        ]


        # Mistral models don't have any system role messages
    if my_model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            chat_messages.pop(0)

    client = client_openAI_init(my_model_name)

        # stream chat.completions
    chat_response = client.chat.completions.create(
            model=my_model_name, # this must be the model name the was deployed to the API server
        #    stream=True,
            max_tokens=1000,
            top_p=0.9,
            temperature=0.2,
            messages=chat_messages
        )
    output = chat_response.choices[0].message.content
    print(output)
    return output

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