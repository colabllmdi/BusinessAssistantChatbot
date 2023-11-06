import openai
import langchain
import os
import json
from pprint import pprint
import pinecone

import time
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from typing import Optional
import pandas as pd
import ast
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
# from langchain.prompts import chat_prompt
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS


def formatted_output(output,master_llm,conversational_memory):
    answer = output
    final_prompt_inst = "You're a Output Formatter. Your job is to display the answer in a well structured format. Answer should be in proper grammar. Don't give repetitive answers. Strictly ensure answer length should be summarized in way that it's not more than 1500 letter including special characters. You are proficient in English. Do not mention that you're giving a well-structured format, make it seem natural. : {answer}"
    prompt_template_final = PromptTemplate(template = final_prompt_inst,input_variables=['answer'] )

    chain = LLMChain(llm=master_llm, prompt=prompt_template_final,memory = conversational_memory)

    final_text = chain({'answer':answer})['text']
    
    return final_text

def intent_recognition(UserPrompt,master_llm):
    examples = [
    {"input":"What is the difference between step X and step y for a opening a XYZ?","output":"Compare"},
    {"input": "Compare differences between the following services", "output": "Compare"},
    {"input": "Differentiate the following services on the basis of price", "output": "Compare"},
    {"input": "Which of the follwing is better", "output": "Compare"},
    {"input": "Explain the difference between pricing of service A and B", "output": "Compare"},
    {"input": "Why should I choose service A over service B", "output": "Compare"},
    {"input": "Weigh the following services on their requirements", "output": "Compare"},
    {"input": "Service A vs Service B", "output": "Compare"},
    {"input":"Describe requirements for the following service", "output": "Describe"},
    {"input":"Tell me about A", "output": "Describe"},
    {"input":"Tell me about wholesale of software and retail sale of software", "output": "Describe"},
    {"input":"How can I appply for the following service", "output": "Describe"},
    {"input":"What are the conditions for A", "output": "Describe"},
    {"input":"How much is the price to apply for the following service", "output": "Describe"},
    {"input":"Explain the requirements of service A", "output": "Describe"},
    {"input":"Define the important characteristics of the following service", "output": "Describe"},
    {"input":"difference between the following services", "output": "Compare"},
    {"input":"How to do service A", "output": "Describe"},
    {"input":"How to enquire service XYZVS","output":"Describe"},
    {"input":"How to access the service XYZVS","output":"Describe"},
    {"input":"How to get information on service XYZVS","output":"Describe"},
    {"input":"How to find about service XYZVS","output":"Describe"},
    {"input":"Provide info on something","output":"Describe"},
    {"input":"Can you send this to me?","output":"Transfer"},
    {"input":"Send it to my number?","output":"Transfer"},
    {"input":"This is my number XXXXXXXXX Send it to me?","output":"Transfer"},
    {"input":"Transfer it to my nummber","output":"Transfer"},
    {"input":"Please send it to mu number","output":"Transfer"},
    {"input":"Send this to me on Whatsapp","output":"Transfer"},
    {"input":"I want to see this on my phone","output":"Transfer"},
    {"input":"Whatsapp it to me","output":"Transfer"},
    {"input":"WhatsApp it to me please","output":"Transfer"},
    {"input":"How much time to get A for License B","output":"Describe"},
    ]


    Intent_PROMPT_TEMPLATE = """
    You are an expert intent classifier. 
    Your job is to understand user input and classify into either of the following 'Describe', 'Compare', 'Others' and 'Transfer' as per the given examples.
    Do not explain.

    """

    example_prompt = PromptTemplate(input_variables=["input", "output"], template="input: {input}\n{output}")

    prompt = FewShotPromptTemplate(
        examples=examples, 
        example_prompt=example_prompt, 
        prefix=Intent_PROMPT_TEMPLATE,
        suffix="input: {input}", 
        input_variables=["input"],
        example_separator='\n'
    )

    conversational_memory_intent = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=20,
        return_messages=True
    )

    llmchain = LLMChain(llm=master_llm, prompt=prompt)

    intent = llmchain.run(UserPrompt)
#     print(intent)
    
    return intent


def entity_recognition_main(UserPrompt, LicenseName,master_llm):
    entity = []
    
    try:
        prompt_template = PromptTemplate(
                    input_variables = ['UserPrompt', 'LicenseName'],
                    template= """You are a Named Entity Recognition expert. You can find relevant entities from a list based on the mentioned user prompt. Following is a list of license names: {LicenseName}. Extract relevant entities from the list of license names based on User Prompt.Extract exact license names from the list matching the extracted entities. Make sure the extracted entities are present in the given list of license names. Strictly make sure that output includes the entity from the given list of license name and not the entity from User Prompt. Answer should only be in a python list format. Make sure to match entities on synonyms as well. Remove duplicate entities from the response. If there is no match, print blank list. Again, strictly make sure the entity returned must be an exact match of license names in the provided list. Following is the user prompt: {UserPrompt}""")

        chain = LLMChain(llm=master_llm, prompt=prompt_template)

        input_1 = chain({'UserPrompt':UserPrompt,'LicenseName':LicenseName})['text']
        output_1 = ast.literal_eval(input_1)
        length_1 = len(output_1)
#         print(output_1)
        entity = output_1
        
    except:
        entity = []
    print("MAIN - ",entity)
    return entity

def entity_recognition_sub(UserPrompt, LicenseFetched,ServiceName,master_llm):
    entity = []
    
    try:
        prompt_template = PromptTemplate(
                    input_variables = ['UserPrompt','ServiceName'],
                    template= """You are a Named Entity Recognition expert. You can find relevant entities from a list based on the mentioned user prompt. Following is a list of service names: {ServiceName}. Extract relevant entities from the list of service names based on User Prompt.Extract exact service names from the list matching the extracted entities. Make sure the extracted entities are present in the given list of service names. Strictly make sure that output includes the entity from the given list of service name and not the entity from User Prompt. Answer should only be in a python list format. Make sure to match entities on synonyms as well. Remove duplicate entities from the response. If there is no match, print blank list. Again, strictly make sure the entity returned must be an exact match of service names in the provided list. Following is the user prompt: {UserPrompt}""")

        chain = LLMChain(llm=master_llm, prompt=prompt_template)

        input_1 = chain({'UserPrompt':UserPrompt,'ServiceName':ServiceName})['text']
        output_1 = ast.literal_eval(input_1)
        length_1 = len(output_1)
#         print(output_1)
        entity = output_1
        
    except:
        entity = []
    
    print("SUB - ",entity)
    return entity


def describe(entity,entity_sub, final_json, UserPrompt,master_llm,session_memory,conversational_memory):

    prompt = f"Provide answer based on {entity} and {entity_sub} on the following data {final_json} and frame as per the following question {UserPrompt}."

    prompt_template = PromptTemplate(
                        input_variables = ['context'],
                        template = "You are an online assistant for Saudi Business Centre. You will provide concise and to the point answers to the user questions. Answer must be well-structured. Do not assume anything while providing output. Strictly make sure no detail gets missed from the available data. Make sure the tone is formal and friendly. Make sure put the answer like a conversation with filler words. If the context asks to provide list, then don't include any extra detail in the answer. Provide answer to the following: {context}")

    chain = LLMChain(llm=master_llm, prompt=prompt_template,memory = conversational_memory)

    final_text = chain({'context':prompt})['text']

    session_memory[UserPrompt] = final_text
    
    print(final_text)
    
    return final_text

def compare(entity, entity_sub,final_json, UserPrompt,master_llm,session_memory,conversational_memory):
    prompt = f"Compare the following {entity} {entity_sub} on their following data {final_json} and frame as per the following question {UserPrompt}"

    prompt_template = PromptTemplate(
                        input_variables = ['context'],
                        template = "You are an online assistant for Saudi Business Centre. You will provide concise and to the point answers to the user questions. Ensure the answers are well-structured. Do not assume anything while providing output. Make sure the tone is formal and friendly. Make sure put the answer like a conversation with filler words. Provide answer to the following: {context}")

    chain = LLMChain(llm=master_llm, prompt=prompt_template,memory = conversational_memory)

    final_text = chain({'context':prompt})['text']

    session_memory[UserPrompt] = final_text

    print(final_text)
    
    return final_text

def set_vector_store(index_name,embeddings,embedding_model):
    text_field = 'text'  # field in metadata that contains text content
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=len(embeddings[0]),
            metric='cosine'
        )
        # wait for index to finish initialization
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    index = pinecone.Index(index_name)
    
#     time.sleep(180)

#     batch_size = 32

#     for i in range(0, len(data), batch_size):
#         i_end = min(len(data), i+batch_size)
#         batch = data.iloc[i:i_end]
#         ids = [f"{x['id']}" for i, x in batch.iterrows()]
#         texts = [x['RowAsJSON'] for i, x in batch.iterrows()]
#         embeds = embedding_model.embed_documents(texts)
#         # get metadata to store in Pinecone
#         metadata = [
#             {'text': x['RowAsJSON']} for i, x in batch.iterrows()
#         ]
#         print(metadata)
#         # add to Pinecone
#         index.upsert(vectors=zip(ids, embeds, metadata))
    
#     time.sleep(180)
    
    vectorstore = Pinecone(index, embedding_model.embed_query, text_field)
    
    return vectorstore


def RAG(UserPrompt,master_llm,session_memory,conversational_memory):

    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"],model_kwargs={'mode': 'text-embedding-ada-002'})
    docs = ["Combined_data_final_V2.csv"]
    
    embeddings = embedding_model.embed_documents(docs)
    vectorstore = set_vector_store('rag-sbc',embeddings,embedding_model)
    
    prompt = """You are an online assistant for Saudi Business Centre. You will provide concise and to the point answers to the user questions. Do not assume anything while providing output. Make sure the tone is formal and friendly. If the answer is not present in given data, say that you don't know the answer. You also have the ability to answer questions based on scenarios provided by user. The logical answer if provided must cover most relevant scopes of the data. Answer the following User query"""       

    qa = RetrievalQA.from_chain_type(
    llm=master_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory = conversational_memory
    )
    
    final_text = qa.run(prompt+":"+UserPrompt)
    
    session_memory[UserPrompt] = final_text
    
    print(final_text)
    
    return final_text

def WhatsApp_Temporary(master_llm,session_memory):

    question = list(session_memory.keys())[-1]
    print('Question:', question)
    answer = session_memory[question]
    print('Answer:', answer)

    final_prompt_inst = """Your job is to summarise the given question-answer pair in strictly not more than 1500 characters.

    Question: {question} 
    Answer: {answer}

    Separate the question summarization under the heading 'Question' and 'Answer'.
    Remove unnecessary new line characters from the answers.
    Extract only the important content."""

    prompt_template_final = PromptTemplate(template = final_prompt_inst,input_variables=['question','answer'] )

    chain = LLMChain(llm=master_llm, prompt=prompt_template_final)

    final_text_tosend = chain({'question':question,'answer':answer})['text']
    print('final_text_tosend:', final_text_tosend)
    print(len(final_text_tosend))
    
    return(final_text_tosend)


def final_output(UserPrompt,LicenseName,ServiceName,master_llm,session_memory,json_result,conversational_memory):
    final_json = {}
    intent = intent_recognition(UserPrompt,master_llm)
    print(intent)
    print()
    entity_main = entity_recognition_main(UserPrompt, LicenseName,master_llm)

    entity_sub = entity_recognition_sub(UserPrompt, entity_main,ServiceName,master_llm)

    entity  = entity_main
    print(entity)
    print()


    type_picked = ""
    
    try:
        json_data = json.loads(json_result)
        filtered_data = {key: json_data[key] for key in entity}
        filtered_json = json.dumps(filtered_data, indent=2)
        final_json = json.loads(filtered_json)
    except:
        final_json = {}
    
    c = 0
    print(final_json)  
    
    if intent in ['Transfer']:
        print(intent)
        output = "I have shared our last conversation on your WhatsApp"
        text_to_send = WhatsApp_Temporary(master_llm,session_memory)
        print(output)
        output = text_to_send
        type_picked = 'transfer'
    
               
    elif len(entity)>0: 
        for i in entity:
            if i in LicenseName:
                c+=1
        print(c)
        if c > 0 or c == len(entity):
            if intent in ['Describe'] and len(entity)>0:
                try:
                    print('Describe')
                    print(UserPrompt)
                    output = describe(entity,entity_sub, final_json, UserPrompt,master_llm,session_memory,conversational_memory)
                except:
                    print(UserPrompt)
                    output = RAG(UserPrompt,master_llm,session_memory,conversational_memory)
                    #output = formatted_output(output,master_llm,conversational_memory)
                    print("RAG0")
                #output = formatted_output(output,master_llm,conversational_memory)

            elif intent in ['Compare'] and len(entity)>0:
                print('Compare')
                print(UserPrompt)
                output = compare(entity, entity_sub,final_json, UserPrompt,master_llm,session_memory,conversational_memory)
                #output = formatted_output(output,master_llm,conversational_memory)

            else:
                print(UserPrompt)
                output = RAG(UserPrompt,master_llm,session_memory,conversational_memory)
                #output = formatted_output(output,master_llm,conversational_memory)
                print("RAG1")

        else:
            print(UserPrompt)
            output = RAG(UserPrompt,master_llm,session_memory,conversational_memory)
            #output = formatted_output(output,master_llm,conversational_memory)
            print("RAG2")
    else:
        print(UserPrompt)
        output = RAG(UserPrompt,master_llm,session_memory,conversational_memory)
        #output = formatted_output(output,master_llm,conversational_memory)
        print("RAG3")

    return output,type_picked