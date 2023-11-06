import openai
from langchain.chains.summarize import load_summarize_chain
import os
import json
from pprint import pprint
import pinecone
import time
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from typing import Optional
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings 
from Utility_New import MasterLLM, intent_recognition, entity_recognition_main, describe, compare, RAG, sentiment

openai.api_type = "azure"
openai.api_base = "https://di-sandbox-gpt4.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "69ec3919a7314784be9c4f7414286fba"


os.environ['OPENAI_API_KEY']=openai.api_key
os.environ['OPENAI_API_BASE'] = openai.api_base
os.environ['OPENAI_API_VERSION'] = openai.api_version
os.environ['OPENAI_API_TYPE'] = openai.api_type


PINECONE_API_KEY = '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7'

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
)

index_name = 'rag-sbc'

embedding_model = OpenAIEmbeddings(openai_api_key = os.environ.get('OPENAI_API_KEY'), 
                              deployment="text-embedding-ada-002",
                              model="text-embedding-ada-002",
                              openai_api_base=os.environ.get('OPENAI_API_BASE'),
                              openai_api_type=os.environ.get('OPENAI_API_TYPE'))
    

index = pinecone.Index(index_name)

vectorstore = Pinecone(index, embedding_model.embed_query, 'text')


llm = AzureChatOpenAI(deployment_name = "GPT4_32k", model_name = "gpt-4-32k", temperature=0)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True)

session_memory = {}
entities = {}

data_json = pd.read_csv('Combined_data_final_V2.csv')
data = pd.read_csv('Combined_data_SBC.csv')
data.drop(columns=['ID'],inplace=True)
result = {}
for main_key, group_df in data.groupby('License Name'):
    group_dict = group_df.to_dict(orient='records')
    result[main_key] = group_dict

json_result = json.dumps(result, indent=2)
json_result = json_result.replace("\\n","")
json_result = json_result.replace("\\u2002","")
json_result = json_result.replace("\n","")
json_result = json_result.replace("/","")

data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
LicenseName = data['License Name'].to_list()
License_Service = data.groupby('License Name')['Service Name'].apply(list).reset_index(name='Service Name')

def final_output(UserPrompt):
    final_json = {}
    
    intent = intent_recognition(UserPrompt)
    intent = "Others"
    print('Intent')
    print(intent)
    print()
    
    oldPrompt = UserPrompt
    
    if len(list(entities.keys())) == 0:
        PrevEntity = []
    else:
        PrevEntity = entities[list(entities.keys())[-1]]
    
    Prev_Entity = [i.strip() for i in PrevEntity]
    Prev_Entity = list(set(Prev_Entity))
    print('Prev_Entity')
    print(Prev_Entity)
    print()

    UserPrompt, Prev_Entity = sentiment(UserPrompt, Prev_Entity)
           
    print('New Prev Entity')
    print(Prev_Entity)
      
    print('UserPrompt')
    print(UserPrompt)
    print()

    entity_main = entity_recognition_main(UserPrompt, LicenseName)
    entity_main = list(set(entity_main))
    print('Main Entity')
    print(entity_main)
    print()

    entities[oldPrompt] = entity_main
    
    for i in Prev_Entity:
        entity_main.append(i)
        
    entity  = entity_main
    entity = list(set(entity))
    print('Full Entity')
    print(entity)
    print()

    try:
        json_data = json.loads(json_result)
        filtered_data = {key: json_data[key] for key in entity}
        filtered_json = json.dumps(filtered_data, indent=2)
        final_json = json.loads(filtered_json)
    except:
        final_json = {}
    
    c = 0

    if intent in ['Transfer']:
        print(intent)
        print()
        output = "I have shared our last conversation on your WhatsApp" 
        return output, intent
               
    elif len(entity)>0: 
        for i in entity:
            if i in LicenseName:
                c+=1

        if c > 0 and c == len(entity):
            if intent in ['Describe'] and len(entity) > 0:
                print('Describe')
                prompt, prompt_template = describe(entity, final_json, UserPrompt)
                chain = LLMChain(llm=llm, prompt=prompt_template,memory = conversational_memory)
                output = chain({'context':prompt})['text']
                print(output)

            elif intent in ['Compare'] and len(entity) > 0:
                print('Compare')
                prompt,prompt_template = compare(entity, final_json, UserPrompt)
                chain = LLMChain(llm=llm, prompt=prompt_template,memory = conversational_memory)
                output = chain({'context':prompt})['text']
                print(output)
            else:
                print("RAG1")
                prompt, prompt_template = RAG(UserPrompt,entity)
                qa = RetrievalQA.from_chain_type(llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                memory = conversational_memory,
                chain_type_kwargs={"prompt": prompt_template})
                output = qa.run(prompt)
                print(output)


        else:
            print("RAG2")
            prompt, prompt_template = RAG(UserPrompt,entity)
            qa = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory = conversational_memory,
            chain_type_kwargs={"prompt": prompt_template})
            output = qa.run(prompt)
            print(output)
    else:
        print("RAG1")
        prompt, prompt_template = RAG(UserPrompt,entity)
        qa = RetrievalQA.from_chain_type(llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory = conversational_memory,
        chain_type_kwargs={"prompt": prompt_template})
        output = qa.run(prompt)
        print(output)
        
    session_memory[oldPrompt] = output

    return output, intent




