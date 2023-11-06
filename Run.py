import openai
import langchain
import os
import json
from pprint import pprint
import pinecone
import time
from langchain.chat_models import AzureChatOpenAI
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
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
# from langchain.chains.openai_functions import (
#     create_openai_fn_chain,
#     create_structured_output_chain,
# )
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
from tqdm.autonotebook import tqdm 
from Utility import MasterLLM, intent_recognition, entity_recognition_main, entity_recognition_sub, describe, compare, RAG, set_vector_store, sentiment



openai.api_type = "azure"
openai.api_base = "https://di-sandbox-gpt4.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "69ec3919a7314784be9c4f7414286fba"



os.environ['OPENAI_API_KEY']=openai.api_key
os.environ['OPENAI_API_BASE'] = openai.api_base
os.environ['OPENAI_API_VERSION'] = openai.api_version
os.environ['OPENAI_API_TYPE'] = openai.api_type


data = pd.read_csv(r"Combined_data_final_V2.csv")
DataList = data.JSON_Data.values.tolist()


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


PINECONE_API_KEY = '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7'

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
)


def final_output(UserPrompt):
    final_json = {}
    
    intent = intent_recognition(UserPrompt)
    
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
    
    entity_sub = entity_recognition_sub(UserPrompt, entity_main,License_Service)
    print('Sub Entity')
    print(entity_sub)
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
                output = describe(entity, final_json, UserPrompt)

            elif intent in ['Compare'] and len(entity) > 0:
                print('Compare')
                output = compare(entity, final_json, UserPrompt)

            else:
                output = RAG(UserPrompt, entity)
                print("RAG1")
        else:
            output = RAG(UserPrompt, entity)
            print("RAG2")
    else:
        output = RAG(UserPrompt, entity)
        print("RAG3")
        
    session_memory[oldPrompt] = output

    return output, intent


UserPrompt = "Explain retail licenses"
pprint(final_output(UserPrompt))