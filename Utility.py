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


def MasterLLM(Role, FewShotExamples, UserPrompt):
    Role = [{"role": "system", "content": Role}]
    UserPrompt = [{"role": "user", "content": UserPrompt}]
    message = [*Role, *FewShotExamples, *UserPrompt]
    response = openai.ChatCompletion.create(
      engine="GPT4_32k",
      messages= message,
      temperature=0,
        # max_tokens=350,
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None
    )
    return response['choices'][0]['message']['content']

def intent_recognition(UserPrompt):
    
    IntentRole = "You are an expert intent classifier. Your job is to understand user input and classify into either of the following 'Describe', 'Compare', 'Others' and 'Transfer' as per the given examples.Do not explain"
 
    IntentExamples = [
        {"role": "user", "content": "What is the difference between step X and step y for a opening a XYZ?"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Compare differences between the following services"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Differentiate the following services on the basis of price"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Which of the follwing is better"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Explain the difference between pricing of service A and B"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Why should I choose service A over service B"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Weigh the following services on their requirements"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Service A vs Service B"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "Describe requirements for the following service"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Tell me about A"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Tell me about wholesale of software and retail sale of software"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How can I appply for the following service"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "What are the conditions for A"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How much is the price to apply for the following service"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Explain the requirements of service A"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Define the important characteristics of the following service"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Difference between the following services"},
        {"role": "assistant", "content": "Compare"},
        {"role": "user", "content": "How to do service A"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How much time to get A for License B"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Can you help me with XYZVS"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Please help me with XYZVS"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How to enquire service XYZVS"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How to access the service XYZVS"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How to get information on service XYZVS"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How to find about service XYZVS"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Provide info on something"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Can you send this to me"},
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "Send it to my number"},
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "This is my number XXXXXXXXX Send it to me"},
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "Transfer it to my nummber"},
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "Send this to me on Whatsapp"},
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "I want to see this on my phone"},
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "Whatsapp it to me"},
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "WhatsApp it to me please"},
        {"role": "assistant", "content": "Transfer"}
    ]
   
    intent = MasterLLM(IntentRole, IntentExamples, UserPrompt)

    return intent

def entity_recognition_main(UserPrompt, LicenseName):
    entity = []
    
    try:
        EntityRole = f"You are a Named Entity Recognition expert. You can find relevant entities from a list based on the mentioned user prompt. Following is a list of license names: {LicenseName}. Extract relevant entities from the list of license names based on User Prompt.Extract exact license names from the list matching the extracted entities. Make sure the extracted entities are present in the given list of license names. Strictly make sure that output includes the entity from the given list of license name and not the entity from User Prompt. Answer should only be in a python list format. Make sure to match entities on synonyms as well. Remove duplicate entities from the response. If there is no match, print blank list. Again, strictly make sure the entity returned must be an exact match of license names in the provided list"
        EntityExamples = []    
        entity = ast.literal_eval(MasterLLM(EntityRole, EntityExamples, UserPrompt))

    except:
        entity = []
        
    entity = list(set(entity))

    return entity


def entity_recognition_sub(UserPrompt, LicenseFetched,License_Service):
    entity = []
    
    Final = License_Service[License_Service['License Name'].isin(LicenseFetched)]
    # print(Final)
    try:
        EntityRole = f"You are a Named Entity Recognition expert. You can find relevant entities from a list based on the mentioned user prompt. Using the following data: {Final}. Extract relevant entities from the list of service names based on User Prompt.Extract exact service names from the list matching the extracted entities. Make sure the extracted entities are present in the given list of service names. Strictly make sure that output includes the entity from the given list of service name and not the entity from User Prompt. Answer should only be in a python list format. Make sure to match entities on synonyms as well, like service fees as price. Remove duplicate entities from the response. If there is no match, print blank list. Again, strictly make sure the entity returned must be an exact match of service names in the provided list"
        EntityExamples = []    
        entity = ast.literal_eval(MasterLLM(EntityRole, EntityExamples, UserPrompt))

    except:
        entity = []
        
    entity = list(set(entity))

    return entity

def describe(entity, final_json, UserPrompt):
    
    DescribeRole = "You are an online assistant for Saudi Business Centre. You will provide concise and to the point answers to the user questions. Ensure answers are in bullets only. Do not assume anything while providing output. Make sure the tone is formal and friendly. Make sure put the answer like a conversation with filler words. Ensure that the answers are in a helpful tone and user friendly.Make sure the answer is not detailed unless asked specifically by the user,strictly a maximum of four lines. But make sure the short answer must cover the crux of all entities in the given context as per user prompt"
    DescribeExamples = []
    UserPrompt = f"Describe {entity} on the following data {final_json} as per the following question {UserPrompt}."
    
    final_text = MasterLLM(DescribeRole, DescribeExamples, UserPrompt)
  
    return final_text

def compare(entity, final_json, UserPrompt):
    
    CompareRole = "You are an online assistant for Saudi Business Centre. You will provide concise and to the point answers to the user questions. Ensure the answers are in bullets only. Do not assume anything while providing output. Make sure the tone is formal and friendly. Make sure put the answer like a conversation with filler words. Make sure the answer is not detailed unless asked specifically by the user, strictly a maximum of four lines. But make sure the short answer must cover the crux of all entities in the given context as per user prompt Ensure that the answers are in a helpful tone and user friendly.. Provide answer to the following: {context}"
    CompareExamples = []
    UserPrompt = f"Compare the following {entity} on their following data {final_json} as per the following question {UserPrompt}"
                        
    final_text = MasterLLM(CompareRole, CompareExamples, UserPrompt)

    return final_text

def RAG(UserPrompt,entity,qa_chain_for_rag):

    listToStr = ', '.join([str(elem) for i, elem in enumerate(entity)]) 
    prompt = f"""You are an online assistant for Saudi Business Centre. You will provide concise and to the point answers to the user questions according to the context provided. Ensure the answers are in bullets only. Do not assume anything while providing output. Make sure the tone is formal and friendly.Make sure you're able to answer generic questions from the provided data. Answer the question given the information in those
    contexts. Apart from questions related to licenses and services, you can also use information from data to answer user queries about businesses, investments and how SBC can help them achieve the same. If you cannot find the answer to the question, say "I don't know".Answer the following User query. Make sure the answer is not detailed unless asked specifically by the user, strictly a maximum of four lines. But make sure the short answer must cover the crux of all entities in the given context as per user prompt"""
    
    if len(entity) == 0:
        UserPrompt = UserPrompt
    else:
        UserPrompt = UserPrompt + f" using the following licenses {listToStr}"
    
    final_text = qa_chain_for_rag.run(prompt+":"+UserPrompt)
 
    return final_text

def set_vector_store(index_name):

    embedding_model = OpenAIEmbeddings(openai_api_key = openai.api_key, 
                              deployment="text-embedding-ada-002",
                              model="text-embedding-ada-002",
                              openai_api_base="https://di-sandbox-gpt4.openai.azure.com/",
                              openai_api_type="azure")  
    
    docs = ["Combined_data_final_V2.csv"]

    embeddings = embedding_model.embed_documents(docs)

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

def sentiment(UserPrompt, PrevEntity):
                       
    MemoryRole = "You are a sentiment analysis assistant. You analyze users prompt and decide whether it requires a conversational context or not. If the user prompt requires memory addition, you will answer as 'Memory' else you will answer as 'Direct'. Make sure to focus on findind pronouns like That, Those, Them, Their"
    
    MemoryExamples=[
        {"role": "user", "content": "Can you share more details about the above license?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Is the above better than license AAAAA"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Share details for all of the above"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Do above licenses require anything"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you Tell me more about this"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you tell me about their prices/fees"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Tell me more about it in detail"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How is it better than service BB?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "What is the price for it"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How can I get that license"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Are there any alternatives to this ?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Is there anything additional I should know"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you explain to me further ?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "What are the key differences between those?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Do I XXXXXX for this?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you elaborate more"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How long does it take"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How much time will it take?"},
        {"role": "assistant", "content": "Memory"}
        ]
    
    if_memory = MasterLLM(MemoryRole, MemoryExamples, UserPrompt)
    print(if_memory)
    print()
    
    listToStr = ', '.join([str(elem) for i, elem in enumerate(PrevEntity)]) 

    if if_memory in 'Direct' or len(PrevEntity) == 0:
        FullPrompt = UserPrompt
        PrevEntity = []
    else:
        FullPrompt = f"{UserPrompt}, where previous conversation had been about {listToStr}"
    
        
    return FullPrompt, PrevEntity
    