from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

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
from langchain.prompts import PromptTemplate
from Utility_New import MasterLLM, intent_recognition, entity_recognition_main, describe, compare, RAG, sentiment

openai.api_type = "azure"
openai.api_base = "https://di-sandbox-gpt4.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "69ec3919a7314784be9c4f7414286fba"


os.environ['OPENAI_API_KEY']=openai.api_key
os.environ['OPENAI_API_BASE'] = openai.api_base
os.environ['OPENAI_API_VERSION'] = openai.api_version
os.environ['OPENAI_API_TYPE'] = openai.api_type

#Pinecone - Statements
PINECONE_API_KEY = '98bbd113-f65a-403b-b44f-507e60506d46'

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '98bbd113-f65a-403b-b44f-507e60506d46',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
)

#Pinecone - JSON (orig)
# PINECONE_API_KEY = '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7'

# pinecone.init(
#     api_key=os.environ.get('PINECONE_API_KEY') or '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7',
#     environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
# )


index_name = 'rag-sbc'

embedding_model = OpenAIEmbeddings(openai_api_key = os.environ.get('OPENAI_API_KEY'), 
                              deployment="text-embedding-ada-002",
                              model="text-embedding-ada-002",
                              openai_api_base=os.environ.get('OPENAI_API_BASE'),
                              openai_api_type=os.environ.get('OPENAI_API_TYPE'))
    

index = pinecone.Index(index_name)

vectorstore = Pinecone(index, embedding_model.embed_query, 'text')


llm = AzureChatOpenAI(deployment_name = "GPT4_32k", model_name = "gpt-4-32k", temperature=0)

llm_35 = AzureChatOpenAI(deployment_name = "GPT_35_16k", model_name = "gpt-35-turbo-16k", temperature=0)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True)

session_memory = {}
entities = {}

# Latest data - SBC_Final_License_Level_Data.csv

# OLD JSON
# data = pd.read_csv('Combined_data_SBC.csv')

# Latest Summarized Data - New JSON
data = pd.read_csv('SBC_Final_License_Level_Data.csv')

data.drop(columns=['ID'],inplace=True)
# result = {}
# for main_key, group_df in data.groupby('License Name'):
#     group_dict = group_df.to_dict(orient='records')
#     result[main_key] = group_dict

# json_result = json.dumps(result, indent=2)
# json_result = json_result.replace("\\n","")
# json_result = json_result.replace("\\u2002","")
# json_result = json_result.replace("\n","")
# json_result = json_result.replace("/","")

data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
LicenseName = data['License Name'].to_list()
#License_Service = data.groupby('License Name')['Service Name'].apply(list).reset_index(name='Service Name')

def final_output(UserPrompt):
    final_json = {}
    
    intent = intent_recognition(UserPrompt)
    # if intent!='Compare':
    #     intent = "Others"

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
    
    if len(session_memory) > 0:
        lastUserPrompt = list(session_memory.keys())[-1]
        # lastUserPrompt = [i.strip() for i in lastUserPrompt]
        lastOutput = session_memory[lastUserPrompt]
        # lastOutput = [i.strip() for i in lastOutput]
    
    else:
        lastUserPrompt = ""
        lastOutput = ""


    print("Last User Prompt")
    print(lastUserPrompt)
    print("Last Output")
    print(lastOutput)
    
    
    UserPrompt, Prev_Entity = sentiment(UserPrompt, Prev_Entity, lastUserPrompt, lastOutput)
           
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

    # try:
    #     json_data = json.loads(json_result)
    #     filtered_data = {key: json_data[key] for key in entity}
    #     filtered_json = json.dumps(filtered_data, indent=2)
    #     final_json = json.loads(filtered_json)
    # except:
    #     final_json = {}
    
    try:
        filtered_data = data[data['License Name'].isin(entity)]
        filtered_data.drop(columns=["Extra Details"],inplace=True)
        filtered_data_dict = filtered_data.to_dict(orient='records')
        
    except:
        filtered_data_dict = {}
    
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
                prompt, prompt_template = describe(entity, filtered_data_dict, UserPrompt)
                if len(entity) <=3:
                    chain = LLMChain(llm=llm_35, prompt=prompt_template,memory = conversational_memory)
                else:
                    chain = LLMChain(llm=llm_35, prompt=prompt_template,memory = conversational_memory)
                output = chain({'context':prompt})['text']
                print(output)

            elif intent in ['Compare'] and len(entity) > 0:
                print('Compare')
                prompt,prompt_template = compare(entity, filtered_data_dict, UserPrompt)
                if len(entity) <=3:
                    chain = LLMChain(llm=llm_35, prompt=prompt_template,memory = conversational_memory)
                else:
                    chain = LLMChain(llm=llm_35, prompt=prompt_template,memory = conversational_memory)
                output = chain({'context':prompt})['text']
                print(output)
            else:
                print("RAG1")
                prompt = RAG(UserPrompt,entity)
                if len(entity) <=3:
                    qa = RetrievalQA.from_chain_type(llm=llm_35,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    memory = conversational_memory)
                else:
                    qa = RetrievalQA.from_chain_type(llm=llm_35,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    memory = conversational_memory)
                output = qa.run(prompt)
                print(output)


        else:
            print("RAG2")
            prompt = RAG(UserPrompt,entity)
            if len(entity) <=3:
                    qa = RetrievalQA.from_chain_type(llm=llm_35,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    memory = conversational_memory)
            else:
                qa = RetrievalQA.from_chain_type(llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                memory = conversational_memory)
            output = qa.run(prompt)
            print(output)
    else:
        print("RAG1")
        prompt = RAG(UserPrompt,entity)
        if len(entity) <=3:
                    qa = RetrievalQA.from_chain_type(llm=llm_35,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    memory = conversational_memory)
        else:
            qa = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory = conversational_memory)
        output = qa.run(prompt)
        print(output)
        
    session_memory[oldPrompt] = output
    return output, intent

def final_output_formatted(UserPrompt):

    answer,_ = final_output(UserPrompt)
    final_prompt_inst = """Your job is to summarise the given answer. 
    You need to make sure the key points of the answer are retained. 
    Your answers will be short and direct without any additional explanation. 
    You need to analyze the answer and provide human like conversational responses.

    User Prompt for your reference: {UserPrompt}
    Answer provided: {answer}

    Based on the above instructions, provide the final summarized answer."""

    prompt_template_final = PromptTemplate(template = final_prompt_inst,input_variables=['UserPrompt','answer'] )

    chain = LLMChain(llm=llm_35, prompt=prompt_template_final)

    final_text = chain({'UserPrompt':UserPrompt,'answer':answer})['text']

    return final_text


app = Flask(__name__)
CORS(app)

@app.route("/")
def get_bot():
    return render_template('index.html')

@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['user_message']
    print(f'Received user message: {user_message}')
    bot_response = final_output_formatted(user_message)
    print(f'Generated bot response: {bot_response}')
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)