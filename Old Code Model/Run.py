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
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from twilio.rest import Client
import gradio as gr
import random
import time
from Utility import final_output, intent_recognition, entity_recognition_main,entity_recognition_sub, describe, compare, set_vector_store, RAG, WhatsApp_Temporary


account_sid = 'ACc5cb24a0a7e04e81d40474b04eee5f36'
auth_token = '0013d5099e6224e40bafcc1da8fb7cc3'
client = Client(account_sid, auth_token)

session_memory = {}
data_dict = {}

data_json = pd.read_csv('Combined_data_final_V2.csv')
data = pd.read_csv('Combined_data_SBC.csv')
data.drop(columns=['ID'],inplace=True)

result = {}
for main_key, group_df in data.groupby('License Name'):
    group_dict = group_df.to_dict(orient='records')
    result[main_key] = group_dict

json_result = json.dumps(result, indent=2)
json_result = json_result.replace("\\n","")
json_result = json_result.replace("\n","")


LicenseName = data['License Name'].to_list()

ServiceName = set(data['Service Name'].to_list())


PINECONE_API_KEY = '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7'

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '49e9d57f-ca7b-45d8-9fe5-b02db54b2dc7',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
)

API_KEY = "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"#"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"
model_id = "gpt-3.5-turbo-16k"
os.environ["OPENAI_API_KEY"] = API_KEY

master_llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.0)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=20,
    return_messages=True
)

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=True)


def bot(history):
    User_Prompt = history[-1][0]
        
    final_output_show,type_picked = final_output(User_Prompt,LicenseName,ServiceName,master_llm,session_memory,json_result,conversational_memory)
    
    
    if type_picked == 'Transfer':
        final_output_show  ="I have shared our last conversation on your WhatsApp"
        message = client.messages.create(
          from_='whatsapp:+14155238886',
          body=final_output_show,
          to='whatsapp:+966539719286'
        )
        print(message)
    
    response = final_output_show
    data_dict[history[-1][0]] = response
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=0.10):
            submit_btn = gr.Button(
                'Submit',
                variant='primary'
            )
        with gr.Column(scale=0.10):
            clear_btn = gr.Button(
                'Clear',
                variant='stop'
            )
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

demo.queue()
if __name__ == "__main__":
    demo.launch(share=True)






