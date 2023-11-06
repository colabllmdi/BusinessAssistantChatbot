import openai
import ast
from langchain.prompts import PromptTemplate


def MasterLLM(Role, FewShotExamples, UserPrompt):
    Role = [{"role": "system", "content": Role}]
    UserPrompt = [{"role": "user", "content": UserPrompt}]
    message = [*Role, *FewShotExamples, *UserPrompt]
    response = openai.ChatCompletion.create(
      engine="GPT4_32k",
      messages= message,
      temperature=0,
    )
    return response['choices'][0]['message']['content']

def intent_recognition(UserPrompt):
    
    IntentRole = """You are an expert intent classifier. 
            Your job is to understand user input and classify into either of the following 
                    'Describe', 'Compare', 'Others' and 'Transfer' as per the given examples.
            Do not explain"""
 
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
        {"role": "user", "content": "how much time will it take?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "how long will it take to get all approvals in place?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Cost for this?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "What is the cost for it?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Hou mcuh will it cost?"},
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
        {"role": "assistant", "content": "Transfer"},
        {"role": "user", "content": "how much time will it take?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "how much time will it take?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "how long will it take to get all approvals in place?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Cost for this?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "What is the cost for it?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "Hou mcuh will it cost?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "What is the total number of possible licenses?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "What is the total number of licenses?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "How many licenses are there?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "What is the total count of licenses available?"},
        {"role": "assistant", "content": "Describe"},
        {"role": "user", "content": "What are the licenses offered by Ministry of XXXXX?"},
        {"role": "assistant", "content": "Describe"},

    ]
   
    intent = MasterLLM(IntentRole, IntentExamples, UserPrompt)

    return intent

def entity_recognition_main(UserPrompt, LicenseName):
    entity = []
    
    try:
        EntityRole = f"""You are a Named Entity Recognition expert. 
                You can find relevant entities from a list based on the mentioned user prompt. 
                Following is a list of license names: {LicenseName}. 
                Extract relevant entities from the list of license names based on User Prompt.
                Extract exact license names from the list matching the extracted entities. 
                Make sure the extracted entities are present in the given list of license names. 
                Strictly make sure that output includes the entity from the given list of license name and not the entity from User Prompt. 
                Answer should only be in a python list format. 
                Make sure the list is in alphabetical order.
                Make sure to match entities on synonyms as well. 
                Remove duplicate entities from the response. 
                If there is no match, print blank list. 
                Again, strictly make sure the entity returned must be an exact match of license names in the provided list"""
        EntityExamples = []    
        entity = ast.literal_eval(MasterLLM(EntityRole, EntityExamples, UserPrompt))

    except:
        entity = []
        
    entity = list(set(entity))

    return entity


def describe(entity, final_json, UserPrompt):
    
    prompt = f"Describe in strictly not more than 2 lines - {entity} on the following data: {final_json} as per the following question: {UserPrompt}. "
    
    prompt_template = PromptTemplate(
                        input_variables = ['context'],
                        template = """You are an online assistant for Saudi Business Centre. 
                        You will provide concise and direct answers without any explanation. 
                        Answer must be well-structured and short in not more than 4 lines.
                        Ensure the answers are in bullet points.
                        Strictly make sure to give the answer human-like and friendly feel.
                        Do not explain the answer by additional information.
                        Make sure to understand and match any synonyms.
                        Provide answer to the following: {context}""")


    return prompt,prompt_template

def compare(entity,final_json, UserPrompt):
    prompt = f"Compare the following {entity} on their following data {final_json} and frame in strictly not more than 2 lines as per the following question {UserPrompt}"

    prompt_template = PromptTemplate(
                        input_variables = ['context'],
                        template = """You are an online assistant for Saudi Business Centre. 
                        You will provide concise and direct answers without any explanation.  
                        Answer must be well-structured and short in not more than 4 lines.
                        Ensure the answers are in bullet points.
                        Strictly make sure to give the answer human-like and friendly feel.
                        Do not explain the answer by additional information.
                        Make sure to understand and match any synonyms.
                        Provide answer to the following: {context}""")
    return prompt,prompt_template

def RAG(UserPrompt,entity):

    context = """You are an online assistant for Saudi Business Centre. 
                        You will provide concise and direct answers without any explanation.  
                        Answer must be well-structured and short in not more than 4 lines.
                        Ensure the answers are in bullet points.
                        Strictly make sure to give the answer human-like and friendly feel.
                        Make sure to understand and match any synonyms.
                        Do not explain the answer by additional information. 
    User Prompt:"""
    if len(entity) != 0:
        listToStr = ', '.join([str(elem) for i, elem in enumerate(entity)])
        prompt = f"{UserPrompt}"+f"using the following licenses {listToStr}"
    else:
        prompt = f"{UserPrompt}"                     
    
    # prompt_template = PromptTemplate(
    #                 input_variables = ['context'],
    #                 template = "You are an online assistant for Saudi Business Centre. You will provide concise and to the point answers to the user questions according to the context provided. Ensure the answers are in bullets only. Do not assume anything while providing output. Make sure the tone is formal and friendly.Make sure you're able to answer generic questions from the provided data. Answer the question given the information in those contexts. Apart from questions related to licenses and services, you can also use information from data to answer user queries about businesses, investments and how SBC can help them achieve the same, these you've to figure out from the context provided. If you cannot find the answer to the question, give a 2 line explanation of why you don't know the answer. Answer must be well-structured, short and must include all essential points in strictly not more than 4 lines but make sure the short answer must cover the crux of all entities in the given context as per user prompt. User Prompt: {context}") 

    return context+":"+prompt


def isolate(PrevEntity, lastUserPrompt, lastOutput):
    
    IntentRole = """"You are an expert intent classifier. 
                    Your job is to understand user input and identify if the user wants to refer to
                    any particular entity out of the previous mentiones on in the memory.
                    Yur answer will be in either 'Isolate' or ''. 
                    Answer in python list.
                    Do not explain"""
 
    IntentExamples = [
        {"role": "user", "content": "Tell me abut the 4th one"},
        {"role": "assistant", "content": "Isolate"},
        {"role": "user", "content": "Elaborate on the 2nd license"},
        {"role": "assistant", "content": "Isolate"},
        {"role": "user", "content": "What is the price for the latter"},
        {"role": "assistant", "content": "Isolate"},
        {"role": "user", "content": "how much time will the former take"},
        {"role": "assistant", "content": "Isolate"},
        {"role": "user", "content": "Tell me about the last two"},
        {"role": "assistant", "content": "Isolate"},
        {"role": "user", "content": "Explain 1st and 3rd"},
        {"role": "assistant", "content": "Isolate"},
        {"role": "user", "content": "Tell me only about AAA and BBBB"},
        {"role": "assistant", "content": "Isolate"},
        {"role": "user", "content": "Explain about license BBBB"},
        {"role": "assistant", "content": "Isolate"}
    ]
   
    intent = MasterLLM(IntentRole, IntentExamples, lastUserPrompt)
    
    if intent in 'Isolate':
        Prev_Entity =  entity_recognition_main(lastOutput, PrevEntity)
    else:
        Prev_Entity = PrevEntity

    return Prev_Entity


def sentiment(UserPrompt, PrevEntity, lastUserPrompt, lastOutput):
                       
    MemoryRole = """You are a sentiment analysis assistant. 
            You analyze users prompt and decide whether it requires a conversational context or not. 
            If the user prompt requires memory addition, 
                    you will answer as 'Memory' else you will answer as 'Direct'. 
            Make sure to focus on finding pronouns like That, Those, Them, Their, it, this"""
    
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
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Do I need a Saudi partner??"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How much will it cost me to apply?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Is there any specific requirement?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Are there any conditions?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Are there any additional requirement?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How long will it take to get all the approvals in place?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you tell me about the prices?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you tell me about the time required?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you tell me about the available services?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you share the steps?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you share the criteria?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you share any requirements?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How can SBC help me?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you share any additional information?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Do I need a XXXXXX"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you summmarize"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How much time is required"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "What will be the total expense"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "What will be the cost and duration"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "How many licenses are there in it?"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you elaborate"},
        {"role": "assistant", "content": "Memory"},
        {"role": "user", "content": "Can you share the steps for this service?"},
        {"role": "assistant", "content": "Memory"},
        ]
    
    if_memory = MasterLLM(MemoryRole, MemoryExamples, UserPrompt)
    print(if_memory)
    print()
    
    listToStr = ', '.join([str(elem) for i, elem in enumerate(PrevEntity)]) 
    
    if if_memory in 'Memory':
        PrevEntity = isolate(PrevEntity, lastUserPrompt, lastOutput)

    if if_memory in 'Direct' or len(PrevEntity) == 0:
        FullPrompt = UserPrompt
        PrevEntity = []
    else:
        FullPrompt = f"{UserPrompt}, where previous conversation had been about {listToStr}"
    
        
    return FullPrompt, PrevEntity
    