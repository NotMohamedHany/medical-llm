import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool,tool

from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

from fastapi import FastAPI
from pydantic import BaseModel

#language
prompt_language=ChatPromptTemplate.from_messages([
    ("system","You are a professional translator. Your task is to translate recognize the language"),
    ("human","First, automatically detect the input language just respond with the language no more comet or text or special txt :{input}")])
llm=ChatGroq(model="llama-3.1-8b-instant")
language_detect=prompt_language|llm|StrOutputParser()
#-------------------------------------------------------
#translate
prompt_translate = ChatPromptTemplate.from_messages(["""You are a professional translator.\
    Your task is to translate the following text into {target_language}.

    - Automatically detect the input language.
    - Respond ONLY with the translated text, nothing else.

    Text: {text}
    """])
translate = prompt_translate|llm|StrOutputParser()

#---------------------------------------------------------------------------------------
#TOOLS
@tool
def symptoms_checker(query:str) -> str:
    """Use when need to analyze  symptoms
get information about the symptoms and give some recommendations and possible causes input query (string) is the patient data or symptoms """
    prompt=ChatPromptTemplate.from_messages([
    ("system",("1-You are a medical assistant AI that respond in the english or arabic language only depending on Case language."
               "2-You must always be safe, cautious, and prioritize patient well-being When analyzing patient symptoms ")),
    """When analyzing patient symptoms or (vital signs, measurements, labs):
1. Extract structured info: symptoms, location, duration, severity, history.(if it in the case and its possible)
2. Provide possible causes (never absolute diagnosis).
3. Provide safe recommendations, including when to seek emergency care.
4. Always include a disclaimer: "This is not medical advice. Consult a doctor."
5. Remember to respond in the same language as the Input.

Notes:
1-respond in the same language the input is
1-Just respond with possible causes with the reasoning and the safe recommendations.
2-must give some safe recommendations
3-dont forget any step of analyzing
4-dont forget the disclaimer and the recommendation
5-dont overtalk
6- if u dont have a recommendation say :"recommend Consult a doctor "

Case:
{case}
"""])
    llm=ChatGroq(model="openai/gpt-oss-120b")
    chain=prompt|llm|StrOutputParser()
    return chain.invoke(query)
#---------------------------------------------------------------------------------------
@tool
def signs_checker(query:str) -> str:
    """Use when need to analyze (vital signs, measurements, labs)s
    get information about the patient's (vital signs, measurements, labs)s and give some recommendations and possible causes input query (string) is the patient data or symptoms  """
    prompt=ChatPromptTemplate.from_messages([
    ("system",("1-You are a medical assistant AI that respond in the english or arabic language only depending on Case language."
               "2-You must always be safe, cautious, and prioritize patient well-being When analyzing patient symptoms ")),
    """When analyzing patient (vital signs, measurements, labs):
1. Extract the possible problem in patient's (vital signs, measurements, labs) .
2. Provide possible causes (never absolute diagnosis).
3. Provide safe recommendations, including when to seek emergency care.
4. Always include a disclaimer: "This is not medical advice. Consult a doctor."
5-remember to respond in the same language as the Input.

Notes:
1-respond in the same language the input is
1-Just respond with possible causes with the reasoning and the safe recommendations.
2-must give some safe recommendations
3-dont forget any step of analyzing
4-dont forget the disclaimer and the recommendation
5-dont overtalk
6- if u dont have a recommendation say :"recommend Consult a doctor "

Input:
{input}
"""])
    llm=ChatGroq(model="openai/gpt-oss-120b")
    chain=prompt|llm|StrOutputParser()
    return chain.invoke(query)

#---------------------------------------------------------------------------------------


tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

tools=[tavily_search_tool,signs_checker,symptoms_checker]

#---------------------------------------------------------------------------------------

#Agent
prompt_agent =ChatPromptTemplate.from_messages(["""You Are a medical Agent that help patients to Answer the their questions as best he can.\
you give the answer in the same language as the Input(dont forget to give the final answer in the same language as the Input ).\
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
You can search to verify or to be more confidence of your answer(search the symptoms or search  the informations you get till now and wanna be sure from it)
Thought: I now know the final answer
Dont forget to give the final answer in the same language as the Input
Final Answer: the final answer to the original input question in this format:

Analysis:

    -- Abnormal Vitals:

        1-Fever (38.3°C, above normal 36–37.5°C)

        2-Mild tachycardia (104 bpm, above normal 60–100 bpm)

    -- Possible Causes:
        1-Appendicitis,
        2-gastroenteritis,
        3-ovarian cyst (if female)
        4-kidney stone.

    -- Recommendation: 
        1-Seek urgent medical evaluation. 
        2-Appendicitis often requires surgery. 
        3-Avoid eating or drinking until examined.

    --Emergency? ✅ Yes — needs ER visit.
or  ------------
Analysis:

    --Abnormal Vitals:

        1-Tachycardia (118 bpm)

        2-Hypertension (150/95 mmHg)

        3-Hypoxemia (90%, normal ≥95%)

    --Possible Causes:
        1- Myocardial infarction (heart attack)
        2- Unstable angina
        3- Arrhythmia.

    --Recommendation:
        1- Call emergency services immediately.
        2- Administer aspirin if no allergy and no contraindications.

    --Emergency? 🚨 Yes — life-threatening.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""])
#-----------------------------------------------------

llm_agent=ChatDeepSeek(model="deepseek-chat")
med_agent = create_react_agent(llm_agent, tools, prompt_agent)
medical_agent = AgentExecutor(agent=med_agent, tools=tools, verbose=False,return_intermediate_steps=True,handle_parsing_errors=True)

#-----------------------------------------------------

#APIS
app = FastAPI(title="Medical Agent API")

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}
    
class Query(BaseModel):
    question: str
@app.post("/ask")
async def ask_agent(query: Query):
    """Endpoint for Flutter to call"""
    result = await medical_agent.ainvoke({"input": query.question})
    language = await language_detect.ainvoke({"input": query.question})
    language = language.strip()
    out = await translate.ainvoke({"target_language": language, "text": result["output"]})
    return {"answer": out, "steps": result["intermediate_steps"]}

