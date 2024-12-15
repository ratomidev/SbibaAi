import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from fastapi import FastAPI
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "What is the capital of {country}, answer by one word?., if the spelling of the contry is incorect just return false."
)

load_dotenv()
key = os.getenv('key')

Model = "llama-3.1-70b-versatile"
model = ChatGroq(api_key=key, model=Model, temperature=0)

app = FastAPI()

@app.get('/')
async def rootHome():
    return "Hello visitor"


@app.get('/{country}')
async def root(country):
    print(country)
    template = "what is the capital of "+country +" , the responce should be one world in any input if you didnt recongise the country or any problem happen you just return false ?"
    return model.invoke(prompt_template.format(country=country)).content

@app.get('/few_shot_learning/{equations}')
async def calculate(equations):
    return model.invoke(equations).content
