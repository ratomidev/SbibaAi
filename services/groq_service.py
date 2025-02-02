from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL

model = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

def get_groq_response(prompt: str) -> str:
    return model.invoke(prompt).content