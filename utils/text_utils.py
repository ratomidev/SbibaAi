from langchain_core.prompts import PromptTemplate

def create_prompt_template():
    return PromptTemplate.from_template(
        "What is the capital of {country}, answer by one word?., if the spelling of the contry is incorect just return false."
    )