from langchain_core.prompts import (
    ChatPromptTemplate, FewShotChatMessagePromptTemplate
)
from main import model; 


#we difine the expmples for the few shot learning
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "5-1", "output": "4"},
    {"input": "12+13", "output": "25"},
]
exemple_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"), 
        ("ai", "{output}"), 
    ]
)



few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt= exemple_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
#print(final_prompt.format(input="5+6"))

print(model.invoke(final_prompt.format(input="12+36")).content); 

