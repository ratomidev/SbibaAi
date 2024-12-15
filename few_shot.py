from langchain_core.prompts import (
    ChatPromptTemplate, FewShotChatMessagePromptTemplate
)


#we difine the expmples for the few shot learning
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "5-1", "output": "4"},
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
print(few_shot_prompt.format())