import os

from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.tools import tool
from langchain.prompts import MessagesPlaceholder

# openai_api_key = "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"
# os.environ['OPENAI_API_KEY'] = openai_api_key


@tool
def get_word_length(word: str) -> int:
    """
    Returns the length of a word.
    不能独立一个文件，不然会报 1 validation error for OpenAIFunctionsAgent
    """
    return len(word)


def word_length(question):
    llm = ChatOpenAI(temperature=0)
    tools = [get_word_length]

    memory_key = "chat_history"
    system_message = SystemMessage(content="You are very powerful assistant, but bad at calculating lengths of words.")
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
    memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    return agent_executor.run(question)


if __name__ == '__main__':
    print(word_length("how many letters in the word educa?"))
