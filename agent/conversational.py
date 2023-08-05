from langchain import SerpAPIWrapper, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


def conversation(question):
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, memory=memory, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                   verbose=True)
    return agent_chain.run(input=question)


def conversation_chatmodel(question):
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, memory=memory, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                   verbose=True)
    return agent_chain.run(input=question)



if __name__ == "__main__":
    print(conversation("你好，我是bob"))
