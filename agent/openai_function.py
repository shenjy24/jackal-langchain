import langchain
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


def openai_func(question):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    db = SQLDatabase.from_uri("sqlite:///../../../../../notebooks/Chinook.db")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name="FooBar-DB",
            func=db_chain.run,
            description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context"
        )
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    return agent.run(question)


def openai_multi_func(question):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    search = SerpAPIWrapper()

    # Define a list of tools offered by the agent
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful when you need to answer questions about current events. You should ask targeted questions.",
        ),
    ]
    mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)

    # Do this so we can see exactly what's going on under the hood
    langchain.debug = True

    return mrkl.run(question)
