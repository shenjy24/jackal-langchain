from langchain import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


def test1():
    template = """\
    You are a naming consultant for new companies.
    What is a good name for a company that makes {product}?
    """

    prompt = PromptTemplate.from_template(template)
    print(prompt.format(product="colorful socks"))


def test2():
    # 无变量
    no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
    print(no_input_prompt.format())
    # 单变量
    one_input_prompt = PromptTemplate(input_variables=["adjective"], template="Tell me a {adjective} joke.")
    print(one_input_prompt.format(adjective="funny"))
    # 多变量
    multiple_input_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}."
    )
    print(multiple_input_prompt.format(adjective="funny", content="chickens"))


def chat_prompt_template():
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt.format_prompt(input_language="English", output_language="French",
                                     text="I love programming.").to_messages()


if __name__ == '__main__':
    chat_prompt_template()
