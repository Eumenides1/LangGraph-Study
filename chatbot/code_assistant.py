from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import getpass
import os

# 使用 html 解析器，将网页读取为文档，根据网页metadata["source"]进行排序，并反转
url = "https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel"

loader = RecursiveUrlLoader(
    url=url,
    max_depth=20,
    extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

### Open AI
# 提示词
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """您是一位精通 LCEL（LangChain 表达式语言）的编码助手。\n 
                这里是完整的 LCEL 文档集：\n ------- \n  {context} \n ------- \n 根据上述提供的文档回答用户的问题。确保您提供的任何代码都可以执行，\n 
                包含所有必要的导入和已定义的变量。用对代码解决方案的描述来构建您的回答。\n
                然后列出导入部分。最后列出功能代码块。这是用户的问题：""",
        ),
        ("placeholder", "{messages}"),
    ]
)


class code(BaseModel):
    """Code output"""
    prefix: str = Field(description="问题和方法的描述")
    imports: str = Field(description="代码块的导入语句")
    code: str = Field(description="不包含导入语句的代码块")
    description = "关于 LCEL 问题的代码解决方案的标准代码返回格式"


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("OPENAI_API_KEY")
# _set_env("API_BASE_URL")
_set_env("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Code Assistant"
expt_llm = "gpt-4-0125-preview"
api_key = os.environ.get("MY_API_KEY")
base_url = os.environ.get("OPENAI_API_BASE")

llm = ChatOpenAI(temperature=0, model=expt_llm, openai_api_key=api_key, openai_api_base=base_url)
code_gen_chain = code_gen_prompt | llm.with_structured_output(code)

# Test
question = "在 LCEL中，如何构建一个 RAG 应用?"
solution = code_gen_chain.invoke(
    {"context": concatenated_content, "messages": [("user", question)]}
)
print(solution)
