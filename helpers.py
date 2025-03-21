# Imports
from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import UpstashVectorStore


# Initialize variables
OPEN_AI_API_KEY = config('OPEN_AI_KEY')
UPSTASH_VECTOR_URL = config('UPSTASH_VECTOR_ENDPOINT')
UPSTASH_VECTOR_TOKEN = config('UPSTASH_VECTOR_TOKEN')

# Create an embedding function using OpenAIEmbeddings Class
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=OPEN_AI_API_KEY
    )

# Usage of UpstashVectoreStore Class to initialize connection and storage to Upstash
store = UpstashVectorStore(
    embedding=embeddings,
    index_url=UPSTASH_VECTOR_URL,
    index_token=UPSTASH_VECTOR_TOKEN
)

# Create a retriever from Store
retriever = store.as_retriever(
    search_type='similarity',
    search_kwargs={'k':2}
)

# Configure OpenAI Model. Model Used "gpt-4o-mini" not huge
LLM_CONFIG = {
    "api_key": OPEN_AI_API_KEY,
    "model": 'gpt-4o-mini'
}

# Initialize Model 
model = ChatOpenAI(**LLM_CONFIG)
parser = StrOutputParser()


# message template
message = """
Answer this question using the provided context only.
{question}

Context:
{context}
"""

# Initialize Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
        ("human", message),
    ])

# LangChain

def get_chain():
    return {"context":retriever, "question":RunnablePassthrough()} | prompt_template | model | parser