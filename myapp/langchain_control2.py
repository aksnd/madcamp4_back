from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever

import os

api_key = os.environ.get('OPENAI_API_KEY')

#embedding 설정
embedding_function = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

#LLM 설정
llm = ChatOpenAI(api_key=api_key)

#vectordb불러오기
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

                                        
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectordb.as_retriever()
advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

response = retrieval_chain.invoke({"input": "한화오션의 미래는 어떨거 같아?"})
print(response['input']) # input (i.e., query)
print(len(response['context'])) # number of retrieved chunks (by default 4)
print(response['context']) # list of retrieved chunks
print(response['answer']) # answer