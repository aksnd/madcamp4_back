from langchain import hub

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain.llms import HuggingFaceHub

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever

#embedding 설정
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#LLM 설정
llm = HuggingFaceHub(huggingfacehub_api_token='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', repo_id='mistralai/Mistral-7B-Instruct-v0.3') # Mistral-7B-Instruct-v0.3 at Hugging Fac

#vectordb불러오기
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("human", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectordb.as_retriever()
advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)