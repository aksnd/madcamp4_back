from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever

from django.core.management.base import BaseCommand

import os

class Command(BaseCommand):
    help = 'Run LangChain retrieval command'

    def handle(self, *args, **kwargs):
        api_key = os.environ.get('OPENAI_API_KEY')

        # Embedding 설정
        embedding_function = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

        # LLM 설정
        llm = ChatOpenAI(api_key=api_key)

        # VectorDB 불러오기
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

        # 프롬프트 설정
        question_prompt = ChatPromptTemplate.from_template("""context를 활용해서 질문에 대답해줘.

<context>
{context}
</context>

질문: {input}""")

        # Document Chain 생성
        document_chain = create_stuff_documents_chain(llm, question_prompt)

        # Retriever 설정
        retriever = vectordb.as_retriever()
        advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

        # Retrieval Chain 생성
        retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

        # 질의에 대한 응답 생성 및 출력
        response = retrieval_chain.invoke({"input": ""})
        self.stdout.write(self.style.SUCCESS('Query: ' + response['input']))
        self.stdout.write(self.style.SUCCESS('Number of Retrieved Chunks: ' + str(len(response['context']))))
        self.stdout.write(self.style.SUCCESS('Retrieved Chunks: ' + str(response['context'])))
        self.stdout.write(self.style.SUCCESS('Answer: ' + response['answer']))

