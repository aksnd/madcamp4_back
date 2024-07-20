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
        prompt = ChatPromptTemplate.from_template("""아래 기사가 출판된 날 이후 주가변동을 주어진 context의 price와 price_change수치를 적극적으로 반영해서, 정확한 price_change 수치로, 소수점 단위의 정확한 숫자로 제시해줘.
                                                  네가 말한 수치가 정답일 필요는 없어.
                                                  이 기사가 나온 시점 주가는 30100원이었어.

<context>
{context}
</context>

기사내용: {input}""")

        # Document Chain 생성
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Retriever 설정
        retriever = vectordb.as_retriever()
        advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

        # Retrieval Chain 생성
        retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

        # 질의에 대한 응답 생성 및 출력
        response = retrieval_chain.invoke({"input": """
            [서울파이낸스 김수현 기자] HD현대중공업·한화오션 등 국내 조선사들이 미국 해군 함정 MRO(유지·보수·정비) 시장 진출을 본격화하고 있다. 골드버그 주한 미국 대사의 국내 사업장 방문으로, 업계는 미국 방산 시장 진출에 한걸음 더 다가갈 것으로 기대를 모은다.

15일 HD한국조선해양에 따르면 필립 S. 골드버그(Philip S. Goldberg) 주한 미국 대사 일행이 당일 HD현대중공업 울산조선소를 찾아 함정사업 분야 상호 협력 방안을 논의했다. 이번 방문은 골드버그 대사가 HD현대중공업의 사업장을 직접 둘러보고 상호 협력 방안을 논의하고 싶다는 의사를 밝히며 성사됐다. 이날 골드버그 대사는 "미국과 HD현대가 상선 및 함정 분야에서 훌륭한 파트너십을 구축할 잠재력이 많다고 생각한다"고 말했다. 업계는 미 해군과의 협업 가능성이 높아졌다고 진단한다.

앞서 HD한국조선해양은 국내 최초로 해군보급체계사령부와 함정정비협약(MSRA) 체결을 알리며 MRO 시장 진출을 알렸다. MSRA는 미 함정의 MRO를 위해 미국 정부와 민간 조선소가 맺는 협약으로, MRO 사업 참여를 위해서는 필수적으로 체결해야 된다. 이 협약을 통해 HD현대중공업은 향후 5년간 미국 해상 수송사령부 소속 지원함과 미 해군 운용 전투함의 MRO 사업 입찰이 가능해졌다.

한화오션은 현지 조선소 인수 전력으로, 미국 시장에 공을 들이고 있다. 한화오션은 지난달 한화시스템과 함께 미국 펠리델피아에 있는 필리 조선소를 지분 100%에 인수했다고 밝혔다. 필리 조선소는 중소형 상선을 건조하는 데 최적화된 도크를 갖고 있으며, 동부 연안 해군기지 3곳과 인접해 MRO 사업 유치에도 유리하다. 이번 인수를 통해 한화오션은 함정 MRO 시장뿐만 아니라 건조 시장까지 공략할 수 있을 것으로 전망된다.

호주의 오스탈 조선소 인수를 위한 노력도 지속하고 있다. 오스탈 조선소는 미국 해군 함정의 납품을 담당하는 주요 방산업체로, 미국 해군 함정을 수주한 경험도 보유하고 있다. 앞서 오스탈은 한화오션의 인수제안을 거절한다고 밝혔지만, 최근 호주 정부가 한화오션에 힘을 실어주며 교섭이 다시 원점으로 돌아간 상황이다.

국내 조선업계가 미국 함정 MRO 시장에 주목하는 것은 미국 단일 시장 규모만으로도 20조원에 달하기 때문이다. 시장조사 업체인 모도 인텔리전스가 추산한 미국 해군 함정 MRO 시장의 규모는 연간 20조원으로 세계 시장의 4분의 1에 해당한다. 여기에 미국이 자국의 함정 유지·보수·정비(MRO) 물량의 일부를 해외에서 수행하는 방안을 검토하고 있다고 알려지자, 국내 기업들이 시장 공략에 나선 것이다.

        """})
        self.stdout.write(self.style.SUCCESS('Query: ' + response['input']))
        self.stdout.write(self.style.SUCCESS('Number of Retrieved Chunks: ' + str(len(response['context']))))
        self.stdout.write(self.style.SUCCESS('Retrieved Chunks: ' + str(response['context'])))
        self.stdout.write(self.style.SUCCESS('Answer: ' + response['answer']))

