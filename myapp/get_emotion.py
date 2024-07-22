
import os
import re
from openai import OpenAI
api_key = os.environ.get('OPENAI_API_KEY')

def calculate_emotion_score(title):
  client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key
  )

  chat_completion = client.chat.completions.create(
  messages=[
    {"role": "system", "content": "당신은 감정 분석 도우미입니다."},
    {"role": "user", "content": f"다음 기사 제목 혹은 내용의 감정을 분석해 주세요:\n\n{title}\n\n전체 감정을 0에서 10까지의 숫자로 표현해 주세요 (0은 매우 부정적, 10은 매우 긍정적입니다). 반드시 숫자만 결과에 있어야 합니다."}
  ],
  model="gpt-3.5-turbo",
  )
  answer = chat_completion.choices[0].message.content
    
    # 정규 표현식을 사용하여 응답에서 숫자만 추출
  emotion_score = re.search(r'\d+', answer)
  if emotion_score:
    return int(emotion_score.group())
  else:
    return 5


def calculate_relevance_score(title, company):
  client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key
  )

  chat_completion = client.chat.completions.create(
  messages=[
    {"role": "system", "content": "당신은 기사 분석 도우미입니다."},
    {"role": "user", "content": f"다음 기사 제목 혹은 내용과 {company} 기업과의 관련성을 분석해 주세요:\n\n{title}\n\n전체 관련성을 0에서 10까지의 숫자로 표현해 주세요 (0은 매우 관련없음, 10은 매우 관련도 깊음). 반드시 숫자만 결과에 있어야 합니다."}
  ],
  model="gpt-3.5-turbo",
  )
  answer = chat_completion.choices[0].message.content
  
  relevance_score = re.search(r'\d+', answer)
  if relevance_score:
    return int(relevance_score.group())
  else:
    return 5

  return answer