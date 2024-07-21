
import os
from openai import OpenAI
api_key = os.environ.get('OPENAI_API_KEY')

def calculate_exception_score(title):
  client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key
  )

  chat_completion = client.chat.completions.create(
  messages=[
    {"role": "system", "content": "당신은 감정 분석 도우미입니다."},
    {"role": "user", "content": f"다음 기사 제목의 감정을 분석해 주세요:\n\n{title}\n\n전체 감정을 0에서 10까지의 숫자로 표현해 주세요 (0은 매우 부정적, 10은 매우 긍정적입니다). 반드시 숫자만 적어 주세요."}
  ],
  model="gpt-3.5-turbo",
  )
  answer = chat_completion.choices[0].message.content

  print(answer)
  return answer