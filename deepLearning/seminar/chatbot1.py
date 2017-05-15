def Greeting(sentence):
  #인사 Intent를 캐치하기 위한 키워드들
  GREETING_KEYWORDS = ["ㅎㅇ", "하이", "안녕하세요", "하잉"]

  #후보 답변
  GREETING_RESPONSES = ["ㅎㅇ", "반갑다", "안녕하세요"]

  # 키워드를 캐치하면 답변을 한다
  for word in sentence:
    if word.lower() in GREETING_KEYWORDS:
      return random.choice(GREETING_RESPONSES)
