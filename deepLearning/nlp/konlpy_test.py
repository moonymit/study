# -*- coding: utf-8 -*-

from konlpy.tag import Kkma
from konlpy.utils import pprint

kkma = Kkma()
# pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
# pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
# pprint(kkma.pos(u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))
# pprint(kkma.pos(u'오키나와에서 엄마와 함께 찍은 사진들만 보여줘'))

from konlpy.tag import Twitter
twitter = Twitter()
pprint(twitter.morphs(u'오키나와에서 엄마와 함께 찍은 사진들만 보여줘', norm=True, stem=True))
pprint(twitter.pos(u'오키나와에서 엄마와 함께 찍은 사진들만 보여줘', norm=True, stem=True))
pprint(twitter.nouns(u'오키나와에서 엄마와 함께 찍은 사진들만 보여줘'))
pprint(twitter.nouns(u'아버지가방에들어가신다'))
pprint(twitter.nouns(u'작년겨울에 찍은 노란 꽃 사진 보여줘'))

import nltk

def printNP(chunks):
    print("# Print whole tree")
    print(chunks.pprint())
    print("\n# Print noun phrases only")
    for subtree in chunks.subtrees():
        if subtree.label()=='NP':
            print(' '.join((e[0] for e in list(subtree))))
            # print(subtree.pprint())

words = twitter.pos(u'작년겨울에 찍은 노란 꽃 사진 보여줘')
parser = nltk.RegexpParser("NP: {<Adjective>*<Noun>*}")
chunks = parser.parse(words)

printNP(chunks)

chunks.draw()
