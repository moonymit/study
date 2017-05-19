#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import konlpy
import nltk

# POS tag a sentence
sentence = u'잘 기억은 안나지만 작년에 여행가서 찍은 사진 검색해줘'
words = konlpy.tag.Twitter().pos(sentence)

# Define a chunk grammar, or chunking rules, then chunk
grammar = "NP: {<N.*>*<Suffix>?}"
"""
NP: {<N.*>*<Suffix>?}   # Noun phrase 명사구
VP: {<V.*>*}            # Verb phrase 동사구
AP: {<A.*>*}            # Adjective phrase 형용사구
  {<DT|JJ>}          # chunk determiners and adjectives
  }<[\.VI].*>+{      # chink any tag beginning with V, I, or .
  <.*>}{<DT>         # split a chunk at a determiner
  <DT|JJ>{}<NN.*>    # merge chunk ending with det/adj
                     # with one starting with a noun
parser = RegexpParser('''
... NP: {<DT>? <JJ>* <NN>*} # NP
... P: {<IN>}           # Preposition
... V: {<V.*>}          # Verb
... PP: {<P> <NP>}      # PP -> P NP
... VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
... ''')
{<N.*>} # 명사(N으로 시작)인 모든 경우
{<P.*>} # 대명사(P로 시작)인 모든 경우
{<DT><JJR>}  # DT와 JJR이 붙어 있는 경우
{<DT><J.*>}  # DT와 J로 시작하는 단어가 붙어 있는 경우
{<DT><JJ>?<NN>} # <DT>와 <NN>사이에 <JJ>가 있든 없든..
"""
parser = nltk.RegexpParser(grammar)
chunks = parser.parse(words)

print("# Print whole tree")
print(chunks.pprint())

print("\n# Print noun phrases only")
for subtree in chunks.subtrees():
    if subtree.label()=='NP':
        print(' '.join((e[0] for e in list(subtree))))
        # print(subtree.pprint())

# Display the chunk tree
chunks.draw()
