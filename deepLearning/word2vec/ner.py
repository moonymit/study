#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import konlpy
import nltk

# POS tag a sentence
sentence = u'만 6세 이하의 초등학교 취학 전 자녀를 양육하기 위해서는'
words = konlpy.tag.Twitter().pos(sentence)

# Define a chunk grammar, or chunking rules, then chunk
grammar = "NP: {<N.*>*<Suffix>?}"
"""
NP: {<N.*>*<Suffix>?}   # Noun phrase 명사구
VP: {<V.*>*}            # Verb phrase 동사구
AP: {<A.*>*}            # Adjective phrase 형용사구
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
