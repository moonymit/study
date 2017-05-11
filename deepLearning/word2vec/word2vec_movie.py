# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import codecs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from konlpy.tag import Twitter
from pprint import pprint

import nltk
from matplotlib import font_manager, rc
font_fname = '/Library/Fonts/AppleGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name, size=6)

from collections import namedtuple
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression


pos_tagger = Twitter()

def print_kor_list(list):
    print(repr(list).decode('string-escape'))

def tokenize(doc):
    #norm, stem은 optional
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def read_file(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] #header 제외
    return data

def read_tagging_data(filename):
    # with codecs.open(filename, 'r', encoding='utf-8') as f:
    with open(filename, 'r') as f:
        data = [(line.split()[:-1], line.split()[-1] )for line in f.read().splitlines()]
    return data

def write_tagging_data(data, filename):
    # with codecs.open(filename, 'w', encoding='utf-8') as f:
    with open(filename, 'w') as f:
        for doc in data:
            word = doc[0];
            word.append(doc[1])
            f.write(' '.join(word) + '\n')

train_data = read_file('./nsmc-master/ratings_train.txt')
test_data = read_file('./nsmc-master/ratings_test.txt')

# #row, column 의 수가 제대로 읽혔는지 확인
# print(len(train_data))
# print(len(train_data[0]))
#
# print(len(test_data))
# print(len(test_data[0]))

# 데이터 Tokening(Tagging) 후 txt 파일로 저장 : 최초 한번만 실행
# train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
# test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
#
# write_tagging_data(train_docs, 'train_docs_voca.txt')
# write_tagging_data(test_docs, 'test_docs_voca.txt')

train_docs = read_tagging_data('train_docs_voca.txt') #태깅된 데이터를 불러온다
test_docs = read_tagging_data('test_docs_voca.txt')

# print_kor_list(train_docs)


################# -- Data exploration (feat.NLTK) -- #################

tokens = [t for d in train_docs for t in d[0]] # 토큰만 모은다.
# print(len(tokens))

text = nltk.Text(tokens, name='NMSC')
# print(text)
# print(len(text.tokens)) # 전체 토큰의 갯수
# print(len(set(text.tokens))) # 유니크한 토큰의 갯수
# print(text.vocab().most_common(10)) # 가장 많이 나온 단어 10개
# print_kor_list(text.vocab().most_common(10))

# text.plot(50) #가장 많이 나오는 상위 50개를 그래프로 출력
# print_kor_list(text.collocations())

######################################################################


######### -- Sentiment classification with term-existance-- ##########

#최빈도 단어 2000개를 피쳐로 사용
# selected_words = [f[0] for f in text.vocab().most_common(2000)]
# print_kor_list(selected_words)

# def term_exists(doc):
#     return {'exists({})'.format(word):(word in set(doc)) for word in selected_words}

#시간 단축을 위해 training corpus의 일부만 사용할 수 있음
# train_docs = train_docs[:10000]
# print_kor_list(train_docs)

# train_xy = [(term_exists(d),c) for d, c in train_docs]
# test_xy = [(term_exists(d),c) for d, c in test_docs]

# Naive Bayes Classifier 사용
# classifier = nltk.NaiveBayesClassifier.train(train_xy)
# print_kor_list(nltk.classify.accuracy(classifier, test_xy))

# classifier.show_most_informative_features(10)
######################################################################


############# -- Sentiment classification with doc2vec-- #############
TaggedDocument = namedtuple('TaggedDocument', 'words tags')

#여기서는 15만개 training documents 전부 사용
# tagged_train_docs = [TaggedDocument(d,[c]) for d, c in train_docs]
# tagged_test_docs = [TaggedDocument(d,[c]) for d, c in test_docs]

#사전 구축
# doc_vectorizer = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025, seed=1234)
# doc_vectorizer.build_vocab(tagged_train_docs)

#Train document vectors!
# for epoch in range(10):
#     doc_vectorizer.train(tagged_train_docs,total_examples=len(train_docs), epochs=10)
#     doc_vectorizer.alpha -= 0.002 #decrease the learning rate
#     doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate no decay

#To save
# doc_vectorizer.save('doc2vec.model')


# 위의 과정을 한 번 거쳤다면 load해서 사용
doc_vectorizer= doc2vec.Doc2Vec.load('doc2vec.model')

print_kor_list(doc_vectorizer.most_similar("ㅋㅋ/KoreanParticle"))
print_kor_list(doc_vectorizer.most_similar("공포/Noun"))
print_kor_list(doc_vectorizer.most_similar(positive=['여자/Noun', '왕/Noun'], negative=['남자/Noun']))

#마지막 분류 단계. 분류를 위한 피쳐들을 마련
train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
train_y = [doc.tags[0] for doc in tagged_train_docs]
len(train_x)       # 사실 이 때문에 앞의 term existance와는 공평한 비교는 아닐 수 있다
len(train_x[0])

test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]
len(test_x)
len(test_x[0])

classifier = LogisticRegression(random_state=1234)
classifier.fit(train_x, train_y)
classifier.score(test_x, test_y)
