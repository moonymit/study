# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import codecs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from konlpy.tag import Twitter

import nltk
from matplotlib import font_manager, rc
font_fname = '/Library/Fonts/AppleGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name, size=6)

from collections import namedtuple
from gensim.models import doc2vec

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from scipy import spatial

from pathlib import Path
import random

pos_tagger = Twitter()

def print_kor_list(list):
    print(repr(list).decode('string-escape'))

def tokenize(doc):
    #norm, stem은 optional
    # return [t for t in pos_tagger.nouns(doc)]
    # return ['/'.join(t) for t in pos_tagger.pos(doc)]
    # return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True) if (t[1] == 'Noun' or t[1] == 'Verb')]

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

train_data = read_file('./data/train.tsv')

# row, column 의 수가 제대로 읽혔는지 확인
print("전체 데이터(row) 수: %d " % len(train_data))
print("전체 항목(col) 수: %d" % len(train_data[0]))


train_docs = ''

# 토큰화된 파일이 있는지 확인
train_voca_file = Path('./data/train_voca.txt')
if train_voca_file.is_file():
    train_docs = read_tagging_data('./data/train_voca.txt')
else:
    # SEARCH, SEND 분리하기
    # search_data_docs = [row[1] for row in train_data if row[2] == 'SEARCH']
    # send_data_docs = [row[1] for row in train_data if row[2] == 'SEND']

    # SEARCH, SEND 합쳐서 BOTH 만들기
    # train_data.extend([['검색 및 분류', '{} {}'.format(row1, row2) , 'BOTH'] for row1, row2 in zip(search_data_docs, send_data_docs)])

    # 토큰화 후에 text로 저장
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    write_tagging_data(train_docs, './data/train_voca.txt')

# 데이터 row 뒤섞기
train_docs = random.sample(train_docs, len(train_docs))

################# -- Data exploration (feat.NLTK) -- #################

tokens = [t for d in train_docs for t in d[0]] # 토큰만 모은다.
text = nltk.Text(tokens, name='NMSC')

# print(text)
# print(len(text.tokens)) # 전체 토큰의 갯수
# print_kor_list(len(set(text.tokens))) # 유니크한 토큰의 갯수
# print_kor_list(text.vocab().most_common(10)) # 가장 많이 나온 단어 10개
# print_kor_list(text.vocab().most_common(10))

# text.plot(50) #가장 많이 나오는 상위 50개를 그래프로 출력
# print_kor_list(text.collocations())

######################################################################

######### -- Sentiment classification with term-existance-- ##########
# 최빈도 단어 100개를 피쳐로 사용
selected_words = [f[0] for f in text.vocab().most_common(len(text.tokens))]
# selected_words = [f[0] for f in text.vocab().most_common(40)]

def term_exists(doc):
    return {'exists({})'.format(word):(word in set(doc)) for word in selected_words}

def doc_terms(doc):
    return {'exists({})'.format(word): True for word in doc}

train_xy = [(term_exists(d),c) for d, c in train_docs]
test_xy = [(term_exists(d),c) for d, c in train_docs]
# print_kor_list(train_xy)

# Naive Bayes Classifier 사용
classifier = nltk.NaiveBayesClassifier.train(train_xy)
classifier.show_most_informative_features(10)

# print_kor_list(doc_terms(tokenize('어제 찍은 사진 보여줘')))


test_docs = ['어제 찍은 사진 보여줘', '작년에 내가 나온 사진 검색', '방금 찍은 사진만 보여줘', '꽃사진좀 보자', '바다가 나온 사진 검색',
             '커피 마시는 사진 필요해', '벚꽃 나온 사진 줘봐 있어?', '다른 사진들도 좀 보고싶어',

             '위 사진들 전송', '미쉘에게 사진 전송', '검색된 사진 소피한테 보내줘', 'ㅋㅋ야 사진 TARS한테 보내!!',
             '사진 찾은거 남자친구한테 보내줘', '그거 찰리한테 보내줘',
             '어제 찍은 사진 보여주고 찰리한테 사진 전송',

            #  '어제 찍은 사진 엄마한테 보내줘', '작년에 찍은 벚꽃사진 찾아서 예지한테 보내줘',
            #  '올해 바다에서 내가 나온 사진 골라서 찰리한테 보내줘',

             '대박이는 대박 귀여워']

for doc in test_docs:
    print_kor_list([doc + ' SEARCH', classifier.prob_classify(doc_terms(tokenize(doc))).prob('SEARCH')])
    print_kor_list([doc + ' SEND', classifier.prob_classify(doc_terms(tokenize(doc))).prob('SEND')])
    # print_kor_list([doc + ' SEARCH&SEND', classifier.prob_classify(doc_terms(tokenize(doc))).prob('SEARCH&SEND')])
    # print_kor_list([doc, classifier.classify(doc_terms(tokenize(doc)))])

######################################################################


############# -- Sentiment classification with doc2vec-- #############
# doc_vectorizer = ''
# tagged_train_docs = [doc2vec.LabeledSentence(words=d,tags=[c]) for d, c in train_docs]
#
# train_model_file = Path('./data/train.model')
# if train_model_file.is_file():
#     doc_vectorizer= doc2vec.Doc2Vec.load('./data/train.model')
# else:
#     #사전 구축
#     doc_vectorizer = doc2vec.Doc2Vec(size=400, alpha=0.025, min_alpha=0.025, seed=1234)
#     doc_vectorizer.build_vocab(tagged_train_docs)
#
#     #Train document vectors!
#     for epoch in range(100):
#         doc_vectorizer.train(tagged_train_docs,total_examples=len(train_docs), epochs=100)
#         doc_vectorizer.alpha -= 0.002 #decrease the learning rate
#         doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate no decay
#
#     # To save
#     doc_vectorizer.save('./data/train.model')
#
#
# filename = './data/train_classifier.joblib.pkl'
# train_classifier_file = Path(filename)
# if train_classifier_file.is_file():
#     # To load
#     classifier = joblib.load(filename)
# else:
#     # #마지막 분류 단계. 분류를 위한 피쳐들을 마련 : (새로운) 문장들을 벡터로 표현
#     train_words = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
#     train_tags = [doc.tags[0] for doc in tagged_train_docs]
#
#     # 생성된 벡터들을 이용하여 Classification
#     classifier = LogisticRegression(random_state=1234)
#     classifier.fit(train_words, train_tags)
#
#     # To save
#     _ = joblib.dump(classifier, filename, compress=9)
#
# print(classifier.predict([doc_vectorizer.infer_vector(tokenize('어제 찍은 사진 보여줘'))]))
# print(classifier.predict([doc_vectorizer.infer_vector(tokenize('작년에 내가 나온 사진 검색'))]))
# print(classifier.predict([doc_vectorizer.infer_vector(tokenize('방금 찍은 사진만 보여줘'))]))
#
# print(classifier.predict([doc_vectorizer.infer_vector(tokenize('위 사진들 전송해줘'))]))
# print(classifier.predict([doc_vectorizer.infer_vector(tokenize('찰리한테 사진 전송'))]))
# print(classifier.predict([doc_vectorizer.infer_vector(tokenize('검색된 사진 소피한테 보내줘'))]))

# print_kor_list(doc_vectorizer.most_similar("사진/Noun"))
