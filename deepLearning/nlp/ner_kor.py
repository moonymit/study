# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import re
import codecs

import pycrfsuite

def print_kor_list(list):
    print(repr(list).decode('string-escape'))

def print_uni_list(list):
    print(repr(list).decode('unicode_escape'))

def raw2corpus(raw_path, corpus_path):
    raw = codecs.open(raw_path, encoding='utf-8')
    raw_sentences = raw.read().split('\n')
    corpus = codecs.open(corpus_path, 'w', encoding='utf-8')
    sentences = []
    for raw_sentence in raw_sentences:
        if not raw_sentence:
            continue
        text = re.sub(r'(\ )+', ' ', raw_sentence).strip()
        taggeds = []
        for i in range(len(text)):
            if i == 0:
                taggeds.append('{}/B'.format(text[i]))
            elif text[i] != ' ':
                successor = text[i - 1]
                if successor == ' ':
                    taggeds.append('{}/B'.format(text[i]))
                else:
                    taggeds.append('{}/I'.format(text[i]))
        sentences.append(' '.join(taggeds))
    corpus.write('\n'.join(sentences))

def corpus2raw(corpus_path, raw_path):
    corpus = codecs.open(corpus_path, encoding='utf-8')
    corpus_sentences = corpus.read().split('\n')
    raw = codecs.open(raw_path, 'w', encoding='utf-8')
    sentences = []
    for corpus_sentence in corpus_sentences:
        taggeds = corpus_sentence.split(' ')
        text = ''
        len_taggeds = len(taggeds)
        for tagged in taggeds:
            try:
                word, tag = tagged.split('/')
                if word and tag:
                    if tag == 'B':
                        text += ' ' + word
                    else:
                        text += word
            except:
                pass
        sentences.append(text.strip())
    raw.write('\n'.join(sentences))

def corpus2sent(path):
    corpus = codecs.open(path, encoding='utf-8').read()
    raws = corpus.split('\n')
    sentences = []
    for raw in raws:
        tokens = raw.split(' ')
        sentence = []
        for token in tokens:
            try:
                word, tag = token.split('/')
                if word and tag:
                    sentence.append([word, tag])
            except:
                pass
        sentences.append(sentence)
    return sentences

def index2feature(sent, i, offset):
    word, tag = sent[i + offset]
    if offset < 0:
        sign = ''
    else:
        sign = '+'
    return '{}{}:word={}'.format(sign, offset, word)

def word2features(sent, i):
    L = len(sent)
    word, tag = sent[i]
    features = ['bias']
    features.append(index2feature(sent, i, 0))
    if i > 1:
        features.append(index2feature(sent, i, -2))
    if i > 0:
        features.append(index2feature(sent, i, -1))
    else:
        features.append('bos')
    if i < L - 2:
        features.append(index2feature(sent, i, 2))
    if i < L - 1:
        features.append(index2feature(sent, i, 1))
    else:
        features.append('eos')
    return features

def sent2words(sent):
    return [word for word, tag in sent]

def sent2tags(sent):
    return [tag for word, tag in sent]

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# crf_raw_train: 마키 머리카락 먹고싶다, crf_train:마/B 키/I 머/B 리/I 카/I 락/I 먹/B 고/I 싶/I 다/I
raw2corpus('./data/crf_raw_train.txt', './data/crf_train.txt')
raw2corpus('./data/crf_raw_test.txt', './data/crf_test.txt')

# corpus2raw('./data/crf_train.txt', './data/crf_restored.txt')
# sent0 = corpus2sent('./data/crf_train.txt')[0]
# print_uni_list(sent0) #[[u'마', u'B'], [u'키', u'I'], [u'머', u'B'], ...]

train_sents = corpus2sent('./data/crf_train.txt') #[[[u'마', u'B'], [u'키', u'I'], [u'머', u'B'], ...]]
test_sents = corpus2sent('./data/crf_test.txt') #[[[u'츠', u'B'], [u'바', u'I'], [u'사', u'I'], ...]]

train_x = [sent2features(sent) for sent in train_sents]
train_y = [sent2tags(sent) for sent in train_sents]
test_x = [sent2features(sent) for sent in test_sents]
test_y = [sent2tags(sent) for sent in test_sents]
trainer = pycrfsuite.Trainer()
for x, y in zip(train_x, train_y):  # 파이썬2에서 돌렸다
    trainer.append(x, y)
trainer.train('./data/space.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('/data/space.crfsuite')

sent = test_x[0]
print_uni_list(sent) #[[u'bias', u'+0:word=츠', u'bos', u'+2:word=사', u'+1:word=바'], ...]

print("Sentence: ", ' '.join(sent2words(sent)))
print("Correct:  ", ' '.join(sent2tags(sent)))
print("Predicted:", ' '.join(tagger.tag(sent2features(sent))))
