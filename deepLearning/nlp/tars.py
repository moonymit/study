# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import nltk
import random
import codecs
import pickle

from nltk.tree import ParentedTree

from pathlib import Path
from matplotlib import font_manager, rc

from konlpy.tag import Kkma
from konlpy.utils import pprint

from konlpy.tag import Twitter
from konlpy.tag import Kkma


# Set Matplot Font
font_fname = '/Library/Fonts/AppleGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name, size=6)

# Pos Tagger
twitter = Twitter()
kkma = Kkma()

def print_kor_list(list):
    print(repr(list).decode('string-escape'))

def tokenize(doc):
    return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True) if (t[1] == 'Noun' or t[1] == 'Verb')]

def read_corpus_file(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] #header 제외
    return data

def read_tagging_data(filename):
    with codecs.open(filename, 'r', encoding='utf-8') as f:
    # with open(filename, 'r',) as f:
        data = [(line.split()[:-1], line.split()[-1] )for line in f.read().splitlines()]
    return data

def write_tagging_data(data, filename):
    with codecs.open(filename, 'w', encoding='utf-8') as f:
    # with open(filename, 'w') as f:
        for doc in data:
            word = doc[0];
            word.append(doc[1])
            f.write(' '.join(word) + '\n')

# Read Train File
train_data = read_corpus_file('./data/train.tsv')

# Data Tokenize
train_docs = ''
train_voca_file = Path('./data/train_voca.txt')
if train_voca_file.is_file():
    train_docs = read_tagging_data('./data/train_voca.txt')
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    write_tagging_data(train_docs, './data/train_voca.txt')

#  Mix Index
train_docs = random.sample(train_docs, len(train_docs))

# Gather Token Text(NLTK)
tokens = [t for d in train_docs for t in d[0]] # 토큰만 모은다.
text = nltk.Text(tokens, name='NMSC')

# Classification with term-existance
selected_words = [f[0] for f in text.vocab().most_common(40)]
# selected_words = [f[0] for f in text.vocab().most_common(len(text.tokens))]

def term_exists(doc):
    return {'exists({})'.format(word):(word in set(doc)) for word in selected_words}

def doc_terms(doc):
    return {'exists({})'.format(word): True for word in doc}


classifier = ''
classifier_filename = './data/tars_classifier.joblib.pkl'
classifier_file = Path(classifier_filename)
if classifier_file.is_file():
    pprint(u'\n# Load Classifer')
    with open(classifier_filename, 'rb') as f:
        classifier = pickle.load(f)
else:
    pprint(u'\n# Save Classifer')
    with open(classifier_filename, 'wb') as f:
        train_xy = [(term_exists(d),c) for d, c in train_docs]
        # test_xy = [(term_exists(d),c) for d, c in train_docs]

        # Use 'Naive Bayes Classifier' for Classification
        classifier = nltk.NaiveBayesClassifier.train(train_xy)
        # classifier.show_most_informative_features(10)

        pickle.dump(classifier, f)


# Classification Test
test_docs = [u'오키나와에서 파이, 닉과 함께 찍은 노란 꽃 사진들만 보여줘',
             u'방금 찾은 사진 소피한테 전송!!',
             u'작년 겨울에 찍은 이쁜 꽃 사진 그리고 구름 사진을 명수 그리고 하하한테 전송해줘',
             u'명수 그리고 하하에게 올해 겨울에 찍은 노란 꽃 사진 그리고 구름 사진 보내줘',
             u'오늘 찍은 꽃 사진, 재석에게 보내줘']

print("\n# Print 'SEARCH' & 'SEND' Classifier")
for doc in test_docs:
    doc_term = doc_terms(tokenize(doc));
    pprint(u'{} : {}'.format(doc + ' SEARCH', classifier.prob_classify(doc_term).prob('SEARCH')))
    pprint(u'{} : {}'.format(doc + ' SEND', classifier.prob_classify(doc_term).prob('SEND')))
    pprint(u'{} => {}'.format(doc, classifier.classify(doc_term)))



# Tagging & Chunking
def printChunks(chunks, label):
    print("\n# Print whole tree")
    print(chunks.pprint())
    print("\n# Print Noun Phrases Only")
    for subtree in chunks.subtrees():
        if subtree.label()== label:
            print(' '.join((e[0] for e in list(subtree))))

# Object, Nouns,
def parseDoc(chunks, pos_doc):

    i_obj_josa = ['에게', '한테']
    d_obj_josa = ['을', '를']

    parse = {'Noun':[], 'Adjective':[], 'NP':[], 'Object':[], 'Tag':[]}
    parse['Noun'] = [ t[0] for t in pos_doc if t[1] == 'Noun' and t[0] != u'사진']
    parse['Adjective'] = [ t[0] for t in pos_doc if t[1] == 'Adjective']

    for idx, node in enumerate(chunks):
        if type(node) == nltk.Tree and node.label()== 'NP':
            pprint(u'label:' + node.label())

            # NP = [leaf[0] for leaf in list(node) if leaf[0] != u'사진' and leaf[1] != 'Conjunction']
            NP = [leaf[0] for leaf in list(node)]
            parse['NP'].append(NP)

            if idx + 1 < len(chunks) \
                and chunks[idx+1][0] in i_obj_josa \
                and chunks[idx+1][1] == 'Josa' :

                pprint(u'Josa in i_obj_josa: {} {}'.format(chunks[idx+1][0], chunks[idx+1][1]))
                parse['Object'] += NP

    parse['Object'] = list(set(parse['Object']) - set([u'그리고']))
    parse['Tag'] += list(set(parse['Noun']) - set(parse['Object']) - set(['전송']))
    return parse

# pos_doc = twitter.pos(test_docs[0], norm=True, stem=True)
pos_doc = twitter.pos(test_docs[3], norm=False, stem=False)
# parser = nltk.RegexpParser("NP: {<Adjective>*<Noun>*}")
parser = nltk.RegexpParser("NP: {(<Adjective>*<Noun>*<Conjunction>*)*}")
chunks = parser.parse(pos_doc)
printChunks(chunks, 'NP')
# chunks.draw()

# Print Pos
print("\n# Print Josa Only")
pos_josa = [ t for t in pos_doc if (t[1] == 'Josa')]
pprint(pos_josa)

# Filter Noun, Adjective
print("\n# Print Noun & Adjective Only")
pos_noun_adj = [ t for t in pos_doc if (t[1] == 'Noun' or t[1] == 'Adjective')]
pprint(pos_noun_adj)

# Parse
print("\n# Print Parsed Dictionary")
parse_doc = parseDoc(chunks, pos_doc)
pprint(parse_doc)
