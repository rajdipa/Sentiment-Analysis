# Extract extra features. Positive, negative, and the ratio of positive to negative words in each phrase.

import cleaning # cleaning.py contains various functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import timeit
import pandas as pd
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer as rt
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as rick
from scipy.sparse import csr_matrix,csc_matrix,hstack


def demo_liu_hu_lexicon(sentence):

    from nltk.corpus import opinion_lexicon
    from nltk.tokenize import treebank

    #tokenizer = treebank.TreebankWordTokenizer()
    pos_words = 0.1
    neg_words = 0.1
    #tokenized_sent = [word.lower() for word in tokenizer.tokenize(sentence)]

    x = list(range(len(sentence)))
    y = []

    for word in sentence:
        if word in opinion_lexicon.positive():
            pos_words += 1
            #y.append(1)  # positive
        elif word in opinion_lexicon.negative():
            neg_words += 1
            #y.append(-1)  # negative
        else:
            pos_words += 0
            neg_words += 0
    p_n_rat = pos_words/neg_words
    return p_n_rat,pos_words,neg_words

def untokenize(texts):
    docs = []
    for doc in texts:
        temp = ""
        for word in doc:
            temp += word + " "
        docs.append(temp)
    return docs


phrases = cleaning.read_data("train.csv", "Phrase")
labels = cleaning.read_data("train.csv", "Sentiment")
cleaned = cleaning.tokenize_data(phrases)
result = cleaning.filter_data(cleaned)

ratio_feat = []
pos_feat = []
neg_feat = []
for phrase in ((result[:2000])):
    rat,pos,neg = demo_liu_hu_lexicon((phrase))
    ratio_feat.append(rat)
    pos_feat.append(pos)
    neg_feat.append(neg)
ratio_feat = np.array(ratio_feat)
pos_feat = np.array(pos_feat)
neg_feat = np.array(neg_feat)
rat_col = csr_matrix((ratio_feat), shape=(1,2000)).toarray()
pos_col = csr_matrix((pos_feat), shape=(1,2000)).toarray()
neg_col = csr_matrix((neg_feat), shape=(1,2000)).toarray()

new_rat = rat_col.reshape(2000,1)
new_pos = pos_col.reshape(2000,1)
new_neg = neg_col.reshape(2000,1)

vect = CountVectorizer(min_df=2, ngram_range=(0, 30))
X_train = vect.fit(untokenize(result[:2000])).transform(untokenize(result[:2000]))
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))

new_X_train  = hstack([X_train,new_rat,new_pos,new_neg]).toarray()

new_X_train.shape

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(new_X_train, labels)

GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid={'C': [0.001, 0.01, 0.1, 1, 10]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, labels)

GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid={'C': [0.001, 0.01, 0.1, 1, 10]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
