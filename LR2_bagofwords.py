from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#import cleaning
import timeit
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer as rt
from nltk.corpus import stopwords
from nltk.corpus import words
# from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as rick

from scipy.sparse import csr_matrix,csc_matrix,hstack

def read_data(fName, colName):
    data = pd.read_csv(fName)
    return data[colName]

# Output result of cleaning into a txt file, by the name of fname
# resultlist is a list of lists, the items in the inner list are strings
def print_result(fName, resultlist):
    with open(fName, 'w') as f:
        for l in result:
            f.write("%s\n" % l)
    f.close()

# Dump the preprocessed data into a pickle file
def generate_cleaned_file(instances, labels):
    col = ["Phrases", "Sentiment"]
    i = range(len((instances)))
    dataframe = pd.DataFrame(columns = col, index = i)
    dataframe["Phrases"] = instances
    dataframe["Sentiment"] = labels
    pkfile = open('preprocessed_train_data.pkl', 'wb')
    rick.dump(dataframe, pkfile)
    pkfile.close()

# Open the pickle file
def open_cleaned_file(fileName):
    pkl = open(fileName, 'rb')
    result = rick.load(pkl)
    pkl.close()
    return result

# Tokenize and filter out non-alphanumeric characters.
def tokenize_data(d):
    # Initialize RegExp tokenizer.
    tokenizer = rt(r'\w+')
    # Make all words lowercase.
    d = (phrase.lower() for phrase in d)
    d = (tokenizer.tokenize(phrase) for phrase in d)
    tokenized = []
    for phrase in d:
        tokenized.append(phrase)
    # print_result('cleaned.txt', result)
    return tokenized

def untokenize(texts):
    docs = []
    for doc in texts:
        temp = ""
        for word in doc:
            temp += word + " "
        docs.append(temp)
    return docs

def filter_data(d):
    stop_words = list(stopwords.words('english'))
    # Additional non-sentimental words to filter out
    custom_words = ['genre', 'film', 'movie']
    stop_words = set(stop_words).union(set(custom_words))
    # Second cleaning
    # Initialize dictionary of useful words from nltk corpus
    good_words = set(list(words.words()))
    result = []
    for phrase in d:
        temp = []
        for word in phrase:
            if word in good_words and not word in stop_words:
                temp.append(word)
        result.append(temp)
    # print_result('secondclean.txt', result)
    return result

phrases = read_data("train.csv", "Phrase")
labels = read_data("train.csv", "Sentiment")

cleaned = tokenize_data(phrases)
result = filter_data(cleaned)

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
X_train = vect.fit(untokenize(result)).transform(untokenize(result))
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
