from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import timeit
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer as rt
from nltk.corpus import stopwords
from nltk.corpus import words
# from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl
import numpy as np

def read_data(fName, colName):
    data = pd.read_csv(fName)
    return data[colName]



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
    # Initialize stopwords list: remove articles, conjunctions, prepositions, pronouns.
    # stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
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

vect = CountVectorizer(min_df=2, ngram_range=(0, 30))
X_train = vect.fit(untokenize(result)).transform(untokenize(result))
param_grid = {'C': [1]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, labels)

x_test = read_data("testset_1.csv", "Phrase")
test_phraseid = read_data("testset_1.csv", "PhraseId")

cleaned_test = tokenize_data(x_test)
result_test = filter_data(cleaned_test)
X_test = vect.transform((untokenize(result_test)))

lr = grid.best_estimator_
y = lr.predict(X_test)

df = pd.DataFrame({'PhraseId':np.array(test_phraseid),'Sentiment':np.array(y)})
outfile = 'out.csv'
df.to_csv(outfile,index = False)
